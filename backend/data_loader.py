"""
data_loader.py
==============
Handles loading, cleaning, and merging all 4 Indian Railways datasets:
  1. stations.json      - GeoJSON with station metadata (zone, state, coords)
  2. schedules.json     - Train schedules (arrival/departure per station)
  3. Train_details CSV  - Train routes with stop sequences and timings
  4. train_delay_data   - Historical delay data with features for ML
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path


# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("RAILWAYS_DATA_DIR", "."))
STATIONS_FILE   = DATA_DIR / "stations.json"
SCHEDULES_FILE  = DATA_DIR / "schedules.json"
TRAIN_DETAILS   = DATA_DIR / "Train_details_22122017.csv"
DELAY_DATA      = DATA_DIR / "train_delay_data_rich.csv"
MODELS_DIR      = DATA_DIR / "models"


def load_stations() -> pd.DataFrame:
    """Load stations.json (GeoJSON FeatureCollection) into a flat DataFrame."""
    with open(STATIONS_FILE, encoding="utf-8") as f:
        geo = json.load(f)

    rows = []
    for feat in geo["features"]:
        props = feat.get("properties", {})
        geom   = feat.get("geometry") or {}
        coords = geom.get("coordinates", [None, None])
        rows.append({
            "station_code": props.get("code", ""),
            "station_name": props.get("name", ""),
            "state":        props.get("state", ""),
            "zone":         props.get("zone", ""),
            "address":      props.get("address", ""),
            "longitude":    coords[0] if len(coords) > 0 else None,
            "latitude":     coords[1] if len(coords) > 1 else None,
        })

    df = pd.DataFrame(rows)
    df["station_code"] = df["station_code"].str.strip().str.upper()
    df.drop_duplicates("station_code", inplace=True)
    return df


def load_schedules() -> pd.DataFrame:
    """Load schedules.json (list of schedule entries) into a DataFrame."""
    with open(SCHEDULES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Normalise column names
    df.rename(columns={
        "train_number": "train_no",
        "train_name":   "train_name",
        "station_code": "station_code",
        "station_name": "station_name",
        "arrival":      "arrival",
        "departure":    "departure",
        "day":          "day",
        "id":           "sched_id",
    }, inplace=True)

    df["station_code"] = df["station_code"].str.strip().str.upper()
    df["train_no"]     = df["train_no"].astype(str).str.strip()

    # Replace string "None" with actual NaN
    df.replace("None", np.nan, inplace=True)
    df.replace("none", np.nan, inplace=True)

    return df


def load_train_details() -> pd.DataFrame:
    """Load Train_details CSV with mixed-type tolerance."""
    df = pd.read_csv(
        TRAIN_DETAILS,
        low_memory=False,
        dtype={
            "Train No":    str,
            "Station Code": str,
            "Distance":    str,
            "SEQ":         str,
        },
    )

    # Standardise column names → snake_case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df.rename(columns={
        "train_no":                   "train_no",
        "train_name":                 "train_name",
        "seq":                        "seq",
        "station_code":               "station_code",
        "station_name":               "station_name",
        "arrival_time":               "arrival_time",
        "departure_time":             "departure_time",
        "distance":                   "distance_km",
        "source_station":             "source_code",
        "source_station_name":        "source_name",
        "destination_station":        "dest_code",
        "destination_station_name":   "dest_name",
    }, inplace=True)

    df["train_no"]      = df["train_no"].astype(str).str.strip()
    df["station_code"]  = df["station_code"].str.strip().str.upper()
    df["distance_km"]   = pd.to_numeric(df["distance_km"], errors="coerce")
    df["seq"]           = pd.to_numeric(df["seq"],          errors="coerce")

    # Drop completely empty rows
    df.dropna(how="all", inplace=True)

    return df


def load_delay_data() -> pd.DataFrame:
    """Load train_delay_data CSV (ML feature table)."""
    df = pd.read_csv(DELAY_DATA)

    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

    # Expected columns after normalisation:
    #   distance_between_stations_km, weather_conditions, day_of_the_week,
    #   time_of_day, train_type, historical_delay_min, route_congestion
    df.rename(columns={
        "distance_between_stations_km": "distance_km",
        "weather_conditions":           "weather",
        "day_of_the_week":              "day_of_week",
        "time_of_day":                  "time_of_day",
        "train_type":                   "train_type",
        "historical_delay_min":         "delay_min",
        "route_congestion":             "congestion",
    }, inplace=True)

    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df["delay_min"]   = pd.to_numeric(df["delay_min"],   errors="coerce")

    # Fill remaining NaN
    df = df.assign(
        weather     = df["weather"].fillna("Clear"),
        day_of_week = df["day_of_week"].fillna("Monday"),
        time_of_day = df["time_of_day"].fillna("Morning"),
        train_type  = df["train_type"].fillna("Express"),
        congestion  = df["congestion"].fillna("Low"),
        delay_min   = df["delay_min"].fillna(df["delay_min"].median()),
        distance_km = df["distance_km"].fillna(df["distance_km"].median()),
    )

    return df


def build_master_dataset(
    stations: pd.DataFrame,
    schedules: pd.DataFrame,
    train_details: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge train_details with station metadata to create a master
    operational dataset used by both dashboards.
    """
    # Start with train_details as the spine
    master = train_details.copy()

    # Attach zone/state from stations
    station_meta = stations[["station_code", "zone", "state", "latitude", "longitude"]].copy()
    master = master.merge(station_meta, on="station_code", how="left")

    # Attach schedule info (use schedules as supplement for arrival/departure if missing)
    sched_slim = schedules[["train_no", "station_code", "arrival", "departure", "day"]].copy()
    sched_slim.rename(columns={"arrival": "sched_arrival", "departure": "sched_departure", "day": "sched_day"}, inplace=True)
    master = master.merge(sched_slim, on=["train_no", "station_code"], how="left")

    # Fill missing arrival/departure from train_details own columns
    master["arrival_time"]    = master["arrival_time"].fillna(master["sched_arrival"])
    master["departure_time"]  = master["departure_time"].fillna(master["sched_departure"])

    # Compute route corridor: source → destination
    master["route_corridor"] = master["source_code"].str.strip() + "→" + master["dest_code"].str.strip()

    # Station traffic volume (proxy: how many trains stop here)
    station_traffic = (
        master.groupby("station_code")["train_no"]
        .nunique()
        .rename("trains_per_station")
        .reset_index()
    )
    master = master.merge(station_traffic, on="station_code", how="left")

    return master


def load_all(verbose: bool = True) -> dict:
    """
    Master loader – call this once from main.py.
    Returns a dict with keys: stations, schedules, train_details, delay_data, master
    """
    if verbose:
        print("  Loading stations.json …")
    stations = load_stations()

    if verbose:
        print("  Loading schedules.json …")
    schedules = load_schedules()

    if verbose:
        print("  Loading Train_details CSV …")
    train_details = load_train_details()

    if verbose:
        print("  Loading train_delay_data CSV …")
    delay_data = load_delay_data()

    if verbose:
        print("  Building master dataset …")
    master = build_master_dataset(stations, schedules, train_details)

    if verbose:
        print(f"  ✓ Master dataset: {len(master):,} rows | {master['train_no'].nunique():,} trains | {master['station_code'].nunique():,} stations")

    return {
        "stations":      stations,
        "schedules":     schedules,
        "train_details": train_details,
        "delay_data":    delay_data,
        "master":        master,
    }
