"""
user_dashboard.py
=================
DASHBOARD 2 – PASSENGER / USER TERMINAL

Input:  Train number (user types it)
Output:
  U1. Current delay (predicted)
  U2. Expected arrival delay at destination
  U3. Predicted arrival time
  U4. Delay reason (context-aware)
  U5. Route status (stop-by-stop)
  U6. Risk level today
  U7. Reliability score
  U8. Alternative trains
"""

import datetime
import time
import math
import sys

import pandas as pd
import numpy as np

from utils import (
    C, clear_screen, term_width, now_str, box, thin_box, render_table,
    gauge_bar, congestion_badge, risk_badge, delay_color, reliability_bar,
    print_banner, print_section, spinner_wait, prompt, error, info, success
)
from models import (
    DelayPredictor, CongestionClassifier,
    get_rerouting_options, get_train_reliability,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _current_time_bucket() -> str:
    h = datetime.datetime.now().hour
    if   h < 5:  return "Late Night"
    elif h < 9:  return "Morning"
    elif h < 12: return "Morning"
    elif h < 17: return "Afternoon"
    elif h < 20: return "Evening"
    else:        return "Night"


def _current_dow() -> str:
    return datetime.datetime.now().strftime("%A")


def _train_type_from_name(name: str) -> str:
    name_up = str(name).upper()
    if "RAJDHANI"  in name_up: return "Rajdhani"
    if "SHATABDI"  in name_up: return "Shatabdi"
    if "DURONTO"   in name_up: return "Duronto"
    if "SUPERFAST" in name_up: return "Superfast"
    if "EXPRESS"   in name_up: return "Express"
    if "LOCAL"     in name_up: return "Local"
    if "PASSENGER" in name_up: return "Passenger"
    return "Express"


def _delay_reason(delay_min: float, weather: str, tod: str, congestion: str) -> list[str]:
    """Generate human-readable delay reasons based on context."""
    reasons = []

    if delay_min < 5:
        reasons.append("No significant delay expected")
        return reasons

    if weather == "Foggy":
        reasons.append(f"🌫️  Foggy conditions reducing visibility — trains running at reduced speed")
    elif weather == "Rainy":
        reasons.append(f"🌧️  Rainfall affecting track adhesion and signal visibility")
    elif weather == "Stormy":
        reasons.append(f"⛈️  Storm advisory active — speed restrictions imposed")

    if congestion == "High":
        reasons.append(f"🔴  Route corridor is HEAVILY congested — multiple trains sharing track")
    elif congestion == "Medium":
        reasons.append(f"🟡  Moderate route congestion — some scheduling pressure")

    if tod in ["Morning", "Evening"]:
        reasons.append(f"⏱️  Peak hour traffic — {tod} rush causing signal queuing")

    if delay_min > 40:
        reasons.append(f"🚨  Severe delay detected — possible track maintenance or prior train delay")
    elif delay_min > 20:
        reasons.append(f"⚠️  Moderate delay — cascading effect from upstream train delays")
    elif delay_min > 10:
        reasons.append(f"ℹ️  Minor delay — operational scheduling buffer")

    if not reasons:
        reasons.append(f"ℹ️  Routine operational delay — no specific cause identified")

    return reasons


def _parse_time(time_str: str) -> datetime.datetime | None:
    """Parse HH:MM:SS or HH:MM into today's datetime."""
    if not time_str or str(time_str).lower() in ("none", "nan", ""):
        return None
    try:
        today = datetime.date.today()
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                t = datetime.datetime.strptime(str(time_str).strip(), fmt).time()
                return datetime.datetime.combine(today, t)
            except ValueError:
                continue
    except Exception:
        pass
    return None


def _add_delay(dt: datetime.datetime | None, minutes: float) -> datetime.datetime | None:
    if dt is None:
        return None
    return dt + datetime.timedelta(minutes=minutes)


def _format_dt(dt: datetime.datetime | None) -> str:
    if dt is None:
        return "N/A"
    return dt.strftime("%H:%M")


# ─── Core: lookup and analyse a train ────────────────────────────────────────
def analyse_train(
    train_no: str,
    master: pd.DataFrame,
    delay_df: pd.DataFrame,
    delay_predictor: DelayPredictor,
    cong_clf: CongestionClassifier,
) -> dict | None:
    """
    Return a full analysis dict for the given train number.
    Returns None if train not found.
    """
    train_rows = master[master["train_no"] == str(train_no).strip()]

    if train_rows.empty:
        # Try partial match
        train_rows = master[master["train_no"].str.contains(str(train_no).strip(), na=False)]

    if train_rows.empty:
        return None

    train_rows = train_rows.sort_values("seq", na_position="last")

    # Basic info
    train_name = str(train_rows["train_name"].iloc[0])
    source     = str(train_rows["source_code"].dropna().iloc[0]) if not train_rows["source_code"].dropna().empty else "?"
    dest       = str(train_rows["dest_code"].dropna().iloc[0])   if not train_rows["dest_code"].dropna().empty else "?"
    source_name = str(train_rows["source_name"].dropna().iloc[0]) if "source_name" in train_rows.columns and not train_rows["source_name"].dropna().empty else source
    dest_name   = str(train_rows["dest_name"].dropna().iloc[0])   if "dest_name" in train_rows.columns   and not train_rows["dest_name"].dropna().empty else dest
    zone        = str(train_rows["zone"].dropna().iloc[0]) if not train_rows["zone"].dropna().empty else "N/A"
    state       = str(train_rows["state"].dropna().iloc[0]) if not train_rows["state"].dropna().empty else "N/A"
    n_stops     = len(train_rows)
    train_type  = _train_type_from_name(train_name)

    # Distance
    dist_km = float(train_rows["distance_km"].max()) if not train_rows["distance_km"].isna().all() else 300.0

    # Context
    dow      = _current_dow()
    tod      = _current_time_bucket()
    weather  = "Clear"     # default; could integrate real weather API

    # ML predictions
    delay_min = delay_predictor.predict(dist_km, weather, dow, tod, train_type)
    congestion = cong_clf.predict(dist_km, weather, dow, tod, train_type)
    cong_proba = cong_clf.predict_proba(dist_km, weather, dow, tod, train_type)

    # Risk level
    if   delay_min > 40 or congestion == "High":    risk = "High"
    elif delay_min > 15 or congestion == "Medium":  risk = "Medium"
    else:                                            risk = "Low"

    # Reliability score
    reliability = get_train_reliability(train_no, master, delay_df)

    # Delay reasons
    reasons = _delay_reason(delay_min, weather, tod, congestion)

    # Stop-by-stop schedule with predicted delay
    stop_schedule = []
    for _, stop in train_rows.head(15).iterrows():
        dep_str  = str(stop.get("departure_time") or stop.get("sched_departure") or "")
        arr_str  = str(stop.get("arrival_time")   or stop.get("sched_arrival")   or "")
        dep_dt   = _parse_time(dep_str)
        arr_dt   = _parse_time(arr_str)
        dep_pred = _add_delay(dep_dt, delay_min)
        arr_pred = _add_delay(arr_dt, delay_min)
        stop_dist = stop.get("distance_km", 0)
        stop_dist = 0 if math.isnan(float(stop_dist)) else float(stop_dist)

        stop_schedule.append({
            "seq":          int(stop.get("seq") or 0),
            "station_code": str(stop.get("station_code", "")),
            "station_name": str(stop.get("station_name", "")),
            "arr_sched":    _format_dt(arr_dt),
            "dep_sched":    _format_dt(dep_dt),
            "arr_pred":     _format_dt(arr_pred),
            "dep_pred":     _format_dt(dep_pred),
            "distance_km":  stop_dist,
        })

    # Destination scheduled arrival
    dest_rows = train_rows[train_rows["station_code"] == dest]
    if not dest_rows.empty:
        dest_arr_str  = str(dest_rows.iloc[-1].get("arrival_time") or dest_rows.iloc[-1].get("sched_arrival") or "")
    else:
        # Use last stop
        last_stop = train_rows.iloc[-1]
        dest_arr_str  = str(last_stop.get("arrival_time") or last_stop.get("sched_arrival") or "")

    dest_arr_dt   = _parse_time(dest_arr_str)
    dest_arr_pred = _add_delay(dest_arr_dt, delay_min)

    # Alternatives
    alts = get_rerouting_options(train_no, master)

    return {
        "train_no":      train_no,
        "train_name":    train_name,
        "train_type":    train_type,
        "source":        source,
        "source_name":   source_name,
        "dest":          dest,
        "dest_name":     dest_name,
        "zone":          zone,
        "state":         state,
        "n_stops":       n_stops,
        "dist_km":       dist_km,
        "dow":           dow,
        "tod":           tod,
        "weather":       weather,
        "delay_min":     delay_min,
        "congestion":    congestion,
        "cong_proba":    cong_proba,
        "risk":          risk,
        "reliability":   reliability,
        "reasons":       reasons,
        "stop_schedule": stop_schedule,
        "dest_arr_sched": _format_dt(dest_arr_dt),
        "dest_arr_pred":  _format_dt(dest_arr_pred),
        "alternatives":  alts,
    }


# ─── Render functions ─────────────────────────────────────────────────────────
def render_train_header(a: dict):
    """Big eye-catching header block for the train."""
    w = term_width()

    delay_str  = delay_color(a["delay_min"])
    risk_str   = risk_badge(a["risk"])
    cong_str   = congestion_badge(a["congestion"])
    rel_str    = reliability_bar(a["reliability"])

    lines = [
        f"  {C.BOLD}{C.B_WHITE}🚆  {a['train_no']}  —  {a['train_name']}{C.RESET}",
        f"  {C.DIM}{a['train_type']}  |  Zone: {a['zone']}  |  {a['state']}{C.RESET}",
        "",
        f"  {C.B_CYAN}📍  Route:{C.RESET}  {C.BOLD}{a['source_name']}{C.RESET}"
        f"  {C.DIM}({a['source']}){C.RESET}  ──▶  "
        f"{C.BOLD}{a['dest_name']}{C.RESET}  {C.DIM}({a['dest']}){C.RESET}",
        f"  {C.DIM}Total Distance: {a['dist_km']:.0f} km  |  {a['n_stops']} scheduled stops{C.RESET}",
        "",
        f"  {C.B_RED}⏱  Current Delay:{C.RESET}             {delay_str}",
        f"  {C.B_YELLOW}🏁  Expected Arrival Delay:{C.RESET}    {delay_str}",
        f"  {C.B_GREEN}🕐  Predicted Arrival Time:{C.RESET}    {C.BOLD}{a['dest_arr_pred']}{C.RESET}"
        f"  {C.DIM}(Scheduled: {a['dest_arr_sched']}){C.RESET}",
        "",
        f"  {C.B_MAGENTA}🌡  Route Congestion:{C.RESET}          {cong_str}",
        f"  {C.B_CYAN}⚡  Risk Level Today:{C.RESET}          {risk_str}",
        f"  {C.B_GREEN}⭐  Reliability Score:{C.RESET}         {rel_str}",
        "",
        f"  {C.DIM}Context: {a['dow']}  |  {a['tod']}  |  Weather: {a['weather']}{C.RESET}",
    ]

    color = {
        "High":   C.B_RED,
        "Medium": C.B_YELLOW,
        "Low":    C.B_GREEN,
    }.get(a["risk"], C.B_CYAN)

    print(box(f"🚆  TRAIN INFORMATION  —  {a['train_no']}", lines, color=color))
    print()


def render_delay_reasons(a: dict):
    """U4: Delay reasons section."""
    print_section("U4 │ DELAY REASONS ANALYSIS", C.B_YELLOW)

    reason_lines = []
    for r in a["reasons"]:
        reason_lines.append(f"  {r}")

    reason_lines.append("")
    reason_lines.append(
        f"  {C.DIM}Congestion probability breakdown:{C.RESET}  "
        f"Low {a['cong_proba']['Low']:.0%}  |  "
        f"Medium {a['cong_proba']['Medium']:.0%}  |  "
        f"High {a['cong_proba']['High']:.0%}"
    )

    print(box("🔍  WHY IS MY TRAIN DELAYED?", reason_lines, color=C.B_YELLOW))
    print()


def render_route_status(a: dict):
    """U5: Stop-by-stop route status."""
    print_section("U5 │ ROUTE STATUS  (STOP-BY-STOP)", C.B_BLUE)

    stops = a["stop_schedule"]
    rows  = []

    now_time = datetime.datetime.now().time()

    for s in stops:
        # Determine if this stop is "passed", "upcoming", or "current"
        dep_dt = None
        try:
            if s["dep_sched"] != "N/A":
                dep_dt = datetime.datetime.strptime(s["dep_sched"], "%H:%M").time()
        except Exception:
            pass

        if dep_dt and dep_dt < now_time:
            status_icon = f"{C.B_GREEN}✓ Passed{C.RESET}"
        elif dep_dt and abs((
            datetime.datetime.combine(datetime.date.today(), dep_dt) -
            datetime.datetime.now()
        ).total_seconds()) < 1800:
            status_icon = f"{C.B_YELLOW}⏳ Soon{C.RESET}"
        else:
            status_icon = f"{C.DIM}○ Upcoming{C.RESET}"

        dist = f"{s['distance_km']:.0f} km" if s["distance_km"] else ""

        rows.append([
            f"{s['seq']:>3}",
            f"{C.B_CYAN}{s['station_code']:<6}{C.RESET}",
            f"{str(s['station_name'])[:20]}",
            f"{s['arr_sched']}",
            f"{s['dep_sched']}",
            f"{C.B_YELLOW}{s['arr_pred']}{C.RESET}" if s["arr_pred"] != "N/A" else "N/A",
            f"{C.B_YELLOW}{s['dep_pred']}{C.RESET}" if s["dep_pred"] != "N/A" else "N/A",
            dist,
            status_icon,
        ])

    print(render_table(
        ["#", "Code", "Station", "Arr(Sch)", "Dep(Sch)", "Arr(Pred)", "Dep(Pred)", "Dist", "Status"],
        rows, max_col_width=22,
    ))
    print(f"\n  {C.DIM}Showing up to 15 stops. All times in HH:MM (24h). "
          f"Pred = Scheduled + {a['delay_min']:.0f} min delay.{C.RESET}\n")


def render_alternatives(a: dict, master: pd.DataFrame):
    """U8: Alternative trains."""
    print_section("U8 │ ALTERNATIVE TRAINS", C.B_GREEN)

    alts = a["alternatives"]

    if not alts:
        print(f"  {C.DIM}No alternative trains found on overlapping routes.{C.RESET}\n")
        return

    rows = []
    for i, alt in enumerate(alts[:5], 1):
        rows.append([
            f"{C.BOLD}{i}{C.RESET}",
            f"{C.B_CYAN}{alt['alt_train_no']}{C.RESET}",
            f"{str(alt['alt_train_name'])[:25]}",
            f"{alt['alt_source']} → {alt['alt_dest']}",
            f"{C.B_GREEN}{alt['shared_stations']} stations{C.RESET}",
            f"{C.B_GREEN}Book Now{C.RESET}",
        ])

    print(render_table(
        ["#", "Train No", "Train Name", "Route", "Overlap", "Action"],
        rows, max_col_width=28,
    ))

    print()
    tip_lines = [
        f"  {C.B_YELLOW}💡  TIP:{C.RESET}  If your train has HIGH congestion risk, consider switching to",
        f"  an alternative that shares key stations along your journey.",
        "",
        f"  {C.B_CYAN}Book alternatives at:{C.RESET}  irctc.co.in  |  NTES App  |  Railway Station Counter",
    ]
    print(box("💡  PASSENGER ADVISORY", tip_lines, color=C.B_GREEN))
    print()


# ─── Main Passenger Dashboard Entry Point ────────────────────────────────────
def run_user_dashboard(data: dict, delay_predictor, cong_clf):
    master   = data["master"]
    delay_df = data["delay_data"]

    while True:
        clear_screen()
        print_banner(
            "PASSENGER INFORMATION TERMINAL  —  INDIAN RAILWAYS",
            "Real-time delay prediction, route status & alternatives",
            color=C.B_BLUE,
        )

        print(f"  {C.B_CYAN}Enter your train number to get live delay predictions & route status.{C.RESET}")
        print(f"  {C.DIM}Example train numbers: 12301, 22222, 47154, 12002, 12951{C.RESET}\n")

        train_no = prompt("Enter Train Number (or 'q' to quit): ").strip()

        if train_no.lower() in ("q", "quit", "exit", "0", ""):
            break

        spinner_wait(f"Fetching data for train {train_no} …")
        result = analyse_train(train_no, master, delay_df, delay_predictor, cong_clf)

        if result is None:
            error(f"Train '{train_no}' not found in database.\n"
                  f"  Please check the train number and try again.\n"
                  f"  {C.DIM}Tip: Try a 5-digit train number (e.g. 12301){C.RESET}")
            prompt("Press ENTER to try again …")
            continue

        clear_screen()
        print_banner(
            f"TRAIN {result['train_no']}  —  {result['train_name']}",
            f"Delay Analysis  |  {now_str()}",
            color={
                "High":   C.B_RED,
                "Medium": C.B_YELLOW,
                "Low":    C.B_GREEN,
            }.get(result["risk"], C.B_BLUE),
        )

        # ── U1-U3, U6-U7: Header block ──────────────────────────
        render_train_header(result)

        # ── U4: Delay Reasons ───────────────────────────────────
        render_delay_reasons(result)

        # ── U5: Route Status ────────────────────────────────────
        render_route_status(result)

        # ── U8: Alternatives ────────────────────────────────────
        render_alternatives(result, master)

        # ── Footer ──────────────────────────────────────────────
        footer_lines = [
            f"  {C.DIM}Data sourced from Indian Railways schedule database & ML model.{C.RESET}",
            f"  {C.DIM}Predictions are estimates based on historical patterns.{C.RESET}",
            f"  {C.DIM}For official information visit: indianrail.gov.in | NTES{C.RESET}",
        ]
        print(thin_box("ℹ️  DISCLAIMER", footer_lines, color=C.DIM))
        print()

        choice = prompt("Search another train? (y / n): ").strip().lower()
        if choice not in ("y", "yes"):
            break
