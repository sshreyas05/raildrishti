"""
admin_dashboard.py
==================
DASHBOARD 1 – AUTHORITIES / ADMIN CONTROL ROOM

Features:
  A1. Most congested stations
  A2. Most congested rail corridors
  A3. Future congestion risk (ML prediction)
  A4. Cascading delay detection across routes
  A5. Rerouting recommendations for trains
  A6. Priority clearance recommendations
  A7. Zone/division operational summary
  A8. Network bottleneck detection
  A9. Full operational summary
"""

import datetime
import time
import math
import sys

import pandas as pd
import numpy as np

from utils import (
    C, clear_screen, term_width, now_str, box, thin_box, render_table,
    gauge_bar, congestion_badge, risk_badge, delay_color,
    print_banner, print_section, spinner_wait, prompt, error, info, success
)
from models import (
    compute_station_congestion, compute_route_congestion,
    compute_zone_summary, detect_bottlenecks, detect_cascading_delays,
    get_rerouting_options, DelayPredictor, CongestionClassifier
)


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


# ─── A1: Congested Stations ───────────────────────────────────────────────────
def section_congested_stations(station_cong: pd.DataFrame, top_n: int = 15):
    print_section("A1 │ MOST CONGESTED STATIONS", C.B_RED)
    top = station_cong.head(top_n).copy()
    max_score = float(top["congestion_score"].max()) or 1.0

    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        bar    = gauge_bar(r["congestion_score"], max_score, 16)
        badge  = congestion_badge("High" if rank <= 5 else ("Medium" if rank <= 10 else "Low"))
        rows.append([
            f"{C.BOLD}{rank:>2}{C.RESET}",
            f"{C.B_WHITE}{r['station_code']}{C.RESET}",
            f"{C.CYAN}{str(r['station_name'])[:22]}{C.RESET}",
            f"{C.DIM}{str(r.get('zone',''))}{C.RESET}",
            f"{C.DIM}{str(r.get('state',''))[:12]}{C.RESET}",
            f"{r['trains_stopping']:>5,.0f}",
            f"{r['routes_through']:>5,.0f}",
            bar,
            badge,
        ])

    print(render_table(
        ["#", "Code", "Station Name", "Zone", "State", "Trains", "Routes", "Congestion Score", "Status"],
        rows,
        col_colors=[C.YELLOW, C.B_CYAN, C.WHITE, C.DIM, C.DIM, C.B_GREEN, C.BLUE, C.WHITE, C.WHITE],
        max_col_width=25,
    ))
    print()
    top5 = top.head(5)
    alert_lines = []
    for _, r in top5.iterrows():
        alert_lines.append(
            f"  {C.B_RED}⚠  CRITICAL:{C.RESET}  {C.BOLD}{r['station_code']}{C.RESET} "
            f"({r['station_name']}) — {int(r['trains_stopping'])} trains | "
            f"Score {r['congestion_score']:.1f}"
        )
    print(box("🚨  TOP 5 CONGESTION ALERTS", alert_lines, color=C.B_RED, title_color=C.B_WHITE))
    print()


# ─── A2: Congested Corridors ──────────────────────────────────────────────────
def section_congested_corridors(route_cong: pd.DataFrame, top_n: int = 12):
    print_section("A2 │ MOST CONGESTED RAIL CORRIDORS", C.B_YELLOW)
    top = route_cong.head(top_n).copy()
    max_score = float(top["route_score"].max()) or 1.0

    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        bar   = gauge_bar(r["route_score"], max_score, 14)
        level = "High" if rank <= 4 else ("Medium" if rank <= 8 else "Low")
        rows.append([
            f"{C.BOLD}{rank:>2}{C.RESET}",
            f"{C.B_YELLOW}{str(r['route_corridor'])[:30]}{C.RESET}",
            f"{r['trains']:>5,.0f}",
            f"{r['stations']:>4,.0f}",
            f"{r.get('total_km', 0):>6,.0f}",
            bar,
            congestion_badge(level),
        ])

    print(render_table(
        ["#", "Corridor (Src→Dst)", "Trains", "Stns", "Max KM", "Load", "Level"],
        rows,
        max_col_width=32,
    ))
    print()


# ─── A3: Future Congestion Risk ───────────────────────────────────────────────
def section_future_congestion_risk(
    delay_predictor: DelayPredictor,
    cong_clf: CongestionClassifier,
    delay_df: pd.DataFrame,
):
    print_section("A3 │ FUTURE CONGESTION RISK PREDICTION (ML)", C.B_MAGENTA)

    dow   = _current_dow()
    tod   = _current_time_bucket()
    now   = datetime.datetime.now()

    # Predict for next 3 time windows
    windows = [
        ("Now",         tod,            dow),
        ("+2 Hours",    "Evening",      dow),
        ("Tomorrow AM", "Morning",      (now + datetime.timedelta(days=1)).strftime("%A")),
    ]

    scenarios = [
        ("Express",   300, "Clear"),
        ("Superfast", 500, "Clear"),
        ("Local",     80,  "Rainy"),
        ("Express",   300, "Foggy"),
    ]

    pred_lines = []
    for window_label, t_tod, t_dow in windows:
        pred_lines.append(
            f"\n  {C.BOLD}{C.B_CYAN}  ⏱  Window: {window_label}  ({t_dow} / {t_tod}){C.RESET}"
        )
        pred_lines.append("")
        for train_type, dist, weather in scenarios:
            delay = delay_predictor.predict(dist, weather, t_dow, t_tod, train_type)
            cong  = cong_clf.predict(dist, weather, t_dow, t_tod, train_type)
            prob  = cong_clf.predict_proba(dist, weather, t_dow, t_tod, train_type)
            high_p = prob.get("High", 0)

            bar = gauge_bar(delay, 60, 12)
            pred_lines.append(
                f"    {C.B_WHITE}{train_type:<12}{C.RESET}  "
                f"{weather:<7}  "
                f"Delay: {delay_color(delay):<30}  "
                f"Congestion: {congestion_badge(cong)}  "
                f"High-risk prob: {C.B_RED}{high_p:.0%}{C.RESET}"
            )
        pred_lines.append("")

    print(box("🔮  ML CONGESTION PREDICTIONS", pred_lines, color=C.B_MAGENTA))
    print()


# ─── A4: Cascading Delays ─────────────────────────────────────────────────────
def section_cascading_delays(cascade_df: pd.DataFrame, top_n: int = 10):
    print_section("A4 │ CASCADING DELAY DETECTION", C.B_RED)

    high_risk = cascade_df[cascade_df["cascade_risk"] == "High"].head(top_n)
    med_risk  = cascade_df[cascade_df["cascade_risk"] == "Medium"].head(5)

    rows = []
    for _, r in high_risk.iterrows():
        rows.append([
            f"{C.B_YELLOW}{str(r['corridor'])[:30]}{C.RESET}",
            f"{r['distance_km']:>6.0f} km",
            delay_color(r["base_delay_min"]),
            delay_color(r["cascade_delay_min"]),
            congestion_badge(r["cascade_risk"]),
        ])

    print(f"  {C.B_RED}⚡  {len(high_risk)} corridors at HIGH cascade risk right now{C.RESET}\n")
    print(render_table(
        ["Corridor", "Distance", "Base Delay", "Cascade Delay", "Risk"],
        rows, max_col_width=32,
    ))
    print()

    warn_lines = []
    for _, r in high_risk.head(3).iterrows():
        warn_lines.append(
            f"  {C.B_RED}▶{C.RESET}  Corridor {C.BOLD}{r['corridor']}{C.RESET}: "
            f"Initial {r['base_delay_min']:.0f} min delay may cascade to "
            f"{C.B_RED}{r['cascade_delay_min']:.0f} min{C.RESET} downstream"
        )
    if not warn_lines:
        warn_lines = [f"  {C.B_GREEN}✓  No critical cascade chains detected at this time.{C.RESET}"]
    print(box("⚡  CASCADE CHAIN ALERTS", warn_lines, color=C.B_RED))
    print()


# ─── A5: Rerouting Recommendations ───────────────────────────────────────────
def section_rerouting(master: pd.DataFrame, congested_trains: list[str], top_n: int = 5):
    print_section("A5 │ REROUTING RECOMMENDATIONS", C.B_CYAN)

    lines_shown = 0
    reroute_count = 0

    for train_no in congested_trains[:top_n]:
        train_info = master[master["train_no"] == str(train_no)]
        if train_info.empty:
            continue
        t_name    = train_info["train_name"].iloc[0]
        t_src     = train_info["source_code"].dropna().iloc[0] if not train_info["source_code"].dropna().empty else "?"
        t_dst     = train_info["dest_code"].dropna().iloc[0]   if not train_info["dest_code"].dropna().empty else "?"
        n_stops   = len(train_info)

        # Get alternatives
        from models import get_rerouting_options
        alts = get_rerouting_options(train_no, master)

        section_lines = []
        section_lines.append(
            f"  {C.B_RED}🚂  TRAIN {train_no}{C.RESET}  {C.BOLD}{t_name}{C.RESET}  "
            f"[{t_src} → {t_dst}]  {n_stops} stops"
        )
        section_lines.append(f"  {C.B_RED}⚠   This train is on a CONGESTED corridor. "
                             f"Rerouting recommended.{C.RESET}")
        section_lines.append("")

        if alts:
            section_lines.append(f"  {C.B_GREEN}✦  ALTERNATIVE ROUTES AVAILABLE:{C.RESET}")
            for i, alt in enumerate(alts[:3], 1):
                section_lines.append(
                    f"    {C.BOLD}{i}.{C.RESET}  Train {C.B_CYAN}{alt['alt_train_no']}{C.RESET}  "
                    f"{alt['alt_train_name'][:22]}  "
                    f"[{alt['alt_source']} → {alt['alt_dest']}]  "
                    f"{C.B_GREEN}{alt['shared_stations']} shared stations{C.RESET}"
                )
            section_lines.append("")
            section_lines.append(
                f"  {C.B_YELLOW}📋  ACTION:{C.RESET}  Divert train {train_no} via alternative "
                f"Train {alts[0]['alt_train_no']} route. "
                f"Estimated delay savings: {C.B_GREEN}15–25 min{C.RESET}"
            )
            reroute_count += 1
        else:
            section_lines.append(f"  {C.DIM}No viable alternative route found. "
                                 f"Request track clearance instead.{C.RESET}")

        print(thin_box(f"TRAIN {train_no}  REROUTE ANALYSIS", section_lines, color=C.B_CYAN))
        print()
        lines_shown += 1

    if reroute_count == 0:
        print(f"  {C.B_GREEN}✓  No critical rerouting required at this time.{C.RESET}\n")
    else:
        success(f"{reroute_count} trains flagged for rerouting action.")


# ─── A6: Priority Clearance ───────────────────────────────────────────────────
def section_priority_clearance(master: pd.DataFrame, station_cong: pd.DataFrame):
    print_section("A6 │ PRIORITY CLEARANCE RECOMMENDATIONS", C.ORANGE)

    # Priority order: Rajdhani > Shatabdi > Duronto > Mail/Express > Superfast > Local
    priority_keywords = {
        "RAJDHANI": 1, "SHATABDI": 2, "DURONTO": 3,
        "GARIB RATH": 4, "MAIL": 5, "EXPRESS": 6,
        "SUPERFAST": 7, "JAN": 8, "PASSENGER": 9, "LOCAL": 10,
    }

    def get_priority(name: str) -> int:
        name_upper = str(name).upper()
        for kw, p in priority_keywords.items():
            if kw in name_upper:
                return p
        return 6  # default Express

    top_stations = station_cong.head(5)["station_code"].tolist()
    at_congested = master[master["station_code"].isin(top_stations)].copy()

    if at_congested.empty:
        print(f"  {C.DIM}No trains at congested stations to evaluate.{C.RESET}\n")
        return

    sample = at_congested.drop_duplicates("train_no").head(20).copy()
    sample["priority_score"] = sample["train_name"].apply(get_priority)
    sample.sort_values("priority_score", inplace=True)

    rows = []
    for rank, (_, r) in enumerate(sample.head(10).iterrows(), 1):
        p = int(r["priority_score"])
        if   p <= 2: badge = f"{C.BG_RED}{C.WHITE} P1-IMMEDIATE {C.RESET}"
        elif p <= 4: badge = f"{C.BG_YELLOW}{C.BLACK} P2-URGENT    {C.RESET}"
        elif p <= 6: badge = f"{C.BG_BLUE}{C.WHITE} P3-EXPRESS   {C.RESET}"
        else:        badge = f"{C.DIM} P4-STANDARD  {C.RESET}"

        rows.append([
            f"{C.BOLD}{rank:>2}{C.RESET}",
            f"{C.B_CYAN}{r['train_no']}{C.RESET}",
            f"{str(r['train_name'])[:22]}",
            f"{r['station_code']}",
            badge,
            f"{C.B_GREEN}CLEAR PLATFORM IMMEDIATELY{C.RESET}" if p <= 2 else
            f"{C.B_YELLOW}EXPEDITE DEPARTURE{C.RESET}" if p <= 5 else
            f"{C.DIM}STANDARD CLEARANCE{C.RESET}",
        ])

    print(render_table(
        ["#", "Train No", "Train Name", "Station", "Priority", "Action Required"],
        rows, max_col_width=30,
    ))
    print()
    print(box(
        "📋  CLEARANCE PROTOCOL",
        [
            f"  {C.B_RED}P1 – Rajdhani/Shatabdi:{C.RESET}  All others halt. Immediate green signal.",
            f"  {C.B_YELLOW}P2 – Duronto/Mail:{C.RESET}      Clear track within 5 minutes.",
            f"  {C.B_BLUE}P3 – Express:{C.RESET}           Schedule clearance within 15 minutes.",
            f"  {C.DIM}P4 – Local/Passenger:{C.RESET}  Normal queue. Hold if higher priority inbound.",
        ],
        color=C.ORANGE,
    ))
    print()


# ─── A7: Zone Summary ─────────────────────────────────────────────────────────
def section_zone_summary(zone_df: pd.DataFrame):
    print_section("A7 │ ZONE / DIVISION OPERATIONAL SUMMARY", C.B_GREEN)

    max_trains = float(zone_df["trains"].max()) or 1

    rows = []
    for rank, (_, r) in enumerate(zone_df.iterrows(), 1):
        bar = gauge_bar(r["trains"], max_trains, 14)
        rows.append([
            f"{C.BOLD}{rank:>2}{C.RESET}",
            f"{C.B_YELLOW}{str(r['zone'])}{C.RESET}",
            f"{r['trains']:>6,.0f}",
            f"{r['stations']:>6,.0f}",
            f"{r.get('states', 0):>3,.0f}",
            bar,
        ])

    print(render_table(
        ["#", "Zone", "Trains", "Stations", "States", "Load"],
        rows, max_col_width=25,
    ))
    print()


# ─── A8: Bottleneck Detection ─────────────────────────────────────────────────
def section_bottlenecks(bottleneck_df: pd.DataFrame):
    print_section("A8 │ NETWORK BOTTLENECK DETECTION", C.B_RED)

    print(f"  {C.B_RED}⚠  {len(bottleneck_df)} bottleneck nodes detected "
          f"(top 10% congestion threshold){C.RESET}\n")

    rows = []
    for _, r in bottleneck_df.head(12).iterrows():
        level = str(r.get("bottleneck_level", "High"))
        rows.append([
            f"{C.B_RED}{r['station_code']}{C.RESET}",
            f"{str(r['station_name'])[:22]}",
            f"{str(r.get('zone','')):<6}",
            f"{r['trains_stopping']:>5,.0f}",
            f"{r['routes_through']:>4,.0f}",
            f"{r['congestion_score']:>8.1f}",
            congestion_badge(level if level != "nan" else "High"),
        ])

    print(render_table(
        ["Code", "Station", "Zone", "Trains", "Routes", "Score", "Severity"],
        rows, max_col_width=25,
    ))

    print()
    lines = [
        f"  {C.B_RED}BOTTLENECK IMPACT:{C.RESET}  Each of these stations acts as a single point",
        f"  of failure. A delay here propagates to ALL {C.BOLD}connecting trains.{C.RESET}",
        "",
        f"  {C.B_YELLOW}RECOMMENDED ACTIONS:{C.RESET}",
        f"    1. Deploy additional signal staff at EXTREME nodes",
        f"    2. Stagger train arrivals during peak hours",
        f"    3. Activate bypass lines where available",
        f"    4. Issue rolling advisories to train masters",
    ]
    print(box("🔴  BOTTLENECK ADVISORY", lines, color=C.B_RED))
    print()


# ─── A9: Full Operational Summary ────────────────────────────────────────────
def section_operational_summary(
    master: pd.DataFrame,
    stations: pd.DataFrame,
    station_cong: pd.DataFrame,
    route_cong: pd.DataFrame,
    zone_df: pd.DataFrame,
):
    print_section("A9 │ OPERATIONAL SUMMARY DASHBOARD", C.B_GREEN)

    total_trains   = master["train_no"].nunique()
    total_stations = master["station_code"].nunique()
    total_routes   = master["route_corridor"].nunique()
    total_zones    = zone_df["zone"].nunique()
    total_rows     = len(master)
    crit_stations  = len(station_cong[station_cong["congestion_score"] >= station_cong["congestion_score"].quantile(0.9)])
    crit_routes    = len(route_cong[route_cong["route_score"] >= route_cong["route_score"].quantile(0.9)])

    w = term_width()
    col_w = (w - 6) // 3

    def stat_block(label: str, value: str, color: str = C.B_CYAN) -> str:
        pad = col_w - len(label) - len(value) - 6
        return f"  {color}{C.BOLD}{value}{C.RESET}  {C.DIM}{label}{C.RESET}{' ' * max(pad, 0)}"

    print(f"\n{'─' * w}")
    print(
        stat_block("TOTAL TRAINS",    f"{total_trains:,}", C.B_GREEN) +
        stat_block("TOTAL STATIONS",  f"{total_stations:,}", C.B_CYAN) +
        stat_block("ACTIVE ROUTES",   f"{total_routes:,}", C.B_YELLOW)
    )
    print(
        stat_block("RAILWAY ZONES",   f"{total_zones:,}", C.B_MAGENTA) +
        stat_block("CRITICAL STATIONS", f"{crit_stations:,}", C.B_RED) +
        stat_block("CRITICAL ROUTES",   f"{crit_routes:,}", C.B_RED)
    )
    print(f"{'─' * w}\n")

    busiest_station = station_cong.iloc[0]
    busiest_route   = route_cong.iloc[0]
    busiest_zone    = zone_df.iloc[0]

    summary_lines = [
        f"  {C.B_YELLOW}🕐  Report Generated:{C.RESET}  {now_str()}",
        "",
        f"  {C.B_RED}🏭  Busiest Station:{C.RESET}  "
        f"{C.BOLD}{busiest_station['station_code']}{C.RESET}  "
        f"({busiest_station['station_name']})  —  "
        f"{int(busiest_station['trains_stopping'])} trains  |  Score {busiest_station['congestion_score']:.1f}",

        f"  {C.B_RED}🛤️   Busiest Corridor:{C.RESET}  "
        f"{C.BOLD}{busiest_route['route_corridor']}{C.RESET}  —  "
        f"{int(busiest_route['trains'])} trains  |  Score {busiest_route['route_score']:.1f}",

        f"  {C.B_GREEN}🗺️   Busiest Zone:{C.RESET}  "
        f"{C.BOLD}{busiest_zone['zone']}{C.RESET}  —  "
        f"{int(busiest_zone['trains'])} trains  |  {int(busiest_zone['stations'])} stations",

        "",
        f"  {C.DIM}Network Health Index:{C.RESET}  "
        + gauge_bar(max(0, 100 - crit_stations * 2), 100, 20)
        + f"  {max(0, 100 - crit_stations * 2):.0f}/100",
    ]
    print(box("📊  LIVE OPERATIONAL SNAPSHOT", summary_lines, color=C.B_GREEN))
    print()


# ─── Main Admin Dashboard Entry Point ────────────────────────────────────────
def run_admin_dashboard(data: dict, delay_predictor, cong_clf):
    """
    Main entry point called by main.py.
    data: dict returned by data_loader.load_all()
    """
    master       = data["master"]
    stations     = data["stations"]
    delay_df     = data["delay_data"]

    clear_screen()
    print_banner(
        "INDIAN RAILWAYS INTELLIGENCE SYSTEM  —  ADMIN CONTROL ROOM",
        f"Live Network Analysis  |  {now_str()}  |  Serving {master['train_no'].nunique():,} trains",
        color=C.B_RED,
    )

    # Pre-compute analytics
    spinner_wait("Computing station congestion …")
    station_cong = compute_station_congestion(master)

    spinner_wait("Computing corridor congestion …")
    route_cong = compute_route_congestion(master)

    spinner_wait("Computing zone summary …")
    zone_df = compute_zone_summary(master)

    spinner_wait("Detecting bottlenecks …")
    bottleneck_df = detect_bottlenecks(station_cong, top_n=15)

    spinner_wait("Detecting cascading delays …")
    cascade_df = detect_cascading_delays(master, delay_predictor)

    # Congested trains = trains that stop at top 5 congested stations
    top5_stations  = station_cong.head(5)["station_code"].tolist()
    congested_trains = (
        master[master["station_code"].isin(top5_stations)]["train_no"]
        .value_counts()
        .head(8)
        .index.tolist()
    )

    success("All analytics computed. Rendering dashboard …")
    time.sleep(0.5)

    # ─── Render all sections ────────────────────────────────────
    while True:
        clear_screen()
        print_banner(
            "ADMIN CONTROL ROOM  —  INDIAN RAILWAYS INTELLIGENCE SYSTEM",
            f"Last refresh: {now_str()}",
            color=C.B_RED,
        )

        print(f"\n  {C.BOLD}{C.B_CYAN}SELECT VIEW:{C.RESET}")
        menu = [
            ("1", "Congested Stations",          C.B_RED),
            ("2", "Congested Corridors",          C.B_YELLOW),
            ("3", "Future Congestion (ML)",        C.B_MAGENTA),
            ("4", "Cascading Delay Detection",    C.B_RED),
            ("5", "Rerouting Recommendations",    C.B_CYAN),
            ("6", "Priority Clearance",           C.ORANGE),
            ("7", "Zone / Division Summary",      C.B_GREEN),
            ("8", "Network Bottlenecks",          C.B_RED),
            ("9", "Full Operational Summary",     C.B_GREEN),
            ("0", "⬅  Back to Main Menu",         C.DIM),
        ]
        for key, label, color in menu:
            print(f"    {C.BOLD}{C.B_WHITE}[{key}]{C.RESET}  {color}{label}{C.RESET}")

        choice = prompt("Enter option (1-9 / 0 to exit): ").strip()

        if choice == "0":
            break
        elif choice == "1":
            clear_screen()
            section_congested_stations(station_cong)
        elif choice == "2":
            clear_screen()
            section_congested_corridors(route_cong)
        elif choice == "3":
            clear_screen()
            section_future_congestion_risk(delay_predictor, cong_clf, delay_df)
        elif choice == "4":
            clear_screen()
            section_cascading_delays(cascade_df)
        elif choice == "5":
            clear_screen()
            section_rerouting(master, congested_trains)
        elif choice == "6":
            clear_screen()
            section_priority_clearance(master, station_cong)
        elif choice == "7":
            clear_screen()
            section_zone_summary(zone_df)
        elif choice == "8":
            clear_screen()
            section_bottlenecks(bottleneck_df)
        elif choice == "9":
            clear_screen()
            section_operational_summary(master, stations, station_cong, route_cong, zone_df)
        elif choice.upper() == "A":
            # Show all sections sequentially
            clear_screen()
            section_congested_stations(station_cong)
            section_congested_corridors(route_cong)
            section_future_congestion_risk(delay_predictor, cong_clf, delay_df)
            section_cascading_delays(cascade_df)
            section_rerouting(master, congested_trains)
            section_priority_clearance(master, station_cong)
            section_zone_summary(zone_df)
            section_bottlenecks(bottleneck_df)
            section_operational_summary(master, stations, station_cong, route_cong, zone_df)
        else:
            error("Invalid option. Please enter 1-9 or 0.")
            time.sleep(1)
            continue

        prompt("Press ENTER to return to Admin Menu …")
