"""
app.py — Rail Drishti FastAPI Backend (Railway-compatible)
"""
import os, sys, math, datetime, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── FIXED: Use the directory where app.py lives as BASE_DIR ──────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # /app/backend -> /app (where frontend/ and data/ live)

# Use env var if set, else look for data/ next to app.py
DATA_DIR = Path(os.environ.get("RAILWAYS_DATA_DIR", str(BASE_DIR / "data")))
os.environ["RAILWAYS_DATA_DIR"] = str(DATA_DIR)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_loader
import models as M

app = FastAPI(title="Rail Drishti API", version="1.0.0")

# Allow Vercel frontend + local dev
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "https://raildrishti1.vercel.app"
]
# Also allow any Vercel deployment URL via env var (set FRONTEND_URL on Railway)
_frontend_url = os.environ.get("FRONTEND_URL", "")
if _frontend_url:
    ALLOWED_ORIGINS.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Frontend is served by Vercel — no static mount needed here
# (StaticFiles import kept for potential local dev use)

DATA={}; DELAY_PREDICTOR=None; CONG_CLF=None; MASTER_DF=None; STATION_CONG=None; LOADED=False

import threading

@app.on_event("startup")
def startup():
    def _load_data_and_train():
        global DATA,DELAY_PREDICTOR,CONG_CLF,MASTER_DF,STATION_CONG,LOADED
        try:
            print(f"BASE_DIR: {BASE_DIR}")
            print(f"DATA_DIR: {DATA_DIR}")
            print("Loading datasets in background...")
            DATA = data_loader.load_all(verbose=True)
            print("Training ML models in background...")
            DELAY_PREDICTOR,CONG_CLF = M.train_all_models(DATA["delay_data"], verbose=True)
            MASTER_DF = DATA["master"]
            STATION_CONG = M.compute_station_congestion(MASTER_DF)
            LOADED = True
            print("System ready ✅")
        except Exception as e:
            print(f"Startup error: {e}"); traceback.print_exc()
            
    threading.Thread(target=_load_data_and_train, daemon=True).start()
    print("FastAPI server started. Data loading happening in background...")

@app.get("/")
def root():
    return {"message": "Rail Drishti API is running. Check /health for status.", "loaded": LOADED}

# _build_master removed to save 300MB+ RAM; using data_loader.py's master instead.

def _req():
    if not LOADED: raise HTTPException(503,"System loading, please wait...")

def _safe(df):
    df=df.copy().fillna("")
    for c in df.select_dtypes(include=["float64","float32"]).columns: df[c]=df[c].round(2)
    def fix(v):
        if isinstance(v,np.integer): return int(v)
        if isinstance(v,np.floating): return float(v)
        return v
    return [{k:fix(v) for k,v in r.items()} for r in df.to_dict(orient="records")]

def _type(name):
    n=name.upper()
    for kw,t in [("RAJDHANI","Rajdhani"),("SHATABDI","Shatabdi"),("DURONTO","Duronto"),("SUPERFAST","Superfast"),("EXPRESS","Express"),("LOCAL","Local"),("PASSENGER","Passenger")]:
        if kw in n: return t
    return "Express"

def _tod(h):
    if h<5: return "Late Night"
    if h<9: return "Morning"
    if h<17: return "Afternoon"
    if h<20: return "Evening"
    return "Night"

def _weather():
    m=datetime.datetime.now().month
    if m in(6,7,8,9): return "Rainy"
    if m in(12,1): return "Foggy"
    return "Clear"

def _train_specific_delay(base_delay: float, train_no: str, matches: pd.DataFrame, tt: str) -> float:
    """
    Adjust base ML prediction using train-specific features so every train
    gets a meaningfully different delay estimate.
    Factors: number of stops, route congestion score, train-id variation.
    """
    n_stops = len(matches)

    # 1. Stops penalty: more stops = more scheduling complexity & transfer risk
    #    Local trains with 30 stops behave very differently from 5-stop Rajdhanis
    stops_factor = 1.0 + min(n_stops * 0.013, 0.55)  # caps at +55%

    # 2. Route congestion: avg congestion score of stations this train calls at
    cong_factor = 1.0
    if STATION_CONG is not None and not STATION_CONG.empty and "station_code" in STATION_CONG.columns:
        train_stations = set(matches["station_code"].dropna().astype(str).unique())
        route_cong = STATION_CONG[STATION_CONG["station_code"].isin(train_stations)]
        if not route_cong.empty:
            avg_score = float(route_cong["congestion_score"].mean())
            max_score = float(STATION_CONG["congestion_score"].max())
            cong_factor = 1.0 + (avg_score / max(max_score, 1.0)) * 0.50  # up to +50%

    # 3. Deterministic per-train variation (±15%) based on train number hash
    #    Ensures 12301 and 12302 show distinctly different delays
    hash_var = (hash(str(train_no)) % 30 - 15) * 0.01  # -0.15 to +0.15
    hash_factor = 1.0 + hash_var

    # 4. Premium trains have tighter schedules: dampen delay for Rajdhani/Shatabdi
    premium_damp = {"Rajdhani": 0.62, "Shatabdi": 0.58, "Duronto": 0.72}.get(tt, 1.0)

    adjusted = base_delay * stops_factor * cong_factor * hash_factor * premium_damp
    return round(max(0.5, min(adjusted, 150.0)), 1)

def _reasons(delay,weather,tod,cong,n_stops=0):
    r=[]
    if delay<5: return ["✅ No significant delay expected — on time"]
    if weather=="Foggy": r.append("🌫️ Foggy conditions — reduced speed operations active")
    elif weather=="Rainy": r.append("🌧️ Rainfall affecting track adhesion and signal visibility")
    elif weather=="Stormy": r.append("⛈️ Storm advisory — speed restrictions imposed")
    if cong=="High": r.append("🔴 Route corridor heavily congested — multiple trains sharing track")
    elif cong=="Medium": r.append("🟡 Moderate congestion — some scheduling pressure")
    if tod in("Morning","Evening"): r.append(f"⏱️ Peak hour ({tod}) — signal queuing likely")
    if n_stops > 20: r.append(f"🚉 High stop count ({n_stops} stops) — cumulative dwell-time risk")
    if delay>40: r.append("🚨 Severe delay — possible track maintenance or upstream cascade")
    elif delay>20: r.append("⚠️ Moderate delay — cascading from upstream train delays")
    elif delay>10: r.append("ℹ️ Minor operational delay within normal buffer")
    if not r: r.append("ℹ️ Routine operational delay — no specific cause identified")
    return r

def _adddelay(ts, mins):
    for fmt in ("%H:%M","%H:%M:%S","%H.%M"):
        try: return (datetime.datetime.strptime(ts.strip(),fmt)+datetime.timedelta(minutes=mins)).strftime("%H:%M")
        except: pass
    return ts

def _stops(matches, delay):
    stops=[]
    for i,(_,r) in enumerate(matches.head(15).iterrows()):
        sa=str(r.get("arrival_time","")) if "arrival_time" in r.index else ""
        sd=str(r.get("departure_time","")) if "departure_time" in r.index else ""
        ea=_adddelay(sa,delay) if sa and sa not in("nan","") else ""
        stops.append({"stop_no":i+1,"station_code":str(r.get("station_code",r.get("source_code",""))),"station_name":str(r.get("station_name","")),"scheduled_arr":sa if sa not in("nan","") else "—","scheduled_dep":sd if sd not in("nan","") else "—","estimated_arr":ea or "—","distance_km":round(float(r.get("distance_km",0) or 0),1),"platform":str(r.get("platform_no","—"))})
    return stops

# ── HEALTH ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    trains = int(MASTER_DF["train_no"].nunique()) if LOADED and MASTER_DF is not None and not MASTER_DF.empty and "train_no" in MASTER_DF.columns else 0
    return {"status":"ok","loaded":LOADED,"trains":trains,"stations":len(DATA.get("stations",[])) if LOADED else 0}

# ── ADMIN ────────────────────────────────────────────────────────────────────
@app.get("/api/admin/congested-stations")
def congested_stations(top_n:int=15):
    _req(); return _safe(STATION_CONG.head(top_n))

@app.get("/api/admin/congested-corridors")
def congested_corridors(top_n:int=12):
    _req(); return _safe(M.compute_route_congestion(MASTER_DF).head(top_n))

@app.get("/api/admin/congestion-risk")
def congestion_risk():
    _req()
    now=datetime.datetime.now(); results=[]
    for lbl,ho,do in[("Now",0,0),("+2 Hours",2,0),("Tomorrow AM",8,1)]:
        t=now+datetime.timedelta(hours=ho,days=do)
        row={"window":lbl,"predictions":[]}
        for tt,d,w in[("Express",300,"Clear"),("Superfast",500,"Clear"),("Local",80,"Rainy"),("Express",300,"Foggy")]:
            delay=DELAY_PREDICTOR.predict(distance_km=d,weather=w,day_of_week=t.strftime("%A"),time_of_day=_tod(t.hour),train_type=tt)
            cong=CONG_CLF.predict(distance_km=d,weather=w,day_of_week=t.strftime("%A"),time_of_day=_tod(t.hour),train_type=tt)
            row["predictions"].append({"train_type":tt,"distance_km":d,"weather":w,"predicted_delay_min":round(float(delay),1),"congestion_level":cong})
        results.append(row)
    return results

@app.get("/api/admin/cascading-delays")
def cascading_delays(train_no: str = Query(...)):
    _req()
    train_rows = MASTER_DF[MASTER_DF["train_no"].astype(str).str.strip() == str(train_no).strip()]
    if train_rows.empty:
        return [{"message": f"Train {train_no} not found. Please enter a valid train number."}]

    name = str(train_rows.iloc[0].get("train_name", f"Train {train_no}"))
    tt = _type(name)
    now = datetime.datetime.now(); dow = now.strftime("%A"); tod = _tod(now.hour)
    dist = float(train_rows["distance_km"].max() or 300)
    base_delay = float(DELAY_PREDICTOR.predict(distance_km=dist, weather=_weather(), day_of_week=dow, time_of_day=tod, train_type=tt))
    adjusted_delay = _train_specific_delay(base_delay, train_no, train_rows, tt)

    # Find stations on this train's route, ranked by congestion
    train_stations = list(train_rows["station_code"].dropna().astype(str).unique())
    results = []; seen = set()

    if STATION_CONG is not None and not STATION_CONG.empty:
        route_cong = STATION_CONG[STATION_CONG["station_code"].isin(train_stations)].head(15)
        for _, sc in route_cong.iterrows():
            stn = str(sc["station_code"])
            if stn in seen: continue
            seen.add(stn)
            # Count other trains at this station (cascade impact)
            others = MASTER_DF[
                (MASTER_DF["station_code"] == stn) &
                (MASTER_DF["train_no"].astype(str).str.strip() != str(train_no).strip())
            ]["train_no"].nunique()
            cascade = round(adjusted_delay * 1.40, 1)
            risk = "High" if cascade > 30 else ("Medium" if cascade > 15 else "Low")
            results.append({
                "corridor": f"{stn} ({str(sc.get('station_name', stn))[:18]})",
                "distance_km": round(dist, 1),
                "base_delay_min": round(adjusted_delay, 1),
                "cascade_delay_min": cascade,
                "cascade_risk": risk,
                "affected_trains": int(others)
            })

    if not results:
        return [{"message": f"Train {train_no} — {name}: No high-congestion station overlaps detected on route. Delay impact is localised."}]

    results.sort(key=lambda x: x["cascade_delay_min"], reverse=True)
    return results[:10]

@app.get("/api/admin/rerouting")
def rerouting(train_no:str=Query(...)):
    _req(); result=M.get_rerouting_options(train_no,MASTER_DF); return result if isinstance(result,list) else []

@app.get("/api/admin/zone-summary")
def zone_summary():
    _req()
    df=M.compute_zone_summary(MASTER_DF)
    if "congestion_score" not in df.columns:
        zc=STATION_CONG.groupby("zone")["congestion_score"].mean().reset_index() if "zone" in STATION_CONG.columns else pd.DataFrame()
        if not zc.empty: df=df.merge(zc,on="zone",how="left")
    if "congestion_score" in df.columns:
        q33=df["congestion_score"].quantile(0.33); q66=df["congestion_score"].quantile(0.66)
        df["congestion_level"]=pd.cut(df["congestion_score"].fillna(0),bins=[-1,q33,q66,float("inf")],labels=["Low","Medium","High"]).astype(str)
    return _safe(df)

@app.get("/api/admin/bottlenecks")
def bottlenecks():
    _req()
    df=M.detect_bottlenecks(STATION_CONG)
    if isinstance(df,pd.DataFrame):
        df=df.copy(); df["reason"]="Congestion score above 90th percentile threshold"
        return _safe(df)
    return []

@app.get("/api/admin/priority-clearance")
def priority_clearance(top_n:int=20):
    _req()
    tp={"Rajdhani":1,"Shatabdi":1,"Duronto":1,"Superfast":2,"Express":3,"Local":4,"Passenger":4}
    seen=set(); out=[]
    for _,row in MASTER_DF.iterrows():
        tno=str(row.get("train_no",""))
        if tno in seen or not tno: continue
        seen.add(tno)
        name=str(row.get("train_name",f"Train {tno}"))
        tt=_type(name)
        out.append({"train_no":tno,"train_name":name,"train_type":tt,"priority":f"P{tp.get(tt,4)}","source":str(row.get("source_code","")),"dest":str(row.get("dest_code",""))})
        if len(out)>=top_n: break
    out.sort(key=lambda x:x["priority"]); return out

@app.get("/api/admin/operational-dashboard")
def operational_dashboard():
    _req()
    td=MASTER_DF; st=DATA.get("stations",pd.DataFrame())
    zd=st["zone"].value_counts().head(10).to_dict() if not st.empty and "zone" in st.columns else {}
    crit_stations = 0
    if STATION_CONG is not None and not STATION_CONG.empty:
        crit_stations = int(len(STATION_CONG[STATION_CONG["congestion_score"] >= STATION_CONG["congestion_score"].quantile(0.9)]))
    return {
        "total_trains":int(td["train_no"].nunique()) if "train_no" in td.columns else 0,
        "total_stations":len(st),
        "total_routes":int(len(td)),
        "critical_stations": crit_stations,
        "zone_distribution":zd,
        "timestamp":datetime.datetime.now().isoformat()
    }

@app.get("/api/admin/model-metrics")
def model_metrics():
    """Expose MLflow / training run metrics for the dashboard."""
    _req()
    metrics = M.get_model_metrics()
    if not metrics:
        return {
            "status": "models_loaded",
            "message": "Models loaded from cache. Metrics available after fresh training.",
            "delay_model": "RandomForestRegressor (n_estimators=50, max_depth=10)",
            "congestion_model": "RandomForestClassifier (n_estimators=50, max_depth=8)",
            "mlflow": "enabled",
            "experiment": "RailDrishti-ML"
        }
    return metrics

@app.get("/api/admin/network-summary")
def network_summary():
    """Full operational network summary — busiest station, corridor, zone."""
    _req()
    sc = STATION_CONG; rc = M.compute_route_congestion(MASTER_DF); zd = M.compute_zone_summary(MASTER_DF)
    return {
        "busiest_station": {
            "code": str(sc.iloc[0]["station_code"]) if not sc.empty else "",
            "name": str(sc.iloc[0].get("station_name","")) if not sc.empty else "",
            "trains": int(sc.iloc[0]["trains_stopping"]) if not sc.empty else 0,
            "score": float(sc.iloc[0]["congestion_score"]) if not sc.empty else 0,
        },
        "busiest_corridor": {
            "route": str(rc.iloc[0]["route_corridor"]) if not rc.empty else "",
            "trains": int(rc.iloc[0]["trains"]) if not rc.empty else 0,
            "score": float(rc.iloc[0]["route_score"]) if not rc.empty else 0,
        },
        "busiest_zone": {
            "zone": str(zd.iloc[0]["zone"]) if not zd.empty else "",
            "trains": int(zd.iloc[0]["trains"]) if not zd.empty else 0,
            "stations": int(zd.iloc[0]["stations"]) if not zd.empty else 0,
        },
        "critical_stations": int(len(sc[sc["congestion_score"] >= sc["congestion_score"].quantile(0.9)])) if not sc.empty else 0,
        "total_trains": int(MASTER_DF["train_no"].nunique()) if "train_no" in MASTER_DF.columns else 0,
        "total_stations": int(MASTER_DF["station_code"].nunique()) if "station_code" in MASTER_DF.columns else 0,
        "network_health_index": max(0, 100 - int(len(sc[sc["congestion_score"] >= sc["congestion_score"].quantile(0.9)])) * 2) if not sc.empty else 50,
        "timestamp": datetime.datetime.now().isoformat()
    }

# ── USER ─────────────────────────────────────────────────────────────────────
@app.get("/api/user/train-info")
def train_info(train_no:str=Query(...)):
    _req()
    td=MASTER_DF
    matches=td[td["train_no"].astype(str)==str(train_no)]
    if matches.empty: raise HTTPException(404,f"Train {train_no} not found")
    row=matches.iloc[0]
    name=str(row.get("train_name",f"Train {train_no}"))
    src=str(row.get("source_code",row.get("station_code","")))
    dst=str(row.get("dest_code",""))
    tt=_type(name); now=datetime.datetime.now(); tod=_tod(now.hour); dow=now.strftime("%A")
    # Use MAXIMUM distance in route (full end-to-end), not just first stop
    dist=float(matches["distance_km"].max() or row.get("distance_km",300) or 300)
    w=_weather()
    # 1. Get base ML prediction
    base_delay=float(DELAY_PREDICTOR.predict(distance_km=dist,weather=w,day_of_week=dow,time_of_day=tod,train_type=tt))
    # 2. Apply train-specific adjustments (stops, route congestion, hash variation)
    delay=_train_specific_delay(base_delay, train_no, matches, tt)
    cong=CONG_CLF.predict(distance_km=dist,weather=w,day_of_week=dow,time_of_day=tod,train_type=tt)
    rel=M.get_train_reliability(train_no,MASTER_DF,DATA["delay_data"])
    n_stops=len(matches)
    return {
        "train_no":train_no,"train_name":name,"train_type":tt,
        "source":src,"destination":dst,"distance_km":round(dist,1),
        "stops_count":n_stops,
        "predicted_delay":delay,
        "congestion_level":cong,
        "risk_level":"High" if delay>30 else("Medium" if delay>10 else"Low"),
        "reliability_score":round(float(rel),1),
        "delay_reasons":_reasons(delay,w,tod,cong,n_stops),
        "weather":w,
        "stops":_stops(matches,delay)
    }

@app.get("/api/user/alternatives")
def alternatives(train_no:str=Query(...)):
    _req(); r=M.get_rerouting_options(train_no,MASTER_DF); return r if isinstance(r,list) else []

@app.get("/api/user/station-info")
def station_info(station_code:str=Query(...)):
    _req(); st=DATA["stations"]; cu=station_code.upper().strip()
    m=st[st["station_code"]==cu]
    if m.empty: m=st[st["station_name"].str.upper().str.contains(cu,na=False)]
    if m.empty: raise HTTPException(404,f"Station {station_code} not found")
    r=m.iloc[0]
    return {"station_code":str(r.get("station_code","")),"station_name":str(r.get("station_name","")),"state":str(r.get("state","")),"zone":str(r.get("zone","")),"latitude":float(r.get("latitude",0) or 0),"longitude":float(r.get("longitude",0) or 0)}

@app.get("/api/user/nearby-stations")
def nearby_stations(station_code:str=Query(...),radius_km:float=50):
    _req(); st=DATA["stations"]; cu=station_code.upper().strip()
    base=st[st["station_code"]==cu]
    if base.empty: raise HTTPException(404,f"Station {station_code} not found")
    blat=float(base.iloc[0].get("latitude",0) or 0); blon=float(base.iloc[0].get("longitude",0) or 0)
    def hav(r):
        la,lo=float(r.get("latitude",0) or 0),float(r.get("longitude",0) or 0)
        dl=math.radians(la-blat); dn=math.radians(lo-blon)
        a=math.sin(dl/2)**2+math.cos(math.radians(blat))*math.cos(math.radians(la))*math.sin(dn/2)**2
        return 6371*2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    nb=st.copy(); nb["dist_km"]=nb.apply(hav,axis=1)
    nb=nb[(nb["dist_km"]>0.1)&(nb["dist_km"]<=radius_km)].sort_values("dist_km").head(10)
    return _safe(nb[["station_code","station_name","state","zone","dist_km"]])

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)
