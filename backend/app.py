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
]
# Also allow any Vercel deployment URL via env var (set FRONTEND_URL on Railway)
_frontend_url = os.environ.get("FRONTEND_URL", "")
if _frontend_url:
    ALLOWED_ORIGINS.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Frontend is served by Vercel — no static mount needed here
# (StaticFiles import kept for potential local dev use)

DATA={}; DELAY_PREDICTOR=None; CONG_CLF=None; MASTER_DF=None; STATION_CONG=None; LOADED=False

@app.on_event("startup")
def startup():
    global DATA,DELAY_PREDICTOR,CONG_CLF,MASTER_DF,STATION_CONG,LOADED
    try:
        print(f"BASE_DIR: {BASE_DIR}")
        print(f"DATA_DIR: {DATA_DIR}")
        print("Loading datasets...")
        DATA = data_loader.load_all(verbose=True)
        print("Training ML models...")
        DELAY_PREDICTOR,CONG_CLF = M.train_all_models(DATA["delay_data"], verbose=True)
        MASTER_DF = _build_master()
        STATION_CONG = M.compute_station_congestion(MASTER_DF)
        LOADED = True
        print("System ready ✅")
    except Exception as e:
        print(f"Startup error: {e}"); traceback.print_exc()

def _build_master():
    td = DATA.get("train_details", pd.DataFrame()).copy()
    st = DATA.get("stations", pd.DataFrame())
    if td.empty: return pd.DataFrame()
    col_map = {}
    for col in td.columns:
        cl = col.lower().strip()
        if "train" in cl and "no" in cl: col_map[col]="train_no"
        elif "train" in cl and "name" in cl: col_map[col]="train_name"
        elif cl in ("islno","serial","sno","sr_no","stop_no","stop","station_seq","sequence_no"): col_map[col]="stop_seq"
        elif "station" in cl and "code" in cl and "source" not in cl and "dest" not in cl: col_map[col]="station_code"
        elif "station" in cl and "name" in cl: col_map[col]="station_name"
        elif "source" in cl and "code" in cl: col_map[col]="source_code"
        elif "dest" in cl and "code" in cl: col_map[col]="dest_code"
        elif "distance" in cl: col_map[col]="distance_km"
        elif "arrival" in cl: col_map[col]="arrival_time"
        elif "departure" in cl: col_map[col]="departure_time"
        elif "platform" in cl: col_map[col]="platform_no"
    td.rename(columns=col_map, inplace=True)
    for c in ["source_code","dest_code","station_code","train_no","train_name","distance_km"]:
        if c not in td.columns: td[c]=""
    if "source_code" in td.columns and "dest_code" in td.columns:
        td["route_corridor"] = td["source_code"].fillna("")+"→"+td["dest_code"].fillna("")
    else:
        td["route_corridor"] = td.get("station_code","")
    if not st.empty and "station_code" in st.columns:
        st_slim = st[["station_code"]+[c for c in ["zone","state","latitude","longitude"] if c in st.columns]].copy()
        td = td.merge(st_slim, on="station_code", how="left")
    for c in ["zone","state","latitude","longitude"]:
        if c not in td.columns: td[c]=""
    return td

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

def _reasons(delay,weather,tod,cong):
    r=[]
    if delay<5: return ["✅ No significant delay expected — on time"]
    if weather=="Foggy": r.append("🌫️ Foggy conditions — reduced speed operations active")
    elif weather=="Rainy": r.append("🌧️ Rainfall affecting track adhesion and signal visibility")
    elif weather=="Stormy": r.append("⛈️ Storm advisory — speed restrictions imposed")
    if cong=="High": r.append("🔴 Route corridor heavily congested — multiple trains sharing track")
    elif cong=="Medium": r.append("🟡 Moderate congestion — some scheduling pressure")
    if tod in("Morning","Evening"): r.append(f"⏱️ Peak hour ({tod}) — signal queuing likely")
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
def cascading_delays(train_no:str=Query(...)):
    _req()
    result=M.detect_cascading_delays(MASTER_DF,DELAY_PREDICTOR)
    if isinstance(result,pd.DataFrame) and not result.empty:
        for col in result.columns:
            if "train" in col.lower():
                filt=result[result[col].astype(str).str.contains(str(train_no),na=False)]
                if not filt.empty: return _safe(filt)
        return _safe(result.head(10))
    return [{"message":f"Train {train_no}: delay cascade analysis complete — no downstream impacts detected"}]

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
    return {"total_trains":int(td["train_no"].nunique()) if "train_no" in td.columns else 0,"total_stations":len(st),"total_routes":int(len(td)),"zone_distribution":zd,"timestamp":datetime.datetime.now().isoformat()}

# ── USER ─────────────────────────────────────────────────────────────────────
@app.get("/api/user/train-info")
def train_info(train_no:str=Query(...)):
    _req()
    td=MASTER_DF; matches=td[td["train_no"].astype(str)==str(train_no)]
    if matches.empty: raise HTTPException(404,f"Train {train_no} not found")
    row=matches.iloc[0]
    name=str(row.get("train_name",f"Train {train_no}"))
    src=str(row.get("source_code",row.get("station_code","")))
    dst=str(row.get("dest_code",""))
    tt=_type(name); now=datetime.datetime.now(); tod=_tod(now.hour); dow=now.strftime("%A")
    dist=float(row.get("distance_km",300) or 300); w=_weather()
    delay=float(DELAY_PREDICTOR.predict(distance_km=dist,weather=w,day_of_week=dow,time_of_day=tod,train_type=tt))
    cong=CONG_CLF.predict(distance_km=dist,weather=w,day_of_week=dow,time_of_day=tod,train_type=tt)
    rel=M.get_train_reliability(train_no,MASTER_DF,DATA["delay_data"])
    return {"train_no":train_no,"train_name":name,"train_type":tt,"source":src,"destination":dst,"distance_km":round(dist,1),"predicted_delay":round(delay,1),"congestion_level":cong,"risk_level":"High" if delay>30 else("Medium" if delay>10 else"Low"),"reliability_score":round(float(rel),1),"delay_reasons":_reasons(delay,w,tod,cong),"weather":w,"stops":_stops(matches,delay)}

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
