# app.py
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from pathlib import Path
import os, math, ssl, smtplib, requests
import pandas as pd

# =========================
# Configuración y conexión
# =========================
load_dotenv()  # lee api/.env

DATABASE_URL   = os.getenv("DATABASE_URL")
REFRESH_TOKEN  = os.getenv("REFRESH_TOKEN")
AIRNOW_KEY     = os.getenv("AIRNOW_KEY")
GMAIL_USER     = os.getenv("GMAIL_USER")
GMAIL_APP_PASS = os.getenv("GMAIL_APP_PASSWORD")

if not DATABASE_URL:
    raise RuntimeError("Falta DATABASE_URL en .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="AireSaludable API (US)", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================
# Utilidades
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0; p=math.pi/180
    dlat=(lat2-lat1)*p; dlon=(lon2-lon1)*p
    a=(math.sin(dlat/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(a))

def aqi_pm25_epa(pm):
    try:
        c = round(float(pm), 1)
    except:
        return None
    brks = [
        (0.0, 12.0,   0,  50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4,101, 150),
        (55.5,150.4,151, 200),
        (150.5,250.4,201,300),
        (250.5,500.4,301,500),
    ]
    for Clow, Chigh, Ilow, Ihigh in brks:
        if Clow <= c <= Chigh:
            return int(round((Ihigh-Ilow)/(Chigh-Clow)*(c-Clow)+Ilow))
    return 500

def nearest_measurements(lat: float, lon: float, horizon_hours: int = 6, limit: int = 50) -> List[Dict[str, Any]]:
    sql = """
    select station_id, ts, lat, lon,
           pm25, pm10, o3, no2, so2, co,
           aqi_us, aqi_overall, source
    from measurement
    where ts > now() - interval :h
    order by ((lat - :lat)*(lat - :lat) + (lon - :lon)*(lon - :lon)) asc
    limit :limit
    """
    with engine.begin() as con:
        rows = con.execute(
            text(sql),
            {"h": f"{horizon_hours} hours", "lat": lat, "lon": lon, "limit": limit}
        ).mappings().all()
    return [dict(r) for r in rows]

def idw_from_rows(lat: float, lon: float, rows: List[Dict[str, Any]], k: int = 5, p: float = 2.0):
    if not rows: 
        return None
    rows = sorted(
        rows,
        key=lambda r: haversine_km(lat, lon, float(r["lat"]), float(r["lon"]))
    )[:k]
    d0 = haversine_km(lat, lon, float(rows[0]["lat"]), float(rows[0]["lon"]))
    if d0 < 0.1:
        r0 = rows[0]
        aqi = r0.get("aqi_overall") or r0.get("aqi_us")
        if aqi is None and r0.get("pm25") is not None:
            aqi = aqi_pm25_epa(r0["pm25"])
        return {
            "pm25": float(r0["pm25"]) if r0.get("pm25") is not None else None,
            "aqi_us": int(aqi) if aqi is not None else None,
            "source": r0.get("source") or "station",
            "ts": r0.get("ts")
        }
    eps = 1e-6
    num_aqi = den_aqi = 0.0
    num_pm  = den_pm  = 0.0
    for r in rows:
        d = haversine_km(lat, lon, float(r["lat"]), float(r["lon"]))
        w = 1.0 / ((d + eps) ** p)
        aqi_val = r.get("aqi_overall") or r.get("aqi_us")
        if aqi_val is None and r.get("pm25") is not None:
            aqi_val = aqi_pm25_epa(r["pm25"])
        if aqi_val is not None:
            num_aqi += w * float(aqi_val)
            den_aqi += w
        if r.get("pm25") is not None:
            num_pm += w * float(r["pm25"])
            den_pm += w
    aqi = int(round(num_aqi / den_aqi)) if den_aqi > 0 else None
    pm  = round(num_pm / den_pm, 1) if den_pm > 0 else None
    return {"pm25": pm, "aqi_us": aqi, "source": f"idw_k{k}"}

# =========================
# Endpoints: salud y consultas
# =========================
@app.get("/health")
def health():
    with engine.begin() as con:
        c = con.execute(text("select count(*) from measurement")).scalar() or 0
    return {"ok": True, "rows": int(c)}

@app.get("/api/aqi")
def api_aqi(lat: float = Query(...), lon: float = Query(...)):
    rows = nearest_measurements(lat, lon, horizon_hours=6, limit=100)
    r = idw_from_rows(lat, lon, rows)
    return {"coord": {"lat": lat, "lon": lon}, **(r or {"error": "no_data"})}

@app.get("/api/forecast")
def forecast(lat: float, lon: float, hours: int = 6):
    hours = max(1, min(24, hours))
    rows = nearest_measurements(lat, lon, horizon_hours=6, limit=100)
    base = idw_from_rows(lat, lon, rows) or {"aqi_us": 80, "pm25": 20.0}
    aqi0 = int(base.get("aqi_us") or 80)
    pm0  = float(base.get("pm25") or 20.0)
    now  = datetime.now(timezone.utc).replace(microsecond=0)
    items = []
    for h in range(1, hours + 1):
        aqi_h = max(0, min(500, int(round(aqi0 * (1 + 0.03 * (1 if h % 2 else -1))))))
        pm_h  = round(pm0  * (1 + 0.04 * (1 if h % 2 else -1)), 1)
        items.append({
            "t": (now + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "aqi_us": aqi_h,
            "pm25": pm_h
        })
    return {"coord": {"lat": lat, "lon": lon}, "horizon_hours": hours, "items": items, "model": "baseline_v0"}

# =========================
# Alertas por email
# =========================
class EmailSub(BaseModel):
    email: EmailStr
    lat: float
    lon: float
    threshold_aqi: int = 150

EMAIL_SUBS: List[Dict[str, Any]] = []

def send_email(to_email: str, subject: str, html: str, text: str = ""):
    if not GMAIL_USER or not GMAIL_APP_PASS:
        raise RuntimeError("Falta GMAIL_USER/GMAIL_APP_PASSWORD en .env")
    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"]   = to_email
    msg["Subject"] = subject
    msg.set_content(text or "Ver HTML")
    msg.add_alternative(html, subtype="html")
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as s:
        s.login(GMAIL_USER, GMAIL_APP_PASS)
        s.send_message(msg)

@app.post("/api/alerts/subscribe")
def subscribe_email(sub: EmailSub):
    EMAIL_SUBS.append(sub.dict())
    return {"ok": True, "total_subs": len(EMAIL_SUBS)}

@app.post("/api/alerts/test-email")
def test_email(to: EmailStr):
    send_email(str(to), "Prueba Aire Saludable", "<h3>Suscripción ok ✅</h3>")
    return {"ok": True}

@app.post("/api/alerts/scan-once")
def scan_once():
    alerts = 0
    for s in EMAIL_SUBS:
        lat, lon, thr = float(s["lat"]), float(s["lon"]), int(s["threshold_aqi"])
        rows = nearest_measurements(lat, lon, horizon_hours=6, limit=100)
        r = idw_from_rows(lat, lon, rows)
        if not r or r.get("aqi_us") is None:
            continue
        if int(r["aqi_us"]) >= thr:
            html = f"""
            <h3>⚠️ Alerta de calidad del aire</h3>
            <p><b>AQI:</b> {r['aqi_us']} — <b>PM2.5:</b> {r.get('pm25','?')} µg/m³</p>
            <p>Ubicación aprox: {lat:.3f}, {lon:.3f}</p>
            <p>Recomendación: reduce exposición al aire libre.</p>
            """
            send_email(s["email"], "⚠️ AQI elevado cerca de ti", html)
            alerts += 1
    return {"ok": True, "subs_checked": len(EMAIL_SUBS), "alerts_sent": alerts}

# =========================
# Ingesta AirNow (tiempo real)
# =========================
AIRNOW_URL = "https://www.airnowapi.org/aq/data/"

def _normalize_airnow_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: Dict[Any, Dict[str, Any]] = {}
    def key_for(d):
        ts = d.get("UTC") or d.get("DateTimeUTC") or d.get("UTCDate")
        lat = d.get("Latitude"); lon = d.get("Longitude")
        site = d.get("SiteName") or d.get("ReportingArea") or None
        return (ts, site, round(lat, 3) if lat is not None else None, round(lon, 3) if lon is not None else None)
    def ensure_row(k, d):
        if k not in norm:
            ts, site, latr, lonr = k
            sid = site or f"airnow-{latr}-{lonr}"
            norm[k] = {
                "ts": ts, "lat": d.get("Latitude"), "lon": d.get("Longitude"),
                "station_id": sid,
                "pm25": None, "pm10": None, "o3": None, "no2": None, "so2": None, "co": None,
                "aqi_overall": None, "aqi_us": None
            }
        return norm[k]
    has_parameter = any(("Parameter" in r or "parameter" in r) for r in rows)
    has_wide_cols = any(
        any(c in r for c in ["PM25", "PM10", "OZONE", "NO2", "SO2", "CO"]) for r in rows
    )
    if has_parameter:
        for d in rows:
            k = key_for(d)
            if None in k[:1] or d.get("Latitude") is None: continue
            row = ensure_row(k, d)
            par = (d.get("Parameter") or d.get("parameter") or "").upper()
            val = d.get("Value"); aqi = d.get("AQI")
            if par in ("PM2.5", "PM25"): row["pm25"] = val
            elif par == "PM10": row["pm10"] = val
            elif par in ("O3", "OZONE"): row["o3"] = val
            elif par == "NO2": row["no2"] = val
            elif par == "SO2": row["so2"] = val
            elif par == "CO": row["co"] = val
            if aqi is not None:
                row["aqi_overall"] = max(row["aqi_overall"] or 0, aqi)
    elif has_wide_cols:
        for d in rows:
            k = key_for(d)
            if None in k[:1] or d.get("Latitude") is None: continue
            row = ensure_row(k, d)
            if "PM25"  in d: row["pm25"] = d.get("PM25")
            if "PM10"  in d: row["pm10"] = d.get("PM10")
            if "OZONE" in d: row["o3"]   = d.get("OZONE")
            if "NO2"   in d: row["no2"]  = d.get("NO2")
            if "SO2"   in d: row["so2"]  = d.get("SO2")
            if "CO"    in d: row["co"]   = d.get("CO")
            if "AQI" in d and d.get("AQI") is not None:
                row["aqi_overall"] = d.get("AQI")
    for r in norm.values():
        if r["aqi_overall"] is not None:
            r["aqi_us"] = int(r["aqi_overall"])
        elif r["pm25"] is not None:
            r["aqi_us"] = aqi_pm25_epa(r["pm25"])
        if isinstance(r["ts"], str):
            try:
                r["ts"] = datetime.fromisoformat(r["ts"].replace("Z","")).replace(tzinfo=timezone.utc)
            except Exception:
                pass
    return list(norm.values())

@app.post("/ingest/airnow/latest")
def ingest_airnow_latest(request: Request):
    if REFRESH_TOKEN:
        tok = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        if tok != REFRESH_TOKEN:
            raise HTTPException(401, "Unauthorized")
    if not AIRNOW_KEY:
        raise HTTPException(500, "Falta AIRNOW_KEY en .env")
    # últimas 6h dinámicas
    end = datetime.utcnow()
    start = end - timedelta(hours=6)
    params = {
        "startDate": start.strftime("%Y-%m-%dT%H"),
        "endDate": end.strftime("%Y-%m-%dT%H"),
        "parameters": "PM25,PM10,OZONE,NO2,SO2,CO",
        "BBOX": "-125,24,-66,50",
        "dataType": "A",
        "format": "application/json",
        "API_KEY": AIRNOW_KEY
    }
    r = requests.get(AIRNOW_URL, params=params, timeout=90)
    r.raise_for_status()
    raw_rows = r.json() if r.content else []
    rows = _normalize_airnow_rows(raw_rows)
    if not rows:
        return {"ok": True, "inserted": 0, "note": "sin filas válidas"}
    stmt = text("""
        insert into measurement
            (source, station_id, ts, lat, lon,
             pm25, pm10, o3, no2, so2, co, aqi_us, aqi_overall)
        values
            ('airnow', :station_id, :ts, :lat, :lon,
             :pm25, :pm10, :o3, :no2, :so2, :co, :aqi_us, :aqi_overall)
        on conflict (station_id, ts) do update set
            pm25 = excluded.pm25, pm10 = excluded.pm10,
            o3 = excluded.o3, no2 = excluded.no2, so2 = excluded.so2, co = excluded.co,
            aqi_us = excluded.aqi_us, aqi_overall = excluded.aqi_overall
    """)
    inserted = 0
    CHUNK = 200
    for i in range(0, len(rows), CHUNK):
        payload = []
        for d in rows[i:i+CHUNK]:
            if not d.get("ts") or d.get("lat") is None: continue
            payload.append({
                "station_id": d["station_id"],
                "ts": d["ts"],
                "lat": float(d["lat"]), "lon": float(d["lon"]),
                "pm25": d.get("pm25"), "pm10": d.get("pm10"),
                "o3": d.get("o3"), "no2": d.get("no2"),
                "so2": d.get("so2"), "co": d.get("co"),
                "aqi_us": d.get("aqi_us"), "aqi_overall": d.get("aqi_overall"),
            })
        if payload:
            with engine.begin() as con:
                con.execute(stmt, payload)
            inserted += len(payload)
    return {"ok": True, "inserted": inserted, "received": len(rows)}

# =========================
# Resumen por estado (AirNow)
# =========================
@app.get("/api/airnow/state-summary")
def airnow_state_summary(hours: int = 6):
    sql = """
    SELECT replace(station_id,'airnow-','') as state, ROUND(AVG(aqi_us),0) as aqi
    FROM measurement
    WHERE source='airnow' AND ts > now() - interval :h
    GROUP BY station_id
    ORDER BY state
    """
    with engine.begin() as con:
        rows = con.execute(text(sql), {"h": f"{hours} hours"}).mappings().all()
    return {"summary": [dict(r) for r in rows]}
