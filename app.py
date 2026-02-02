import os
import csv
import math
import threading
from datetime import datetime

import requests
import numpy as np
import joblib
import folium

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# ============================================================
# CONFIGURACIÓN
# ============================================================
MIN_EVENT_MAGNITUDE = float(os.getenv("MIN_EVENT_MAGNITUDE", "0"))
MIN_INTENSITY_TO_SHOW = int(os.getenv("MIN_INTENSITY_TO_SHOW", "2"))
DEFAULT_TABLE_ROWS = int(os.getenv("DEFAULT_TABLE_ROWS", "200"))

PRELOAD_MODEL_ON_STARTUP = True

HTTP_TIMEOUT = 25
MODEL_DOWNLOAD_TIMEOUT = 600

XOR_API_URL = "https://api.xor.cl/sismo/recent"

CSV_PATH = "Localidades_Enero_2026_con_coords.csv"

MODEL_PATH = "Sismos_RF_joblib_Ene_2026.pkl"
MODEL_URL = "https://github.com/juansotodaniels/sismos-railway/releases/download/v1.0/Sismos_RF_joblib_Ene_2026.pkl"

HEADERS = {"User-Agent": "SismoTrack/1.0"}

app = FastAPI(title="SismoTrack")

# ============================================================
# UTILIDADES
# ============================================================
def _to_float(x):
    return float(str(x).replace(",", ".").strip())

def safe_get(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    m = {k.lower(): k for k in d.keys()}
    for k in keys:
        if k.lower() in m:
            return d[m[k.lower()]]
    return default

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def round_intensity(x):
    try:
        return int(round(float(x)))
    except Exception:
        return 0

# ============================================================
# FECHA (FORMATO DD-MM-YYYY HH:MM:SS)
# ============================================================
def parse_datetime_flexible(value):
    if value is None:
        return "No disponible"

    s = str(value).strip()

    # Epoch
    if s.isdigit() and len(s) in (10, 13):
        ts = int(s)
        if len(s) == 13:
            ts /= 1000
        return datetime.utcfromtimestamp(ts).strftime("%d-%m-%Y %H:%M:%S")

    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]

    for f in formats:
        try:
            return datetime.strptime(s, f).strftime("%d-%m-%Y %H:%M:%S")
        except Exception:
            pass

    return s

# ============================================================
# PARSEO EVENTO XOR
# ============================================================
def extract_event(ev):
    mag = safe_get(ev, ["magnitude", "mag", "magnitud"])
    mag_type = None
    if isinstance(mag, dict):
        mag_type = mag.get("measure_unit")
        mag = mag.get("value")

    lat = safe_get(ev, ["lat", "latitude"])
    lon = safe_get(ev, ["lon", "longitude"])
    depth = safe_get(ev, ["depth", "profundidad"]) or 0

    fecha = (
        safe_get(ev, ["local_date"]) or
        safe_get(ev, ["utc_date"]) or
        safe_get(ev, ["date"]) or
        safe_get(ev, ["datetime"])
    )

    ref = safe_get(ev, ["geo_reference", "reference", "place", "location"]) or "No disponible"

    return {
        "Latitud_sismo": _to_float(lat),
        "Longitud_sismo": _to_float(lon),
        "Profundidad": _to_float(depth),
        "magnitud": _to_float(mag),
        "mag_type": mag_type or "No disponible",
        "FechaHora": parse_datetime_flexible(fecha),
        "Referencia": ref,
    }

def fetch_latest_event():
    r = requests.get(XOR_API_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    events = data if isinstance(data, list) else data.get("events", [])
    for ev in events:
        evento = extract_event(ev)
        if evento["magnitud"] >= MIN_EVENT_MAGNITUDE:
            return evento

    raise RuntimeError("No se encontró sismo válido")

# ============================================================
# CSV LOCALIDADES
# ============================================================
def read_localidades():
    locs = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            locs.append({
                "localidad": r["Localidad"],
                "comuna": r.get("Comuna", ""),
                "region": r.get("Region", ""),
                "lat": _to_float(r["Latitud"]),
                "lon": _to_float(r["Longitud"]),
            })
    return locs

# ============================================================
# MODELO
# ============================================================
MODEL = None
LOCK = threading.Lock()

def load_model():
    global MODEL
    with LOCK:
        if MODEL is None:
            if not os.path.exists(MODEL_PATH):
                r = requests.get(MODEL_URL, stream=True)
                with open(MODEL_PATH, "wb") as f:
                    for c in r.iter_content(1024 * 1024):
                        f.write(c)
            MODEL = joblib.load(MODEL_PATH)
    return MODEL

def predict_intensities(evento):
    model = load_model()
    locs = read_localidades()

    X = []
    meta = []

    for l in locs:
        d = haversine_km(evento["Latitud_sismo"], evento["Longitud_sismo"], l["lat"], l["lon"])
        X.append([
            evento["Latitud_sismo"],
            evento["Longitud_sismo"],
            evento["Profundidad"],
            evento["magnitud"],
            l["lat"],
            l["lon"],
            d,
        ])
        meta.append({**l, "dist_km": int(round(d))})

    y = model.predict(np.array(X))

    out = []
    for i, m in enumerate(meta):
        I = round_intensity(y[i])
        if I >= MIN_INTENSITY_TO_SHOW:
            out.append({**m, "intensidad": I})

    out.sort(key=lambda x: (-x["intensidad"], x["dist_km"]))
    return out

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/", response_class=HTMLResponse)
def home():
    ev = fetch_latest_event()
    preds = predict_intensities(ev)

    rows = "".join(
        f"<tr><td>{i+1}</td><td>{p['localidad']}</td><td>{p['comuna']}</td><td>{p['region']}</td>"
        f"<td>{p['dist_km']}</td><td><b>{p['intensidad']}</b></td></tr>"
        for i, p in enumerate(preds[:DEFAULT_TABLE_ROWS])
    )

    return f"""
    <html><body style="font-family:Arial;padding:20px">
    <h1>SismoTrack</h1>
    <ul>
      <li><b>Fecha/Hora:</b> {ev['FechaHora']}</li>
      <li><b>Magnitud:</b> {ev['magnitud']} ({ev['mag_type']})</li>
      <li><b>Profundidad:</b> {ev['Profundidad']} km</li>
      <li><b>Referencia:</b> {ev['Referencia']}</li>
    </ul>

    <table border="1" cellpadding="6">
      <tr><th>#</th><th>Localidad</th><th>Comuna</th><th>Región</th><th>Distancia (km)</th><th>I</th></tr>
      {rows}
    </table>
    </body></html>
    """

@app.get("/health")
def health():
    return {
        "api": XOR_API_URL,
        "csv": os.path.exists(CSV_PATH),
        "model": os.path.exists(MODEL_PATH),
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

