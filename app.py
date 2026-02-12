import os
import csv
import math
import threading
from datetime import datetime, timedelta

import requests
import numpy as np
import joblib
import folium

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ============================================================
# CONFIGURACIÓN (MODIFICABLE)
# ============================================================
MIN_EVENT_MAGNITUDE = float(os.getenv("MIN_EVENT_MAGNITUDE", "4"))          # evento: M >=
MIN_INTENSITY_TO_SHOW = int(os.getenv("MIN_INTENSITY_TO_SHOW", "3"))        # mostrar: I >=
DEFAULT_TABLE_ROWS = int(os.getenv("DEFAULT_TABLE_ROWS", "200"))            # filas mostradas en Home

PRELOAD_MODEL_ON_STARTUP = os.getenv("PRELOAD_MODEL_ON_STARTUP", "1") == "1"

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "25"))
MODEL_DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "600"))

# ✅ API XOR
XOR_API_URL = os.getenv("XOR_API_URL", "https://api.xor.cl/sismo/recent")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RailwayBot/1.0; +https://railway.app)"}

# CSV con localidades y coords
CSV_PATH = os.getenv("LOCALIDADES_CSV", "Localidades_Enero_2026_con_coords.csv")

MODEL_PATH = os.getenv("MODEL_PATH", "Sismos_RF_joblib_Ene_2026.pkl")
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/juansotodaniels/sismos-railway/releases/download/v1.0/Sismos_RF_joblib_Ene_2026.pkl"
)

LOGO_VERSION = "20260203"

app = FastAPI(title="YATI — Predicción de Intensidad Sísmica (RF)")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# -------------------------
# Utilidades
# -------------------------
def _to_float(s) -> float:
    s = str(s).strip().replace(",", ".")
    return float(s)

def _clean_txt(x) -> str:
    if x is None: return ""
    return str(x).strip()

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calcular_distancia_critica(magnitud: float) -> float:
    """Calcula dc = 11.522 * M^2 - 8.1164 * M + 37.591"""
    return 11.522 * (magnitud ** 2) - 8.1164 * magnitud + 37.591

def detect_delimiter(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=";,\t|")
        return dialect.delimiter
    except Exception:
        return ";"

def round_intensity(x) -> int:
    try:
        v = float(x)
        return int(round(v)) if not np.isnan(v) else 0
    except:
        return 0

def safe_get(d: dict, keys: list[str], default=None):
    if not isinstance(d, dict): return default
    lower_map = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        kk = str(k).lower()
        if kk in lower_map: return d.get(lower_map[kk])
    return default

def parse_datetime_flexible(value):
    if value is None: return None
    s = str(value).strip()
    if s.isdigit() and len(s) in (10, 13):
        try:
            ts = int(s) / 1000 if len(s) == 13 else int(s)
            return datetime.utcfromtimestamp(ts).strftime("%d-%m-%Y %H:%M:%S")
        except: pass
    candidates = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"]
    for fmt in candidates:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%d-%m-%Y %H:%M:%S")
        except: continue
    return s

# -------------------------
# Header HTML
# -------------------------
def render_header_html() -> str:
    return f"""
    <div style="display:flex; align-items:center; gap:18px; margin-bottom:18px;">
      <img src="/static/logo.png?v={LOGO_VERSION}" alt="logo" style="height:180px; width:auto;">
      <div>
        <h1 style="margin:0; font-size:56px; line-height:1;">
          Y<span style="color:#f57c00;">A</span>T<span style="color:#f57c00;">I</span>
        </h1>
        <h2 style="margin:6px 0 0 0; font-size:28px; font-weight:600; color:#000;">
          Sistema de predicción de intensidad sísmica (Chile)
        </h2>
      </div>
    </div>
    """

# -------------------------
# Parseo XOR
# -------------------------
def extract_lat_lon(ev: dict):
    lat = safe_get(ev, ["lat", "latitude", "latitud", "y"])
    lon = safe_get(ev, ["lon", "lng", "long", "longitude", "longitud", "x"])
    if lat is not None and lon is not None: return lat, lon
    return None, None

def extract_magnitude(ev: dict):
    mag = safe_get(ev, ["magnitude", "magnitud", "mag", "m"])
    mag_unit = None
    if isinstance(mag, dict):
        mag_unit = safe_get(mag, ["unit", "type"])
        mag = safe_get(mag, ["value"])
    return mag, mag_unit

def extract_datetime(ev: dict):
    dt_raw = (safe_get(ev, ["local_date", "fecha_local"]) or safe_get(ev, ["date", "datetime", "timestamp"]))
    return parse_datetime_flexible(dt_raw) or "No disponible"

def extract_georef(ev: dict):
    g = safe_get(ev, ["geo_reference", "reference", "place", "location", "ubicacion"])
    return _clean_txt(g) or "No disponible"

# -------------------------
# Modelo y Localidades
# -------------------------
MODEL = None
MODEL_LOCK = threading.Lock()

def load_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            if not os.path.exists(MODEL_PATH):
                resp = requests.get(MODEL_URL, stream=True, timeout=MODEL_DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024): f.write(chunk)
            MODEL = joblib.load(MODEL_PATH)
    return MODEL

def read_localidades(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        delim = detect_delimiter(f.read(4096))
    locs = []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            try:
                lat = _to_float(row.get("latitud") or row.get("lat") or row.get("Latitud_localidad"))
                lon = _to_float(row.get("longitud") or row.get("lon") or row.get("Longitud_localidad"))
                locs.append({
                    "localidad": _clean_txt(row.get("localidad") or row.get("nombre")),
                    "Latitud_localidad": lat, "Longitud_localidad": lon,
                    "comuna": _clean_txt(row.get("comuna")), "region": _clean_txt(row.get("region"))
                })
            except: continue
    return locs

def predict_intensidades(evento: dict, min_intensity: int = MIN_INTENSITY_TO_SHOW):
    locs = read_localidades(CSV_PATH)
    lat_s, lon_s = evento["Latitud_sismo"], evento["Longitud_sismo"]
    mag_sismo = float(evento["magnitud"])
    dist_critica = calcular_distancia_critica(mag_sismo) # ✅ Cálculo DC

    model = load_model()
    features_order = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else [
        "Latitud_sismo", "Longitud_sismo", "Profundidad", "magnitud", 
        "Latitud_localidad", "Longitud_localidad", "distancia_epicentro"
    ]

    rows = []
    meta = []
    for loc in locs:
        dist = haversine_km(lat_s, lon_s, loc["Latitud_localidad"], loc["Longitud_localidad"])
        feat = {
            "Latitud_sismo": lat_s, "Longitud_sismo": lon_s,
            "Profundidad": evento["Profundidad"], "magnitud": mag_sismo,
            "Latitud_localidad": loc["Latitud_localidad"], "Longitud_localidad": loc["Longitud_localidad"],
            "distancia_epicentro": dist
        }
        rows.append([float(feat.get(c, 0)) for c in features_order])
        meta.append({**loc, "distancia_epicentro_km": int(round(dist))})

    y_pred = model.predict(np.array(rows))
    out = []
    for i, m in enumerate(meta):
        # ✅ Aplicación de Corrección por Distancia Crítica
        if m["distancia_epicentro_km"] > dist_critica:
            intensidad = 0
        else:
            intensidad = round_intensity(y_pred[i])
            
        if intensidad < int(min_intensity): continue
        out.append({**m, "intensidad_predicha": intensidad})

    out.sort(key=lambda x: (-x["intensidad_predicha"], x["distancia_epicentro_km"]))
    return out, features_order

# -------------------------
# Renderizado de Tabla y Mapa
# -------------------------
def render_table(preds: list[dict], n: int) -> str:
    show = preds[:n]
    if not show:
        return '<div style="padding:14px; background:#fafafa; border-radius:10px;">No hay localidades que cumplan el umbral.</div>'
    rows_html = "".join([
        f"<tr><td style='text-align:center;'>{i+1}</td><td>{x['localidad']}</td><td>{x.get('comuna','')}</td><td>{x.get('region','')}</td>"
        f"<td style='text-align:center;'>{x['distancia_epicentro_km']}</td><td style='text-align:center;'><b>{x['intensidad_predicha']}</b></td></tr>"
        for i, x in enumerate(show)
    ])
    return f"""
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%;">
      <thead><tr><th>#</th><th>Localidad</th><th>Comuna</th><th>Región</th><th>Distancia (km)</th><th>Intensidad</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """

def build_map_html(evento: dict, preds: list[dict], n: int) -> str:
    lat_s, lon_s = evento["Latitud_sismo"], evento["Longitud_sismo"]
    m = folium.Map(location=[lat_s, lon_s], zoom_start=6, tiles="OpenStreetMap")
    
    folium.Marker(
        location=[lat_s, lon_s],
        popup=f"Epicentro\nM:{evento['magnitud']}\nProf:{evento['Profundidad']}km",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    for x in preds[:n]:
        i = int(x["intensidad_predicha"])
        col = "#d32f2f" if i >= 6 else "#f57c00" if i >= 4 else "#2e7d32"
        folium.CircleMarker(
            location=[x["Latitud_localidad"], x["Longitud_localidad"]],
            radius=4 + 3 * i, color=col, fill=True, fill_color=col, fill_opacity=0.55,
            popup=f"<b>{x['localidad']}</b><br>Intensidad: {i}",
        ).add_to(m)
    return m.get_root().render()

# -------------------------
# Endpoints
# -------------------------
@app.on_event("startup")
def startup():
    if PRELOAD_MODEL_ON_STARTUP: load_model()

def fetch_latest_event(min_mag: float):
    resp = requests.get(XOR_API_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    events = data if isinstance(data, list) else safe_get(data, ["data", "events"]) or [data]
    
    cutoff = datetime.utcnow() - timedelta(hours=48)
    for ev in events:
        mag, unit = extract_magnitude(ev)
        lat, lon = extract_lat_lon(ev)
        fecha = extract_datetime(ev)
        try:
            m_f, la_f, lo_f = _to_float(mag), _to_float(lat), _to_float(lon)
            dt_obj = datetime.strptime(fecha, "%d-%m-%Y %H:%M:%S")
            if m_f >= min_mag and dt_obj >= cutoff:
                return {
                    "Latitud_sismo": la_f, "Longitud_sismo": lo_f, "magnitud": m_f,
                    "Profundidad": _to_float(safe_get(ev, ["depth", "profundidad"]) or 0),
                    "FechaHora": fecha, "Referencia": extract_georef(ev), "mag_type": unit or "Mw"
                }
        except: continue
    raise RuntimeError("No hay sismos recientes.")

@app.get("/", response_class=HTMLResponse)
def home(n: int = Query(DEFAULT_TABLE_ROWS, ge=1, le=2000)):
    try:
        evento = fetch_latest_event(MIN_EVENT_MAGNITUDE)
    except:
        return f"<html><body style='font-family:Arial; padding:24px;'>{render_header_html()}<div>No se hallaron sismos de M>={MIN_EVENT_MAGNITUDE} en 48h.</div></body></html>"

    preds, _ = predict_intensidades(evento, MIN_INTENSITY_TO_SHOW)
    table_html = render_table(preds, n)
    map_html = build_map_html(evento, preds, n)
    srcdoc = map_html.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

    return f"""
    <html>
      <head><meta charset="utf-8"><title>YATI</title></head>
      <body style="font-family: Arial, sans-serif; padding: 24px;">
        {render_header_html()}
        <h2>Último sismo de magnitud igual o mayor a {MIN_EVENT_MAGNITUDE} en 48 hrs.</h2>
        <ul>
          <li><b>Fecha/Hora:</b> {evento['FechaHora']}</li>
          <li><b>Latitud_sismo:</b> {evento['Latitud_sismo']}</li>
          <li><b>Longitud_sismo:</b> {evento['Longitud_sismo']}</li>
          <li><b>Profundidad (km):</b> {evento['Profundidad']}</li>
          <li><b>Magnitud:</b> {evento['magnitud']} ({evento['mag_type']})</li>
          <li><b>Referencia:</b> {evento['Referencia']}</li>
        </ul>
        <div style="margin: 10px 0 18px 0;"><b>Fuente:</b> <a href="https://www.sismologia.cl/">https://www.sismologia.cl/</a></div>
        <h2>Intensidades Mercalli estimadas iguales o mayores a {MIN_INTENSITY_TO_SHOW}</h2>
        {table_html}
        <h2 style="margin-top: 24px;">Mapa (Epicentro + localidades)</h2>
        <iframe srcdoc="{srcdoc}" style="width:100%; height:650px; border:0; border-radius:10px;" loading="lazy"></iframe>
      </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
