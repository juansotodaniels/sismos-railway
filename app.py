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
# CONFIGURACIÓN (MODIFICABLE)
# ============================================================
MIN_EVENT_MAGNITUDE = float(os.getenv("MIN_EVENT_MAGNITUDE", "0"))          # evento: M >=
MIN_INTENSITY_TO_SHOW = int(os.getenv("MIN_INTENSITY_TO_SHOW", "2"))        # mostrar: I >=
DEFAULT_TABLE_ROWS = int(os.getenv("DEFAULT_TABLE_ROWS", "200"))            # filas mostradas en Home

PRELOAD_MODEL_ON_STARTUP = os.getenv("PRELOAD_MODEL_ON_STARTUP", "1") == "1"

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "25"))
MODEL_DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "600"))
# ============================================================

# ✅ API XOR
XOR_API_URL = os.getenv("XOR_API_URL", "https://api.xor.cl/sismo/recent")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RailwayBot/1.0; +https://railway.app)"}

# CSV con localidades y coords (se usa para predecir intensidades por localidad)
CSV_PATH = os.getenv("LOCALIDADES_CSV", "Localidades_Enero_2026_con_coords.csv")

MODEL_PATH = os.getenv("MODEL_PATH", "Sismos_RF_joblib_Ene_2026.pkl")
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/juansotodaniels/sismos-railway/releases/download/v1.0/Sismos_RF_joblib_Ene_2026.pkl"
)

app = FastAPI(title="SismoTrack — Último sismo + distancias + intensidades (RF)")

# -------------------------
# Utilidades
# -------------------------
def _to_float(s) -> float:
    s = str(s).strip().replace(",", ".")
    return float(s)

def _clean_txt(x) -> str:
    if x is None:
        return ""
    return str(x).strip()

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def detect_delimiter(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=";,\t|")
        return dialect.delimiter
    except Exception:
        return ";"

def round_intensity(x) -> int:
    try:
        v = float(x)
        if np.isnan(v):
            return 0
        return int(round(v))
    except Exception:
        return 0

def safe_get(d: dict, keys: list[str], default=None):
    """Busca la primera clave existente (case-insensitive) en dict."""
    if not isinstance(d, dict):
        return default
    lower_map = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        kk = str(k).lower()
        if kk in lower_map:
            return d.get(lower_map[kk])
    return default

def parse_datetime_flexible(value):
    """
    Intenta parsear fecha/hora desde varios formatos comunes.
    Devuelve string 'YYYY-MM-DD HH:MM:SS' o el original si no pudo.
    """
    if value is None:
        return None

    s = str(value).strip()

    # epoch (10 o 13 dígitos)
    if s.isdigit() and len(s) in (10, 13):
        try:
            ts = int(s)
            if len(s) == 13:
                ts = ts / 1000
            return datetime.utcfromtimestamp(ts).strftime("%d-%m-%Y %H:%M:%S UTC")
        except Exception:
            pass

    candidates = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

    return s

# -------------------------
# Helpers de parseo XOR (robusto)
# -------------------------
def extract_lat_lon(ev: dict):
    """
    Intenta extraer lat/lon desde múltiples estructuras:
    - ev["lat"], ev["lon"/"lng"]
    - ev["coords"]={"lat":..,"lon":..} o {"latitude":..,"longitude":..}
    - ev["coordinate"] / ev["coordinates"]
    - GeoJSON: ev["geometry"]["coordinates"] = [lon, lat]
    """
    lat = safe_get(ev, ["lat", "latitude", "latitud", "y"])
    lon = safe_get(ev, ["lon", "lng", "long", "longitude", "longitud", "x"])

    if lat is not None and lon is not None:
        return lat, lon

    for k in ["coords", "coord", "coordinate", "coordinates", "location", "pos", "position"]:
        obj = safe_get(ev, [k])
        if isinstance(obj, dict):
            lat2 = safe_get(obj, ["lat", "latitude", "latitud", "y"])
            lon2 = safe_get(obj, ["lon", "lng", "long", "longitude", "longitud", "x"])
            if lat2 is not None and lon2 is not None:
                return lat2, lon2
        elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
            a, b = obj[0], obj[1]
            try:
                a_f = _to_float(a); b_f = _to_float(b)
                # si |a|>90 probablemente es lon primero
                if abs(a_f) > 90 and abs(b_f) <= 90:
                    return b, a  # [lon, lat]
                return a, b    # [lat, lon]
            except Exception:
                pass

    geom = safe_get(ev, ["geometry"])
    if isinstance(geom, dict):
        coords = safe_get(geom, ["coordinates"])
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return coords[1], coords[0]  # GeoJSON => [lon, lat]

    return None, None

def extract_magnitude(ev: dict):
    """
    Soporta magnitud como:
    - número/string
    - dict {"value":4.1,"measure_unit":"Ml"} (estilo CSN)
    """
    mag = safe_get(ev, ["magnitude", "magnitud", "mag", "m"])
    mag_unit = None

    if isinstance(mag, dict):
        mag_unit = safe_get(mag, ["measure_unit", "unit", "type"])
        mag = safe_get(mag, ["value", "val", "magnitude", "magnitud", "mag", "m"])

    return mag, mag_unit

def extract_datetime(ev: dict):
    """
    Busca fecha/hora en múltiples claves comunes, incluyendo estilo CSN:
    - local_date / utc_date
    - date/datetime/time/timestamp/created_at/updated_at
    También soporta que venga anidado como dict.
    """
    dt_raw = (
        safe_get(ev, ["local_date", "fecha_local", "hora_local", "localdatetime"]) or
        safe_get(ev, ["utc_date", "fecha_utc", "hora_utc", "utcdatetime"]) or
        safe_get(ev, ["date", "datetime", "time", "timestamp", "created_at", "updated_at"]) or
        safe_get(ev, ["fecha", "hora"])
    )

    if isinstance(dt_raw, dict):
        dt_raw = (
            safe_get(dt_raw, ["local_date", "utc_date", "date", "datetime", "time", "timestamp", "value"])
            or str(dt_raw)
        )

    return parse_datetime_flexible(dt_raw) or "No disponible"

def extract_georef(ev: dict):
    """
    Extrae referencia geográfica/ubicación del evento desde varias claves:
    - geo_reference (CSN)
    - reference/ref/place/location/ubicacion
    - también soporta dict anidado
    """
    g = (
        safe_get(ev, ["geo_reference", "georeference", "reference", "ref", "place", "location", "ubicacion", "zona"]) or
        safe_get(ev, ["georef", "geo_ref"])
    )

    if isinstance(g, dict):
        g = safe_get(g, ["geo_reference", "reference", "place", "location", "value"]) or str(g)

    return _clean_txt(g) or "No disponible"


# -------------------------
# CSV: localidades (para predicción)
# -------------------------
def read_localidades(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        raise RuntimeError(f"No existe el archivo CSV de localidades en: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    delim = detect_delimiter(sample)

    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            raise RuntimeError("El CSV no tiene encabezados (headers).")

        fields = [c.strip() for c in reader.fieldnames]
        fields_lower = [c.lower() for c in fields]

        def find_col_contains(candidates):
            for cand in candidates:
                cand = cand.lower()
                for i, col in enumerate(fields_lower):
                    if cand in col:
                        return fields[i]
            return None

        lat_col = find_col_contains(["lat", "latitude", "latitud"])
        lon_col = find_col_contains(["lon", "long", "longitude", "longitud"])
        name_col = find_col_contains(["localidad", "nombre", "name", "ciudad", "poblado", "locality"]) or fields[0]
        comuna_col = find_col_contains(["comuna"])
        region_col = find_col_contains(["región", "region"])

        if not lat_col or not lon_col:
            raise RuntimeError(f"No pude identificar columnas de lat/lon en el CSV. Headers: {fields}")

        locs = []
        for row in reader:
            try:
                lat = _to_float(row.get(lat_col, ""))
                lon = _to_float(row.get(lon_col, ""))
            except Exception:
                continue

            nombre = _clean_txt(row.get(name_col)) or "Sin nombre"
            comuna = _clean_txt(row.get(comuna_col)) if comuna_col else ""
            region = _clean_txt(row.get(region_col)) if region_col else ""

            locs.append(
                {
                    "localidad": nombre,
                    "Latitud_localidad": lat,
                    "Longitud_localidad": lon,
                    "comuna": comuna,
                    "region": region,
                }
            )

    if not locs:
        raise RuntimeError("No se pudieron leer localidades válidas desde el CSV.")
    return locs


# -------------------------
# API XOR: último sismo >= magnitud mínima
#   ✅ Usa SOLO la referencia geográfica del JSON
#   ❌ NO calcula localidad más cercana
# -------------------------
def fetch_latest_event(min_mag: float = MIN_EVENT_MAGNITUDE) -> dict:
    resp = requests.get(XOR_API_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    events = None
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict):
        events = (
            safe_get(data, ["data"]) or
            safe_get(data, ["events"]) or
            safe_get(data, ["result"]) or
            safe_get(data, ["results"])
        )
        if events is None and any(k in data for k in ["lat", "lon", "latitude", "longitude", "magnitude", "magnitud", "mag"]):
            events = [data]

    if not isinstance(events, list) or not events:
        raise RuntimeError("La API XOR no devolvió una lista de sismos válida.")

    parse_fails = 0

    for ev in events:
        if not isinstance(ev, dict):
            continue

        mag_raw, mag_unit = extract_magnitude(ev)
        lat_raw, lon_raw = extract_lat_lon(ev)
        depth_raw = safe_get(ev, ["depth", "profundidad", "depth_km"])
        fecha = extract_datetime(ev)
        geo_ref = extract_georef(ev)

        try:
            mag_f = _to_float(mag_raw)
            lat_f = _to_float(lat_raw)
            lon_f = _to_float(lon_raw)
            depth_f = _to_float(depth_raw) if depth_raw is not None and str(depth_raw).strip() != "" else 0.0
        except Exception as e:
            parse_fails += 1
            if parse_fails <= 5:
                print("[PARSE FAIL] keys=", list(ev.keys()))
                print("[PARSE FAIL] mag_raw=", mag_raw)
                print("[PARSE FAIL] lat_raw=", lat_raw, "lon_raw=", lon_raw, "depth_raw=", depth_raw)
                print("[PARSE FAIL] err=", repr(e))
            continue

        if mag_f < float(min_mag):
            continue

        return {
            "Latitud_sismo": lat_f,
            "Longitud_sismo": lon_f,
            "Profundidad": depth_f,
            "magnitud": mag_f,
            "mag_type": mag_unit or safe_get(ev, ["mag_type", "magnitude_type", "tipo"]) or "No disponible",
            "Fuente_informe": XOR_API_URL,
            "FechaHora": fecha,
            "Referencia": geo_ref,        # ✅ solo georef del JSON
            "min_magnitud_usada": float(min_mag),
        }

    raise RuntimeError(
        f"No se encontró un sismo con magnitud >= {min_mag} en la API XOR. "
        f"(se descartaron {parse_fails} eventos por parseo)."
    )


# -------------------------
# Modelo: descarga + carga (con lock)
# -------------------------
MODEL = None
MODEL_LOCK = threading.Lock()

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
        return

    print(f"[MODEL] Descargando modelo desde: {MODEL_URL}")
    resp = requests.get(MODEL_URL, headers=HEADERS, stream=True, timeout=MODEL_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[MODEL] Modelo descargado correctamente ({size_mb:.2f} MB).")

def load_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            ensure_model()
            MODEL = joblib.load(MODEL_PATH)
            print("[MODEL] Modelo cargado en memoria.")
    return MODEL

FEATURES = [
    "Latitud_sismo",
    "Longitud_sismo",
    "Profundidad",
    "magnitud",
    "Latitud_localidad",
    "Longitud_localidad",
    "distancia_epicentro",
]

def build_feature_matrix(evento: dict, locs: list[dict]):
    lat_s = evento["Latitud_sismo"]
    lon_s = evento["Longitud_sismo"]

    rows = []
    meta = []
    for loc in locs:
        dist = haversine_km(lat_s, lon_s, loc["Latitud_localidad"], loc["Longitud_localidad"])

        feats = {
            "Latitud_sismo": lat_s,
            "Longitud_sismo": lon_s,
            "Profundidad": evento["Profundidad"],
            "magnitud": evento["magnitud"],
            "Latitud_localidad": loc["Latitud_localidad"],
            "Longitud_localidad": loc["Longitud_localidad"],
            "distancia_epicentro": dist,
        }
        rows.append(feats)

        meta.append(
            {
                "localidad": loc["localidad"],
                "comuna": loc.get("comuna", ""),
                "region": loc.get("region", ""),
                "Latitud_localidad": loc["Latitud_localidad"],
                "Longitud_localidad": loc["Longitud_localidad"],
                "distancia_epicentro_km": int(round(dist)),
            }
        )

    model = load_model()
    order = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else FEATURES
    X = np.array([[float(r.get(c, np.nan)) for c in order] for r in rows], dtype=float)
    return X, meta, order

def predict_intensidades(evento: dict, min_intensity: int = MIN_INTENSITY_TO_SHOW):
    locs = read_localidades(CSV_PATH)
    X, meta, order = build_feature_matrix(evento, locs)

    model = load_model()
    y_pred = model.predict(X)

    out = []
    for i, m in enumerate(meta):
        intensidad = round_intensity(y_pred[i])
        if intensidad < int(min_intensity):
            continue
        out.append({**m, "intensidad_predicha": intensidad})

    out.sort(key=lambda x: (-x["intensidad_predicha"], x["distancia_epicentro_km"]))
    return out, order


# -------------------------
# Startup: precarga del modelo
# -------------------------
@app.on_event("startup")
def startup():
    if PRELOAD_MODEL_ON_STARTUP:
        try:
            print("[STARTUP] Precargando modelo...")
            load_model()
            print("[STARTUP] OK: modelo listo.")
        except Exception as e:
            print(f"[STARTUP] WARNING: no pude precargar modelo: {e}")


# -------------------------
# Render HTML tabla (reutilizable)
# -------------------------
def render_table(preds: list[dict], n: int) -> str:
    show = preds[:n]
    rows = "\n".join(
        f"<tr>"
        f"<td style='text-align:center;'>{i+1}</td>"
        f"<td>{x['localidad']}</td>"
        f"<td>{x.get('comuna','')}</td>"
        f"<td>{x.get('region','')}</td>"
        f"<td style='text-align:center;'>{x['distancia_epicentro_km']}</td>"
        f"<td style='text-align:center;'><b>{x['intensidad_predicha']}</b></td>"
        f"</tr>"
        for i, x in enumerate(show)
    )
    return f"""
      <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%;">
        <thead>
          <tr>
            <th>#</th>
            <th>Localidad</th>
            <th>Comuna</th>
            <th>Región</th>
            <th>Distancia al epicentro (km)</th>
            <th>Intensidad (Mercalli)</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    """


# -------------------------
# Mapa Folium embebido
# -------------------------
def intensity_color(i: int) -> str:
    if i >= 6:
        return "#d32f2f"
    if i >= 4:
        return "#f57c00"
    return "#2e7d32"

def intensity_radius(i: int) -> int:
    return 4 + 3 * int(i)

def build_map_html(evento: dict, preds: list[dict], n: int) -> str:
    show = preds[:n]

    lat_s = evento["Latitud_sismo"]
    lon_s = evento["Longitud_sismo"]

    m = folium.Map(location=[lat_s, lon_s], zoom_start=6, tiles="OpenStreetMap")

    ref = evento.get("Referencia") or "No disponible"
    tooltip_html = (
        f"<div style='font-size:13px;'>"
        f"<b>Sismo</b><br>"
        f"<b>Fecha/Hora:</b> {evento.get('FechaHora','No disponible')}<br>"
        f"<b>Magnitud:</b> {evento['magnitud']}<br>"
        f"<b>Epicentro:</b> ({lat_s}, {lon_s})<br>"
        f"<b>Referencia:</b> {ref}"
        f"</div>"
    )

    folium.Marker(
        location=[lat_s, lon_s],
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
        popup=folium.Popup(
            f"<b>Epicentro</b><br>"
            f"Fecha/Hora: {evento.get('FechaHora','No disponible')}<br>"
            f"Lat: {lat_s}<br>Lon: {lon_s}<br>"
            f"Prof: {evento['Profundidad']} km<br>M: {evento['magnitud']}<br>"
            f"Ref: {ref}",
            max_width=320
        ),
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    bounds = [[lat_s, lon_s]]

    for x in show:
        i = int(x["intensidad_predicha"])
        lat = float(x["Latitud_localidad"])
        lon = float(x["Longitud_localidad"])

        comuna = x.get("comuna") or "No disponible"
        region = x.get("region") or "No disponible"
        dist = x.get("distancia_epicentro_km", "")

        popup = (
            f"<b>{x['localidad']}</b><br>"
            f"Comuna: {comuna}<br>"
            f"Región: {region}<br>"
            f"Distancia: {dist} km<br>"
            f"Intensidad: <b>{i}</b>"
        )

        col = intensity_color(i)
        folium.CircleMarker(
            location=[lat, lon],
            radius=intensity_radius(i),
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.55,
            weight=2,
            popup=folium.Popup(popup, max_width=320),
            tooltip=f"{x['localidad']} (I={i})",
        ).add_to(m)

        bounds.append([lat, lon])

    if len(bounds) >= 2:
        m.fit_bounds(bounds, padding=(20, 20))

    return m.get_root().render()


# -------------------------
# Endpoints
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(n: int = Query(DEFAULT_TABLE_ROWS, ge=1, le=20000)):
    try:
        evento = fetch_latest_event(MIN_EVENT_MAGNITUDE)
        preds, order = predict_intensidades(evento, MIN_INTENSITY_TO_SHOW)

        table_html = render_table(preds, n)

        map_html = build_map_html(evento, preds, n)
        srcdoc = (
            map_html.replace("&", "&amp;")
                    .replace('"', "&quot;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
        )

        html = f"""
        <html>
          <head>
            <meta charset="utf-8">
            <title>SismoTrack</title>
          </head>
          <body style="font-family: Arial, sans-serif; padding: 24px;">
            <h1 style="margin-bottom: 6px;">SismoTrack</h1>
            <div style="margin-bottom: 18px; color:#333;">
              Sistema de estimación temprana de intensidades sísmicas (Chile)
            </div>

            <h2>Último sismo (API XOR | M ≥ {MIN_EVENT_MAGNITUDE})</h2>
            <ul>
              <li><b>Fecha/Hora:</b> {evento.get("FechaHora","No disponible")}</li>
              <li><b>Latitud_sismo:</b> {evento["Latitud_sismo"]}</li>
              <li><b>Longitud_sismo:</b> {evento["Longitud_sismo"]}</li>
              <li><b>Profundidad (km):</b> {evento["Profundidad"]}</li>
              <li><b>Magnitud:</b> {evento["magnitud"]} ({evento.get("mag_type","")})</li>
              <li><b>Referencia:</b> {evento.get("Referencia") or "No disponible"}</li>
            </ul>

            <div style="margin: 10px 0 18px 0;">
              <b>Fuente:</b> <a href="{XOR_API_URL}" target="_blank">{XOR_API_URL}</a>
            </div>

            <h2>Intensidades Mercalli estimadas (I ≥ {MIN_INTENSITY_TO_SHOW})</h2>
            {table_html}

            <h2 style="margin-top: 24px;">Mapa (Epicentro + localidades)</h2>
            <div style="margin: 6px 0 12px 0; color:#333;">
              El tamaño del círculo es proporcional a la intensidad y el color depende del rango.
            </div>

            <iframe
              srcdoc="{srcdoc}"
              style="width:100%; height:650px; border:0; border-radius:10px;"
              loading="lazy"
            ></iframe>

            <div style="margin-top:16px;">
              <a href="/intensidades/json">Ver JSON</a> |
              <a href="/health">Health</a> |
              <a href="/debug/xor">Debug XOR</a>
            </div>

          </body>
        </html>
        """
        return HTMLResponse(content=html)

    except Exception as e:
        err = str(e)
        html = f"""
        <html>
          <head><meta charset="utf-8"><title>Error</title></head>
          <body style="font-family: Arial, sans-serif; padding: 24px;">
            <h2>Ocurrió un error al construir la página</h2>
            <p style="color:#b00020;"><b>Error:</b> {err}</p>
            <p>Revisa <a href="/health">/health</a>.</p>
          </body>
        </html>
        """
        return HTMLResponse(content=html, status_code=500)


@app.get("/intensidades/json")
def intensidades_json():
    evento = fetch_latest_event(MIN_EVENT_MAGNITUDE)
    preds, order = predict_intensidades(evento, MIN_INTENSITY_TO_SHOW)
    return JSONResponse(
        {
            "config": {
                "MIN_EVENT_MAGNITUDE": MIN_EVENT_MAGNITUDE,
                "MIN_INTENSITY_TO_SHOW": MIN_INTENSITY_TO_SHOW,
            },
            "evento": evento,
            "csv": CSV_PATH,
            "modelo_local": MODEL_PATH,
            "modelo_url": MODEL_URL,
            "features_orden": order,
            "cantidad_localidades_int_ge_min": len(preds),
            "resultados": preds,
        }
    )


@app.get("/health")
def health():
    status = {
        "csv_exists": os.path.exists(CSV_PATH),
        "csv_path": CSV_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
        "model_url": MODEL_URL,
        "XOR_API_URL": XOR_API_URL,
        "MIN_EVENT_MAGNITUDE": MIN_EVENT_MAGNITUDE,
        "MIN_INTENSITY_TO_SHOW": MIN_INTENSITY_TO_SHOW,
        "PRELOAD_MODEL_ON_STARTUP": PRELOAD_MODEL_ON_STARTUP,
        "DEFAULT_TABLE_ROWS": DEFAULT_TABLE_ROWS,
    }
    if status["model_exists"]:
        status["model_size_mb"] = round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2)

    try:
        r = requests.get(XOR_API_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
        status["api_ok"] = r.ok
        status["api_status_code"] = r.status_code
    except Exception as e:
        status["api_ok"] = False
        status["api_error"] = str(e)

    try:
        _ = load_model()
        status["model_load_ok"] = True
    except Exception as e:
        status["model_load_ok"] = False
        status["model_load_error"] = str(e)

    return JSONResponse(status)


@app.get("/debug/xor")
def debug_xor(limit: int = Query(3, ge=1, le=20)):
    """
    Devuelve una muestra del JSON crudo de XOR para debug.
    """
    r = requests.get(XOR_API_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, list):
        return JSONResponse({"type": "list", "sample": data[:limit]})

    if isinstance(data, dict):
        events = (
            safe_get(data, ["data"]) or
            safe_get(data, ["events"]) or
            safe_get(data, ["result"]) or
            safe_get(data, ["results"])
        )
        if isinstance(events, list):
            return JSONResponse({"type": "dict+list", "keys": list(data.keys()), "sample": events[:limit]})
        return JSONResponse({"type": "dict", "keys": list(data.keys()), "data": data})

    return JSONResponse({"type": str(type(data))})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
