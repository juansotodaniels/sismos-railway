import os
import csv
import math
import threading

import requests
import numpy as np
import joblib
import folium

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# ============================================================
# CONFIGURACIÓN (MODIFICABLE)
# ============================================================
MIN_EVENT_MAGNITUDE = float(os.getenv("MIN_EVENT_MAGNITUDE", "0"))       # evento: M >=
MIN_INTENSITY_TO_SHOW = int(os.getenv("MIN_INTENSITY_TO_SHOW", "2"))     # mostrar: I >=
DEFAULT_TABLE_ROWS = int(os.getenv("DEFAULT_TABLE_ROWS", "200"))         # filas mostradas en Home

PRELOAD_MODEL_ON_STARTUP = os.getenv("PRELOAD_MODEL_ON_STARTUP", "1") == "1"

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "25"))
MODEL_DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "600"))
# ============================================================

# ✅ NUEVO: API de sismos recientes
SISMOS_API_URL = os.getenv("SISMOS_API_URL", "https://api.xor.cl/sismo/recent")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RailwayBot/1.0; +https://railway.app)"}

CSV_PATH = os.getenv("LOCALIDADES_CSV", "Localidades_Enero_2026_con_coords.csv")

MODEL_PATH = os.getenv("MODEL_PATH", "Sismos_RF_joblib_Ene_2026.pkl")
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/juansotodaniels/sismos-railway/releases/download/v1.0/Sismos_RF_joblib_Ene_2026.pkl"
)

app = FastAPI(title="SismoTrack — Intensidades (RF) + Mapa")

# -------------------------
# Utilidades
# -------------------------
def _to_float(s: str) -> float:
    s = str(s).strip().replace(",", ".")
    return float(s)

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

def _clean_txt(x) -> str:
    if x is None:
        return ""
    return str(x).strip()

# -------------------------
# CSV: localidades
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

def referencia_por_localidad_mas_cercana(evento: dict, locs: list[dict]) -> str | None:
    lat_s = evento["Latitud_sismo"]
    lon_s = evento["Longitud_sismo"]

    best = None
    best_d = float("inf")
    for loc in locs:
        d = haversine_km(lat_s, lon_s, loc["Latitud_localidad"], loc["Longitud_localidad"])
        if d < best_d:
            best_d = d
            best = loc

    if best is None:
        return None

    comuna = best.get("comuna") or "No disponible"
    region = best.get("region") or "No disponible"
    return f"{round(best_d, 1)} km de {best['localidad']} (Comuna: {comuna}, Región: {region})"

# -------------------------
# ✅ NUEVO: Obtener último sismo desde API xor.cl
# -------------------------
def fetch_latest_event(min_mag: float = MIN_EVENT_MAGNITUDE) -> dict:
    """
    Usa https://api.xor.cl/sismo/recent
    Respuesta típica: { status_code, status_description, events:[{...}] }
    Campos útiles: local_date, utc_date, latitude, longitude, depth, magnitude:{value, measure_unit}, url, map_url
    """
    params = {"magnitude": float(min_mag)}
    r = requests.get(SISMOS_API_URL, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    events = data.get("events") or []
    if not events:
        raise RuntimeError(f"La API no devolvió eventos para magnitude >= {min_mag}.")
    ev = events[0]  # el más reciente

    mag_obj = ev.get("magnitude") or {}
    mag_val = mag_obj.get("value", None)
    if mag_val is None:
        # fallback por si viene plano
        mag_val = ev.get("magnitude", None)

    evento = {
        "Latitud_sismo": float(ev.get("latitude")),
        "Longitud_sismo": float(ev.get("longitude")),
        "Profundidad": float(ev.get("depth")),
        "magnitud": float(mag_val),
        "FechaLocal": ev.get("local_date") or "",
        "FechaUTC": ev.get("utc_date") or "",
        "Lugar": ev.get("id") or "",  # a veces el "id" es un texto tipo "XX km al ...", si no, queda vacío
        "Fuente_informe": ev.get("url") or SISMOS_API_URL,
        "MapURL": ev.get("map_url") or "",
        "min_magnitud_usada": float(min_mag),
    }

    # Referencia (localidad más cercana con región/comuna)
    try:
        locs = read_localidades(CSV_PATH)
        evento["Referencia"] = referencia_por_localidad_mas_cercana(evento, locs)
    except Exception:
        evento["Referencia"] = None

    return evento

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
        f"<tr><td style='text-align:center;'>{i+1}</td>"
        f"<td>{x['localidad']}</td>"
        f"<td>{x.get('comuna','')}</td>"
        f"<td>{x.get('region','')}</td>"
        f"<td style='text-align:center;'>{x['distancia_epicentro_km']}</td>"
        f"<td style='text-align:center;'><b>{x['intensidad_predicha']}</b></td></tr>"
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
    fecha_local = evento.get("FechaLocal") or "No disponible"
    lugar = evento.get("Lugar") or ""

    tooltip_html = (
        f"<div style='font-size:13px;'>"
        f"<b>Sismo</b><br>"
        f"<b>Fecha (local):</b> {fecha_local}<br>"
        f"<b>Magnitud:</b> {evento['magnitud']}<br>"
        f"<b>Epicentro:</b> ({lat_s}, {lon_s})<br>"
        f"<b>Referencia:</b> {ref}"
        f"</div>"
    )

    popup_txt = (
        f"<b>Epicentro</b><br>"
        f"Fecha local: {fecha_local}<br>"
        f"Lat: {lat_s}<br>Lon: {lon_s}<br>"
        f"Prof: {evento['Profundidad']} km<br>"
        f"M: {evento['magnitud']}<br>"
        f"{('Lugar: ' + lugar + '<br>') if lugar else ''}"
        f"{('<a href=' + '\"' + evento['Fuente_informe'] + '\"' + ' target=\"_blank\">Fuente</a>') if evento.get('Fuente_informe') else ''}"
    )

    folium.Marker(
        location=[lat_s, lon_s],
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
        popup=folium.Popup(popup_txt, max_width=320),
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
        evento = fetch_latest_event()
        ref = evento.get("Referencia") or "No disponible"

        preds, order = predict_intensidades(evento, MIN_INTENSITY_TO_SHOW)
        table_html = render_table(preds, n)

        map_html = build_map_html(evento, preds, n)
        srcdoc = (
            map_html.replace("&", "&amp;")
                    .replace('"', "&quot;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
        )

        fecha_local = evento.get("FechaLocal") or "No disponible"
        fecha_utc = evento.get("FechaUTC") or ""

        html = f"""
        <html>
          <head><meta charset="utf-8"><title>SismoTrack</title></head>
          <body style="font-family: Arial, sans-serif; padding: 24px;">
            <h1>SismoTrack</h1>
            <h4>Sistema de estimación temprana de intensidades (Chile)</h4>

            <h2>Último sismo (API xor.cl, M ≥ {MIN_EVENT_MAGNITUDE})</h2>
            <ul>
              <li><b>Fecha/Hora local:</b> {fecha_local}</li>
              <li><b>Fecha/Hora UTC:</b> {fecha_utc or "No disponible"}</li>
              <li><b>Latitud_sismo:</b> {evento["Latitud_sismo"]}</li>
              <li><b>Longitud_sismo:</b> {evento["Longitud_sismo"]}</li>
              <li><b>Profundidad (km):</b> {evento["Profundidad"]}</li>
              <li><b>Magnitud:</b> {evento["magnitud"]}</li>
              <li><b>Referencia:</b> {ref}</li>
            </ul>

            <p><b>Fuente evento:</b> <a href="{evento["Fuente_informe"]}" target="_blank">{evento["Fuente_informe"]}</a></p>

            <hr/>
            <h2>Intensidades Mercalli estimadas (I ≥ {MIN_INTENSITY_TO_SHOW})</h2>
            {table_html}

            <h2 style="margin-top: 24px;">Mapa (Epicentro + localidades)</h2>
            <iframe
              srcdoc="{srcdoc}"
              style="width:100%; height:650px; border:1px solid #ccc; border-radius:8px;"
              loading="lazy"
            ></iframe>

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
    evento = fetch_latest_event()
    preds, order = predict_intensidades(evento, MIN_INTENSITY_TO_SHOW)
    return JSONResponse(
        {
            "config": {
                "MIN_EVENT_MAGNITUDE": MIN_EVENT_MAGNITUDE,
                "MIN_INTENSITY_TO_SHOW": MIN_INTENSITY_TO_SHOW,
                "SISMOS_API_URL": SISMOS_API_URL,
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
        "SISMOS_API_URL": SISMOS_API_URL,
        "MIN_EVENT_MAGNITUDE": MIN_EVENT_MAGNITUDE,
        "MIN_INTENSITY_TO_SHOW": MIN_INTENSITY_TO_SHOW,
        "PRELOAD_MODEL_ON_STARTUP": PRELOAD_MODEL_ON_STARTUP,
        "DEFAULT_TABLE_ROWS": DEFAULT_TABLE_ROWS,
    }
    if status["model_exists"]:
        status["model_size_mb"] = round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2)

    try:
        _ = load_model()
        status["model_load_ok"] = True
    except Exception as e:
        status["model_load_ok"] = False
        status["model_load_error"] = str(e)

    # prueba rápida API
    try:
        ev = fetch_latest_event(MIN_EVENT_MAGNITUDE)
        status["api_ok"] = True
        status["api_last_local_date"] = ev.get("FechaLocal")
        status["api_last_magnitude"] = ev.get("magnitud")
    except Exception as e:
        status["api_ok"] = False
        status["api_error"] = str(e)

    return JSONResponse(status)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


