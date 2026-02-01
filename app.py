import os
import re
import csv
import math
import threading

import requests
import numpy as np
import joblib

from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# ============================================================
# CONFIGURACIÓN (MODIFICABLE)
# ============================================================
MIN_EVENT_MAGNITUDE = float(os.getenv("MIN_EVENT_MAGNITUDE", "0.0"))     # evento: M >=
MIN_INTENSITY_TO_SHOW = int(os.getenv("MIN_INTENSITY_TO_SHOW", "2"))     # mostrar: I >=
MAX_EVENTS_TO_SCAN = int(os.getenv("MAX_EVENTS_TO_SCAN", "25"))          # cuántos informes revisar
DEFAULT_TABLE_ROWS = int(os.getenv("DEFAULT_TABLE_ROWS", "200"))         # filas mostradas en Home

PRELOAD_MODEL_ON_STARTUP = os.getenv("PRELOAD_MODEL_ON_STARTUP", "1") == "1"

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "25"))
MODEL_DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "600"))
# ============================================================

BASE_URL = "https://www.sismologia.cl/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RailwayBot/1.0; +https://railway.app)"}

CSV_PATH = os.getenv("LOCALIDADES_CSV", "Localidades_Enero_2026_con_coords_region_comuna.csv")

MODEL_PATH = os.getenv("MODEL_PATH", "Sismos_RF_joblib_Ene_2026.pkl")
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/juansotodaniels/sismos-railway/releases/download/v1.0/Sismos_RF_joblib_Ene_2026.pkl"
)

app = FastAPI(title="Último sismo + distancias + intensidades (RF)")

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
# Scraping: último sismo que cumpla magnitud mínima
# -------------------------
def parse_event_from_informe(informe_url: str) -> dict:
    r2 = requests.get(informe_url, headers=HEADERS, timeout=HTTP_TIMEOUT)
    r2.raise_for_status()
    text = BeautifulSoup(r2.text, "html.parser").get_text("\n", strip=True)

    lat_m = re.search(r"Latitud\s*([-+]?\d+(?:[.,]\d+)?)", text)
    lon_m = re.search(r"Longitud\s*([-+]?\d+(?:[.,]\d+)?)", text)
    prof_m = re.search(r"Profundidad\s*(\d+(?:[.,]\d+)?)\s*km", text, re.IGNORECASE)
    mag_m = re.search(r"Magnitud\s*([-+]?\d+(?:[.,]\d+)?)", text)

    if not (lat_m and lon_m and prof_m and mag_m):
        raise RuntimeError("No se pudieron extraer todos los campos (lat/lon/prof/mag).")

    return {
        "Latitud_sismo": _to_float(lat_m.group(1)),
        "Longitud_sismo": _to_float(lon_m.group(1)),
        "Profundidad": _to_float(prof_m.group(1)),
        "magnitud": _to_float(mag_m.group(1)),
        "Fuente_informe": informe_url,
    }

def fetch_latest_event(min_mag: float = MIN_EVENT_MAGNITUDE) -> dict:
    r = requests.get(BASE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = soup.select('a[href^="sismicidad/informes/"]')
    if not links:
        links = soup.find_all("a", href=re.compile(r"sismicidad/informes/"))
    if not links:
        raise RuntimeError("No se encontraron links a informes de sismos en la portada.")

    scanned = 0
    for a in links[:MAX_EVENTS_TO_SCAN]:
        scanned += 1
        informe_url = BASE_URL.rstrip("/") + "/" + a["href"].lstrip("/")
        try:
            evento = parse_event_from_informe(informe_url)
            if evento["magnitud"] >= float(min_mag):
                try:
                    locs = read_localidades(CSV_PATH)
                    evento["Referencia"] = referencia_por_localidad_mas_cercana(evento, locs)
                except Exception:
                    evento["Referencia"] = None
                evento["min_magnitud_usada"] = float(min_mag)
                evento["informes_revisados"] = scanned
                return evento
        except Exception:
            continue

    raise RuntimeError(f"No se encontró un sismo con magnitud >= {min_mag} revisando {scanned} informes.")

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
                "distancia_epicentro_km": round(dist, 2),
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
        f"<tr><td>{i+1}</td>"
        f"<td>{x['localidad']}</td>"
        f"<td>{x.get('comuna','')}</td>"
        f"<td>{x.get('region','')}</td>"
        f"<td>{x['distancia_epicentro_km']}</td>"
        f"<td><b>{x['intensidad_predicha']}</b></td></tr>"
        for i, x in enumerate(show)
    )
    return f"""
      <p>Mostrando <b>{len(show)}</b> filas. (cambia con <code>?n=500</code>)</p>
      <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
        <thead>
          <tr>
            <th>#</th>
            <th>Localidad</th>
            <th>Comuna</th>
            <th>Región</th>
            <th>Distancia (km)</th>
            <th>Intensidad</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    """

# -------------------------
# Endpoints
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(n: int = Query(DEFAULT_TABLE_ROWS, ge=1, le=20000)):
    """
    HOME = resumen del sismo + tabla de intensidades (filtrada) en la MISMA página
    """
    try:
        evento = fetch_latest_event()
        ref = evento.get("Referencia") or "No disponible"

        preds, order = predict_intensidades(evento, MIN_INTENSITY_TO_SHOW)

        table_html = render_table(preds, n)

        html = f"""
        <html>
          <head><meta charset="utf-8"><title>Último sismo + intensidades</title></head>
          <body style="font-family: Arial, sans-serif; padding: 24px;">
            <h2>Último sismo (magnitud ≥ {MIN_EVENT_MAGNITUDE})</h2>
            <ul>
              <li><b>Latitud_sismo:</b> {evento["Latitud_sismo"]}</li>
              <li><b>Longitud_sismo:</b> {evento["Longitud_sismo"]}</li>
              <li><b>Profundidad (km):</b> {evento["Profundidad"]}</li>
              <li><b>Magnitud:</b> {evento["magnitud"]}</li>
              <li><b>Referencia:</b> {ref}</li>
              <li><b>Informes revisados:</b> {evento.get("informes_revisados", "-")}</li>
            </ul>
            <p><b>Fuente:</b> <a href="{evento["Fuente_informe"]}" target="_blank">{evento["Fuente_informe"]}</a></p>

            <hr/>
            <h2>Intensidades estimadas (solo ≥ {MIN_INTENSITY_TO_SHOW})</h2>
            <p>
              ✅ Intensidades redondeadas al entero más cercano<br/>
              ✅ Filtro: intensidad ≥ <b>{MIN_INTENSITY_TO_SHOW}</b><br/>
              Features usadas (orden): <code>{", ".join(order)}</code>
            </p>

            {table_html}

            <p style="margin-top: 16px;">
              <a href="/intensidades/json">Ver JSON</a> |
              <a href="/health">Health</a>
            </p>
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

@app.get("/intensidades", response_class=HTMLResponse)
def intensidades_only(n: int = Query(200, ge=1, le=20000)):
    """
    Mantengo este endpoint por compatibilidad, pero ya no es necesario.
    """
    try:
        evento = fetch_latest_event()
        preds, order = predict_intensidades(evento, MIN_INTENSITY_TO_SHOW)

        ref = evento.get("Referencia") or "No disponible"
        table_html = render_table(preds, n)

        html = f"""
        <html>
          <head><meta charset="utf-8"><title>Intensidades</title></head>
          <body style="font-family: Arial, sans-serif; padding: 24px;">
            <h2>Intensidades (solo ≥ {MIN_INTENSITY_TO_SHOW})</h2>
            <p><b>Referencia:</b> {ref}</p>
            <p>Features: <code>{", ".join(order)}</code></p>
            {table_html}
            <p style="margin-top: 16px;"><a href="/">Volver</a></p>
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
            <h2>Error al calcular intensidades</h2>
            <p style="color:#b00020;"><b>Error:</b> {err}</p>
            <p>Revisa <a href="/health">/health</a>.</p>
            <p><a href="/">Volver</a></p>
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
                "MAX_EVENTS_TO_SCAN": MAX_EVENTS_TO_SCAN,
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
        "MIN_EVENT_MAGNITUDE": MIN_EVENT_MAGNITUDE,
        "MIN_INTENSITY_TO_SHOW": MIN_INTENSITY_TO_SHOW,
        "MAX_EVENTS_TO_SCAN": MAX_EVENTS_TO_SCAN,
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

    return JSONResponse(status)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
