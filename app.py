import os
import re
import csv
import math

import requests
import numpy as np
import joblib

from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

BASE_URL = "https://www.sismologia.cl/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RailwayBot/1.0; +https://railway.app)"}

CSV_PATH = os.getenv("LOCALIDADES_CSV", "Localidades_Enero_2026_con_coords.csv")

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
    """Distancia Haversine (Tierra como esfera), km."""
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
    """
    Convierte salida del modelo a int redondeando al entero más cercano.
    Cualquier NaN/error => 0.
    """
    try:
        v = float(x)
        if np.isnan(v):
            return 0
        return int(round(v))
    except Exception:
        return 0


# -------------------------
# Scraping: último sismo
# -------------------------
def fetch_latest_event() -> dict:
    r = requests.get(BASE_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    first = soup.select_one('a[href^="sismicidad/informes/"]')
    if not first:
        first = soup.find("a", href=re.compile(r"sismicidad/informes/"))
    if not first:
        raise RuntimeError("No se encontró link al informe del último sismo en la portada.")

    informe_url = BASE_URL.rstrip("/") + "/" + first["href"].lstrip("/")

    r2 = requests.get(informe_url, headers=HEADERS, timeout=20)
    r2.raise_for_status()
    soup2 = BeautifulSoup(r2.text, "html.parser")

    # Texto “plano” para regex numéricas
    text = soup2.get_text("\n", strip=True)

    lat_m = re.search(r"Latitud\s*([-+]?\d+(?:[.,]\d+)?)", text)
    lon_m = re.search(r"Longitud\s*([-+]?\d+(?:[.,]\d+)?)", text)
    prof_m = re.search(r"Profundidad\s*(\d+(?:[.,]\d+)?)\s*km", text, re.IGNORECASE)
    mag_m = re.search(r"Magnitud\s*([-+]?\d+(?:[.,]\d+)?)", text)

    if not (lat_m and lon_m and prof_m and mag_m):
        raise RuntimeError("No se pudieron extraer todos los campos (lat/lon/prof/mag).")

    # -------------------------
    # NUEVO: Extraer "Referencia"
    # -------------------------
    # Intento 1: regex exacto "Referencia ..."
    ref = None
    ref_m = re.search(r"Referencia\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if ref_m:
        ref = ref_m.group(1).strip()

    # Intento 2 (fallback): a veces el sitio usa otro label
    # (ajusta/expande si en tu informe aparece con otro nombre)
    if not ref:
        alt_labels = [
            "Referencia geográfica",
            "Referencias",
            "Región",
            "Lugar",
            "Ubicación",
            "Localidad",
        ]
        for lbl in alt_labels:
            m = re.search(rf"{re.escape(lbl)}\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
            if m:
                ref = m.group(1).strip()
                break

    # Limpieza mínima: corta si viene con saltos raros pegados
    if ref:
        # Si por algún motivo quedó con más de una línea pegada, toma la primera
        ref = ref.split("\n")[0].strip()

    return {
        "Latitud_sismo": _to_float(lat_m.group(1)),
        "Longitud_sismo": _to_float(lon_m.group(1)),
        "Profundidad": _to_float(prof_m.group(1)),
        "magnitud": _to_float(mag_m.group(1)),
        "Referencia": ref,  # ✅ NUEVO CAMPO
        "Fuente_informe": informe_url,
    }

# -------------------------
# Lectura CSV: localidades
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

        def find_col(candidates):
            for cand in candidates:
                for i, col in enumerate(fields_lower):
                    if cand in col:
                        return fields[i]
            return None

        lat_col = find_col(["lat", "latitude", "latitud"])
        lon_col = find_col(["lon", "long", "longitude", "longitud"])
        name_col = find_col(["localidad", "nombre", "name", "ciudad", "comuna", "poblado", "locality"]) or fields[0]

        if not lat_col or not lon_col:
            raise RuntimeError(f"No pude identificar columnas de lat/lon en el CSV. Headers: {fields}")

        locs = []
        for row in reader:
            try:
                lat = _to_float(row.get(lat_col, ""))
                lon = _to_float(row.get(lon_col, ""))
            except Exception:
                continue

            nombre = (row.get(name_col) or "").strip() or "Sin nombre"
            locs.append({"localidad": nombre, "Latitud_localidad": lat, "Longitud_localidad": lon})

    if not locs:
        raise RuntimeError("No se pudieron leer localidades válidas desde el CSV.")
    return locs


# -------------------------
# Descarga + carga del modelo (cache)
# -------------------------
MODEL = None


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
        return

    print(f"[MODEL] Descargando modelo desde: {MODEL_URL}")
    resp = requests.get(MODEL_URL, headers=HEADERS, stream=True, timeout=300)
    resp.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[MODEL] Modelo descargado correctamente ({size_mb:.2f} MB).")


def load_model():
    global MODEL
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
                "Latitud_localidad": loc["Latitud_localidad"],
                "Longitud_localidad": loc["Longitud_localidad"],
                "distancia_epicentro_km": round(dist, 2),
            }
        )

    model = load_model()
    order = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else FEATURES
    X = np.array([[float(r.get(c, np.nan)) for c in order] for r in rows], dtype=float)
    return X, meta, order


def predict_intensidades(evento: dict):
    locs = read_localidades(CSV_PATH)
    X, meta, order = build_feature_matrix(evento, locs)

    model = load_model()
    y_pred = model.predict(X)

    out = []
    for i, m in enumerate(meta):
        # ✅ redondeo al entero más cercano
        intensidad = round_intensity(y_pred[i])

        # ✅ filtrar intensidades <= 0
        if intensidad <= 0:
            continue

        out.append({**m, "intensidad_predicha": intensidad})

    # ordenar: primero por mayor intensidad, y a igual intensidad por menor distancia
    out.sort(key=lambda x: (-x["intensidad_predicha"], x["distancia_epicentro_km"]))
    return out, order


# -------------------------
# Endpoints
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    d = fetch_latest_event()
    html = f"""
    <html>
      <head><meta charset="utf-8"><title>Último sismo</title></head>
      <body style="font-family: Arial, sans-serif; padding: 24px;">
        <h2>Último sismo registrado (sismologia.cl)</h2>
        <ul>
          <li><b>Latitud_sismo:</b> {d["Latitud_sismo"]}</li>
          <li><b>Longitud_sismo:</b> {d["Longitud_sismo"]}</li>
          <li><b>Profundidad (km):</b> {d["Profundidad"]}</li>
          <li><b>Magnitud:</b> {d["magnitud"]}</li>
        </ul>
        <p><b>Fuente:</b> <a href="{d["Fuente_informe"]}" target="_blank">{d["Fuente_informe"]}</a></p>
        <hr/>
        <p>
          <a href="/intensidades">Ver intensidades por localidad (solo > 0)</a> |
          <a href="/intensidades/json">Ver JSON</a> |
          <a href="/health">Health</a>
        </p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/json")
def latest_json():
    return JSONResponse(fetch_latest_event())


@app.get("/intensidades", response_class=HTMLResponse)
def intensidades_html(n: int = Query(200, ge=1, le=20000)):
    evento = fetch_latest_event()
    preds, order = predict_intensidades(evento)

    show = preds[:n]
    rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{x['localidad']}</td><td>{x['distancia_epicentro_km']}</td>"
        f"<td><b>{x['intensidad_predicha']}</b></td><td>{x['Latitud_localidad']}</td><td>{x['Longitud_localidad']}</td></tr>"
        for i, x in enumerate(show)
    )

    html = f"""
    <html>
      <head><meta charset="utf-8"><title>Intensidades por localidad</title></head>
      <body style="font-family: Arial, sans-serif; padding: 24px;">
        <h2>Intensidad estimada por localidad (modelo RF)</h2>
        <p>
          <b>Sismo:</b> lat {evento["Latitud_sismo"]}, lon {evento["Longitud_sismo"]},
          prof {evento["Profundidad"]} km, M {evento["magnitud"]}
          (<a href="{evento["Fuente_informe"]}" target="_blank">fuente</a>)
        </p>
        <p>
          ✅ Intensidades redondeadas al entero más cercano<br/>
          ✅ Se muestran solo localidades con intensidad &gt; 0<br/>
          Features usadas (orden): <code>{", ".join(order)}</code>
        </p>
        <p>Mostrando <b>{len(show)}</b> filas. (cambia con <code>?n=500</code>)</p>

        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>#</th>
              <th>Localidad</th>
              <th>Distancia (km)</th>
              <th>Intensidad (pred)</th>
              <th>Lat</th>
              <th>Lon</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>

        <p style="margin-top: 16px;">
          <a href="/">Volver</a> |
          <a href="/intensidades/json">Ver JSON</a>
        </p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/intensidades/json")
def intensidades_json():
    evento = fetch_latest_event()
    preds, order = predict_intensidades(evento)
    return JSONResponse(
        {
            "evento": evento,
            "csv": CSV_PATH,
            "modelo_local": MODEL_PATH,
            "modelo_url": MODEL_URL,
            "features_orden": order,
            "cantidad_localidades_filtradas_int_gt_0": len(preds),
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
