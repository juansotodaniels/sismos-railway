import os
import re
import csv
import math
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

BASE_URL = "https://www.sismologia.cl/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RailwayBot/1.0; +https://railway.app)"}

CSV_PATH = os.getenv("LOCALIDADES_CSV", "Localidades_Enero_2026_con_coords.csv")

app = FastAPI(title="Último sismo + distancias a localidades")


def _to_float(s: str) -> float:
    s = str(s).strip().replace(",", ".")
    return float(s)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distancia Haversine (Tierra como esfera).
    Retorna km.
    """
    R = 6371.0  # radio medio Tierra en km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def fetch_latest_event() -> dict:
    # 1) Portada: sacar el primer link al informe
    r = requests.get(BASE_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    first = soup.select_one('a[href^="sismicidad/informes/"]')
    if not first:
        first = soup.find("a", href=re.compile(r"sismicidad/informes/"))
    if not first:
        raise RuntimeError("No se encontró link al informe del último sismo en la portada.")

    informe_url = BASE_URL.rstrip("/") + "/" + first["href"].lstrip("/")

    # 2) Abrir informe y extraer campos
    r2 = requests.get(informe_url, headers=HEADERS, timeout=20)
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
        "Profundidad_km": _to_float(prof_m.group(1)),
        "Magnitud": _to_float(mag_m.group(1)),
        "Fuente_informe": informe_url,
    }


def detect_delimiter(sample_text: str) -> str:
    # intenta detectar delimitador; fallback ';'
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=";,\t|")
        return dialect.delimiter
    except Exception:
        return ";"


def read_localidades(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        raise RuntimeError(f"No existe el archivo CSV de localidades en: {csv_path}")

    # leer una muestra para detectar delimitador
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    delim = detect_delimiter(sample)

    # cargar CSV completo
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            raise RuntimeError("El CSV no tiene encabezados (headers).")

        # buscar columnas lat/lon por nombre flexible
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
        # para el nombre de localidad, intenta varias opciones
        name_col = find_col(["localidad", "nombre", "name", "ciudad", "comuna", "poblado", "locality"]) or fields[0]

        if not lat_col or not lon_col:
            raise RuntimeError(
                f"No pude identificar columnas de lat/lon en el CSV. "
                f"Headers detectados: {fields}"
            )

        localidades = []
        for row in reader:
            # ignora filas vacías
            if row is None:
                continue
            try:
                lat = _to_float(row.get(lat_col, ""))
                lon = _to_float(row.get(lon_col, ""))
            except Exception:
                # si una fila viene mala, la saltamos
                continue

            nombre = (row.get(name_col) or "").strip()
            if not nombre:
                nombre = "Sin nombre"

            localidades.append({"localidad": nombre, "lat": lat, "lon": lon})

    if not localidades:
        raise RuntimeError("No se pudieron leer localidades válidas desde el CSV.")
    return localidades


def compute_distancias(evento: dict) -> list[dict]:
    locs = read_localidades(CSV_PATH)
    lat_s = evento["Latitud_sismo"]
    lon_s = evento["Longitud_sismo"]

    out = []
    for loc in locs:
        d_km = haversine_km(lat_s, lon_s, loc["lat"], loc["lon"])
        out.append(
            {
                "localidad": loc["localidad"],
                "latitud": loc["lat"],
                "longitud": loc["lon"],
                "distancia_km": round(d_km, 2),
            }
        )

    out.sort(key=lambda x: x["distancia_km"])
    return out


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
          <li><b>Profundidad (km):</b> {d["Profundidad_km"]}</li>
          <li><b>Magnitud:</b> {d["Magnitud"]}</li>
        </ul>
        <p><b>Fuente:</b> <a href="{d["Fuente_informe"]}" target="_blank">{d["Fuente_informe"]}</a></p>

        <hr/>
        <p>
          Ver distancias a localidades:
          <a href="/distancias">/distancias</a> |
          <a href="/distancias/json">/distancias/json</a>
        </p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/json")
def latest_json():
    return JSONResponse(fetch_latest_event())


@app.get("/distancias", response_class=HTMLResponse)
def distancias_html(n: int = Query(50, ge=1, le=5000)):
    """
    Muestra una tabla con las N localidades más cercanas (ordenadas por distancia).
    Puedes cambiar N con ?n=200, etc.
    """
    evento = fetch_latest_event()
    dist = compute_distancias(evento)

    dist_show = dist[:n]

    rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{x['localidad']}</td><td>{x['distancia_km']}</td><td>{x['latitud']}</td><td>{x['longitud']}</td></tr>"
        for i, x in enumerate(dist_show)
    )

    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>Distancias a localidades</title>
      </head>
      <body style="font-family: Arial, sans-serif; padding: 24px;">
        <h2>Distancias desde el último sismo a localidades</h2>

        <p>
          <b>Sismo:</b> lat {evento["Latitud_sismo"]}, lon {evento["Longitud_sismo"]},
          prof {evento["Profundidad_km"]} km, M {evento["Magnitud"]}
          (<a href="{evento["Fuente_informe"]}" target="_blank">fuente</a>)
        </p>

        <p>
          Mostrando <b>{len(dist_show)}</b> de <b>{len(dist)}</b> localidades.
          (Puedes cambiar con <code>?n=200</code>)
        </p>

        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>#</th>
              <th>Localidad</th>
              <th>Distancia (km)</th>
              <th>Latitud</th>
              <th>Longitud</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>

        <p style="margin-top: 16px;">
          <a href="/">Volver</a> |
          <a href="/distancias/json">Ver JSON</a>
        </p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/distancias/json")
def distancias_json():
    evento = fetch_latest_event()
    dist = compute_distancias(evento)
    return JSONResponse(
        {
            "evento": evento,
            "csv": CSV_PATH,
            "cantidad_localidades": len(dist),
            "distancias": dist,
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
