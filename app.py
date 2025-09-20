import os, re, math, time, logging, unicodedata, csv
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

import joblib

# ----------------------------
# Configuración y utilidades
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sismos")

ROMANOS = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'X':10,'XI':11,'XII':12}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SismosBot/1.0; +https://example.com/bot)"}

def env(name: str, default: Optional[str]=None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Falta variable de entorno: {name}")
    return v

def build_engine() -> Engine:
    usuario = env("DB_USER")
    contraseña = env("DB_PASS")
    host = env("DB_HOST")
    puerto = env("DB_PORT")
    base_de_datos = env("DB_NAME")
    url = f"mysql+mysqlconnector://{usuario}:{contraseña}@{host}:{puerto}/{base_de_datos}"
    return create_engine(url, connect_args={"connect_timeout": 120}, pool_pre_ping=True)

def backoff_get(url: str, max_tries: int = 5, sleep_base: float = 1.0) -> requests.Response:
    last_exc = None
    for i in range(max_tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r
            else:
                log.warning(f"GET {url} -> {r.status_code}")
        except requests.RequestException as e:
            last_exc = e
            log.warning(f"GET {url} error: {e}")
        time.sleep(sleep_base * (2 ** i))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"No se pudo obtener {url}")

def haversine(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        radius = 6371.0
        return float(np.round(radius * c, 0))
    except Exception:
        return None

# ----------------------------
# LOOKUP CSV (reemplaza geocoding)
# ----------------------------
LOCALIDADES_CSV = os.getenv("LOCALIDADES_CSV", "Localidades_CL_osm_places.csv")
_LOCALIDADES_MAP = None

def _normalize_key(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower().replace(":", "")
    s = re.sub(r"^(comuna\s+de|municipalidad\s+de|ciudad\s+de)\s+", "", s)
    s_nfkd = unicodedata.normalize("NFKD", s)
    s_ascii = "".join(c for c in s_nfkd if not unicodedata.combining(c))
    return " ".join(s_ascii.split())

def _name_variants(raw: str):
    base = _normalize_key(raw)
    variants = {base}
    variants.add(re.sub(r"\b(de|del|la|las|los|el|y)\b", " ", base).strip())
    variants.add(re.sub(r"\([^)]*\)", "", base).strip())
    variants.add(base.replace(",", ""))
    variants.add(base.replace("-", " "))
    return {" ".join(v.split()) for v in variants if v}

def _build_localidades_map(csv_path: str):
    idx = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                loc = row.get("Localidad", "").strip()
                lat = float(row.get("Latitud"))
                lon = float(row.get("Longitud"))
            except Exception:
                continue

            k = _normalize_key(loc)
            if k and k not in idx:
                idx[k] = (lat, lon)

            nn = row.get("name_normalized")
            if nn:
                nnk = _normalize_key(nn)
                if nnk and nnk not in idx:
                    idx[nnk] = (lat, lon)

            for v in _name_variants(loc):
                if v and v not in idx:
                    idx[v] = (lat, lon)
    return idx

def coord_from_csv(localidad: str):
    global _LOCALIDADES_MAP
    if _LOCALIDADES_MAP is None:
        if not os.path.exists(LOCALIDADES_CSV):
            raise FileNotFoundError(f"No se encontró el CSV de localidades: {LOCALIDADES_CSV}")
        _LOCALIDADES_MAP = _build_localidades_map(LOCALIDADES_CSV)

    key = _normalize_key(localidad)
    if key in _LOCALIDADES_MAP:
        return _LOCALIDADES_MAP[key]
    for v in _name_variants(localidad):
        if v in _LOCALIDADES_MAP:
            return _LOCALIDADES_MAP[v]
    key2 = key.replace(":", "")
    return _LOCALIDADES_MAP.get(key2)

# ==== FALLBACK: centroides de comunas ====
COMUNAS_CSV = os.getenv("COMUNAS_CSV", "Comunas_CL_osm_centroids.csv")
_COMUNAS_MAP = None

def _build_comunas_map(csv_path: str):
    if not os.path.exists(csv_path):
        return {}
    idx = {}
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                nombre = row.get("Comuna") or row.get("Localidad") or ""
                lat = float(row.get("Latitud"))
                lon = float(row.get("Longitud"))
            except Exception:
                continue
            k = _normalize_key(nombre)
            if k and k not in idx:
                idx[k] = (lat, lon)
            nn = row.get("Comuna_normalized") or row.get("name_normalized")
            if nn:
                nnk = _normalize_key(nn)
                if nnk and nnk not in idx:
                    idx[nnk] = (lat, lon)
            for v in _name_variants(nombre):
                if v and v not in idx:
                    idx[v] = (lat, lon)
    return idx

def coord_from_comuna(nombre: str):
    global _COMUNAS_MAP
    if _COMUNAS_MAP is None:
        _COMUNAS_MAP = _build_comunas_map(COMUNAS_CSV)
    if not nombre:
        return None
    key = _normalize_key(nombre)
    if key in _COMUNAS_MAP:
        return _COMUNAS_MAP[key]
    for v in _name_variants(nombre):
        if v in _COMUNAS_MAP:
            return _COMUNAS_MAP[v]
    return None

def get_coords(localidad_o_comuna: str):
    """Intenta primero en Localidades (places). Si no está, cae a Comunas (centroides)."""
    p = coord_from_csv(localidad_o_comuna)
    if p:
        return p, "place"
    c = coord_from_comuna(localidad_o_comuna)
    if c:
        return c, "comuna"
    return None, None
# ==== FIN FALLBACK ====

# ----------------------------
# Parsing HTML sismologia.cl
# ----------------------------
def url_dia(fecha: datetime) -> str:
    y = fecha.year
    m = f"{fecha.month:02d}"
    d = f"{fecha.day:02d}"
    return f"https://www.sismologia.cl/sismicidad/catalogo/{y}/{m}/{y}{m}{d}.html"

def extrae_id_sismo_desde_url(url: str) -> Optional[int]:
    m = re.search(r"(\\d{6})\\.html$", url)
    return int(m.group(1)) if m else None

def parse_info_sismo(soup: BeautifulSoup) -> Dict[str,str]:
    data = {}
    table = None
    articles = soup.select("main article")
    if len(articles) >= 2:
        table = articles[1].select_one("table")
    if not table:
        table = soup.select_one("article table")

    def limpia(t): 
        return re.sub(r"\\s+", " ", t.strip())

    if table:
        for tr in table.select("tr"):
            tds = tr.select("td")
            if len(tds) >= 2:
                k = limpia(tds[0].get_text())
                v = limpia(tds[1].get_text())
                if "Referencia" in k:
                    data["Referencia"] = v
                elif "Fecha y Hora Local" in k or "Fecha Local" in k:
                    data["Fecha_Local_raw"] = v
                elif "Fecha y Hora UTC" in k or "Fecha UTC" in k:
                    data["Fecha_UTC_raw"] = v
                elif "Latitud" in k:
                    data["Latitud"] = re.sub(",", ".", v)
                elif "Longitud" in k:
                    data["Longitud"] = re.sub(",", ".", v)
                elif "Profundidad" in k:
                    mnum = re.search(r"(-?\\d+(\\.\\d+)?)", v)
                    data["Profundidad"] = mnum.group(1) if mnum else v
                elif "Magnitud" in k:
                    mnum = re.search(r"([0-9]+(?:\\.[0-9]+)?)\\s*([A-Za-z]+)", v)
                    if mnum:
                        data["Magnitud"] = mnum.group(1)
                        data["Unidad"] = mnum.group(2)

    def to_iso(raw: str) -> Optional[str]:
        raw = raw.strip()
        for fmt in ("%H:%M:%S %d/%m/%Y", "%H:%M %d/%m/%Y"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                continue
        return None

    if "Fecha_Local_raw" in data:
        data["Fecha_Local"] = to_iso(data["Fecha_Local_raw"])
    if "Fecha_UTC_raw" in data:
        data["Fecha_UTC"] = to_iso(data["Fecha_UTC_raw"])
    return data

def encuentra_link_reporte_intensidades(soup: BeautifulSoup) -> Optional[str]:
    a = soup.select_one("a[href*='reporte'], a:contains('Reporte de intensidades')")
    if a and a.has_attr("href"):
        href = a["href"]
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return "https://www.sismologia.cl" + href
    for a in soup.select("article a"):
        txt = a.get_text().lower()
        if "intens" in txt and a.has_attr("href"):
            href = a["href"]
            if href.startswith("http"):
                return href
            if href.startswith("/"):
                return "https://www.sismologia.cl" + href
    return None

def parse_intensidades(soup: BeautifulSoup) -> List[Tuple[str, int]]:
    resultados = []
    tablas = soup.select("#publishView table") or soup.select("div#publishView table")
    if not tablas:
        tablas = soup.select("table")
    for table in tablas:
        for tr in table.select("tr"):
            tds = [td.get_text(strip=True) for td in tr.select("td")]
            if len(tds) == 2 and "Región" not in tds[0] and tds[0]:
                comuna = tds[0].replace(":", "").strip()
                rom = tds[1].strip().upper()
                val = ROMANOS.get(rom)
                if val:
                    resultados.append((comuna, val))
    return resultados

# ----------------------------
# Núcleo de procesamiento
# ----------------------------
def carga_estado_inicial(engine: Engine) -> Tuple[Optional[int], Optional[str], int, int]:
    with engine.connect() as conn:
        id_sismo_0 = conn.execute(text("SELECT MAX(Id_sismo) FROM Almacen_Intensidades")).scalar()
        fecha_str  = conn.execute(text("SELECT MAX(Fecha_sismo) FROM Almacen_Intensidades")).scalar()
        count_loc0 = conn.execute(text("SELECT COUNT(*) FROM Almacen_Intensidades")).scalar() or 0
        count_sis0 = conn.execute(text("SELECT COUNT(*) FROM Almacen_Magnitudes")).scalar() or 0
    return id_sismo_0, fecha_str, count_loc0, count_sis0

def inserta_dedup(engine: Engine):
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE t1 FROM Almacen_Intensidades t1
            JOIN Almacen_Intensidades t2
              ON t1.Id_sismo = t2.Id_sismo
             AND t1.Fecha_sismo = t2.Fecha_sismo
             AND t1.Localidad = t2.Localidad
             AND t1.Latitud_localidad = t2.Latitud_localidad
             AND t1.Longitud_localidad = t2.Longitud_localidad
             AND t1.distancia_epicentro = t2.distancia_epicentro
             AND t1.y = t2.y
             AND t1.ypred = t2.ypred
             AND t1.Etapa = t2.Etapa
             AND t1.Id_modelo = t2.Id_modelo
             AND t1.id > t2.id
        """))
        conn.execute(text("""
            DELETE t1 FROM Almacen_Magnitudes t1
            JOIN Almacen_Magnitudes t2
              ON t1.Id_sismo = t2.Id_sismo
             AND t1.Fecha_Local = t2.Fecha_Local
             AND t1.Fecha_UTC = t2.Fecha_UTC
             AND t1.Latitud = t2.Latitud
             AND t1.Longitud = t2.Longitud
             AND t1.Profundidad = t2.Profundidad
             AND t1.Magnitud = t2.Magnitud
             AND t1.Unidad = t2.Unidad
             AND t1.Referencia = t2.Referencia
             AND t1.id > t2.id
        """))

def run_scraper(desde_fecha: Optional[str] = None):
    engine = build_engine()

    model_path = env("MODEL_PATH", "modelos/Sismos_17-01-2025.pkl")
    nuevo_modelo = joblib.load(model_path)
    id_modelo = os.path.basename(model_path).replace(".pkl", "")

    id_sismo_0, fecha_str, count_loc0, count_sis0 = carga_estado_inicial(engine)
    now = datetime.now()
    log.info(f"Fecha/hora ejecución: {now}")
    log.info(f"Último Id_sismo en BD: {id_sismo_0}")
    log.info(f"Última fecha en BD: {fecha_str}")

    if desde_fecha:
        fecha_inicial = datetime.strptime(desde_fecha[:10], "%Y-%m-%d")
    else:
        if fecha_str:
            fecha_inicial = datetime.strptime(str(fecha_str)[:10], "%Y-%m-%d")
        else:
            fecha_inicial = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)

    num_loc_tot = 0
    num_sis_tot = 0

    dias = (now.date() - fecha_inicial.date()).days
    for i in range(dias + 1):
        f = fecha_inicial + timedelta(days=i)
        url = url_dia(f)
        log.info(f"Procesando día: {url}")

        try:
            r = backoff_get(url)
        except Exception as e:
            log.warning(f"Saltando {url}: {e}")
            continue

        soup = BeautifulSoup(r.text, "lxml")

        percibidos = soup.select(".percibido a[href]")
        if not percibidos:
            percibidos = [a for a in soup.select("a[href]") if "percib" in a.get_text(strip=True).lower()]
        if not percibidos:
            log.info("Sin sismos percibidos en este día.")
            continue

        for a in percibidos:
            try:
                link = a["href"]
                if not link.startswith("http"):
                    link = "https://www.sismologia.cl" + link if link.startswith("/") else link
                id_sismo = extrae_id_sismo_desde_url(link)
                if id_sismo_0 and id_sismo and id_sismo <= id_sismo_0:
                    continue

                sismo_resp = backoff_get(link)
                sismo_soup = BeautifulSoup(sismo_resp.text, "lxml")
                meta = parse_info_sismo(sismo_soup)

                lat_sis = meta.get("Latitud")
                lon_sis = meta.get("Longitud")
                prof = meta.get("Profundidad")
                mag = meta.get("Magnitud")
                unidad = meta.get("Unidad")
                ref = meta.get("Referencia")
                fecha_local = meta.get("Fecha_Local")
                fecha_utc = meta.get("Fecha_UTC")

                rep_link = encuentra_link_reporte_intensidades(sismo_soup)
                if not rep_link:
                    log.warning(f"No se encontró link de intensidades para sismo {id_sismo} ({link})")
                    continue

                rep_resp = backoff_get(rep_link)
                rep_soup = BeautifulSoup(rep_resp.text, "lxml")
                pares = parse_intensidades(rep_soup)  # [(comuna, y)]

                nuevos_int = []
                for comuna, y in pares:
                    par, origen = get_coords(comuna)
                    if not par:
                        log.warning(f"[CSV] Localidad/Comuna no encontrada: {comuna}")
                        continue
                    lat_loc, lon_loc = par
                    if origen == "comuna":
                        log.info(f"[FALLBACK] Usando centroide de comuna para: {comuna}")
                    dist = haversine(lat_sis, lon_sis, lat_loc, lon_loc)
                    X = [[float(lat_sis), float(lon_sis), float(prof), float(mag), float(lat_loc), float(lon_loc), float(dist or 0.0)]]
                    ypred = int(np.round(nuevo_modelo.predict(X))[0])
                    nuevos_int.append({
                        "Id_sismo": id_sismo,
                        "Fecha_sismo": fecha_local,
                        "Localidad": comuna,
                        "Latitud_localidad": lat_loc,
                        "Longitud_localidad": lon_loc,
                        "distancia_epicentro": dist,
                        "y": int(y),
                        "ypred": ypred,
                        "Etapa": "Producción",
                        "Id_modelo": id_modelo
                    })

                if nuevos_int:
                    df_int = pd.DataFrame(nuevos_int).drop_duplicates(subset=["Id_sismo", "Localidad"])
                    df_mag = pd.DataFrame([{
                        "Id_sismo": id_sismo,
                        "Url": link,
                        "Fecha_Local": fecha_local,
                        "Fecha_UTC": fecha_utc,
                        "Latitud": lat_sis,
                        "Longitud": lon_sis,
                        "Profundidad": prof,
                        "Magnitud": mag,
                        "Unidad": unidad,
                        "Referencia": ref
                    }]).drop_duplicates(subset=["Id_sismo"])

                    with build_engine().begin() as conn:
                        df_mag.to_sql("Almacen_Magnitudes", con=conn.connection, if_exists="append", index=False, chunksize=1000, method="multi")
                        df_int.to_sql("Almacen_Intensidades", con=conn.connection, if_exists="append", index=False, chunksize=1000, method="multi")

                    num_loc_tot += len(df_int)
                    num_sis_tot += 1

                time.sleep(1.0)  # Respeto mínimo al sitio

            except Exception as e:
                log.error(f"Error en sismo {id_sismo if 'id_sismo' in locals() else 'desconocido'}: {e}")

    try:
        inserta_dedup(engine)
    except SQLAlchemyError as e:
        log.warning(f"Deduplicación con warning: {e}")

    with engine.connect() as conn:
        count_loc = conn.execute(text("SELECT COUNT(*) FROM Almacen_Intensidades")).scalar() or 0
        count_sis = conn.execute(text("SELECT COUNT(*) FROM Almacen_Magnitudes")).scalar() or 0

    log.info(f"Total sismos nuevos: {count_sis - count_sis0} de {num_sis_tot}")
    log.info(f"Total localidades nuevas: {count_loc - count_loc0} de {num_loc_tot}")

# ----------------------------
# FastAPI opcional (endpoint)
# ----------------------------
if os.getenv("ENABLE_API", "0") == "1":
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse

    app = FastAPI()

    @app.get("/", response_class=PlainTextResponse)
    def root():
        return "OK"

    @app.post("/actualizar_sismos", response_class=PlainTextResponse)
    def actualizar_sismos(desde: Optional[str] = None):
        run_scraper(desde_fecha=desde)
        return "Actualización completa"

# ----------------------------
# Modo script
# ----------------------------
if __name__ == "__main__":
    desde = os.getenv("DESDE_FECHA")
    run_scraper(desde_fecha=desde)
