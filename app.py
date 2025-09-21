# app.py
# Scraper de sismologia.cl (percibidos) sin Selenium, pensado para Railway.
# - Abre cada página del día
# - Parseo robusto con BeautifulSoup
# - Limpieza numérica (° , km , comas)
# - Inserción con SQLAlchemy (to_sql con Connection)
# - Descarga automática del modelo si falta (MODEL_URL)
# - Lookup de Localidades + fallback a Comunas (CSV)
# - Fechas en TZ America/Santiago (tz-aware); override con FECHA_ACTUAL=YYYY-MM-DD

import os, re, math, time, logging, csv, unicodedata
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sismos")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SismosBot/1.0; +https://example.com/bot)"}
ROMANOS = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'X':10,'XI':11,'XII':12}

# ========= Helpers de entorno / DB =========

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

# ========= Descarga modelo si falta =========

def ensure_model_available() -> str:
    path = os.getenv("MODEL_PATH", "modelos/Sismos_17-01-2025.pkl")
    if os.path.exists(path):
        return path
    url = os.getenv("MODEL_URL")
    if not url:
        raise FileNotFoundError(f"Modelo no encontrado: {path} y falta MODEL_URL")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    log.info(f"Descargando modelo desde {url} ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    log.info(f"Modelo guardado en {path}")
    return path

# ========= CSV Localidades + Comunas =========

LOCALIDADES_CSV = os.getenv("LOCALIDADES_CSV", "Localidades_CL_osm_places.csv")
COMUNAS_CSV     = os.getenv("COMUNAS_CSV",     "Comunas_CL_osm_centroids.csv")
_LOCALIDADES_MAP: Optional[Dict[str, Tuple[float,float]]] = None
_COMUNAS_MAP: Optional[Dict[str, Tuple[float,float]]] = None

def _normalize_key(s: str) -> str:
    s = (s or "").strip().lower().replace(":", "")
    s = re.sub(r"^(comuna|municipalidad|ciudad)\s+de\s+", "", s)
    s_nfkd = unicodedata.normalize("NFKD", s)
    s_ascii = "".join(c for c in s_nfkd if not unicodedata.combining(c))
    return " ".join(s_ascii.split())

def _name_variants(raw: str):
    base = _normalize_key(raw)
    if not base: return set()
    variants = {base}
    variants.add(re.sub(r"\b(de|del|la|las|los|el|y)\b", " ", base).strip())
    variants.add(re.sub(r"\([^)]*\)", "", base).strip())
    variants.add(base.replace(",", ""))
    variants.add(base.replace("-", " "))
    return {" ".join(v.split()) for v in variants if v}

def _build_map_from_csv(csv_path: str, name_field: str, lat_field: str, lon_field: str, alt_norm_field: Optional[str]=None):
    idx = {}
    if not os.path.exists(csv_path):
        log.warning(f"No se encontró CSV: {csv_path}")
        return idx
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nombre = (row.get(name_field) or "").strip()
            try:
                lat = float(row.get(lat_field))
                lon = float(row.get(lon_field))
            except Exception:
                continue
            for k in (_normalize_key(nombre), *( _name_variants(nombre) )):
                if k and k not in idx:
                    idx[k] = (lat, lon)
            if alt_norm_field and row.get(alt_norm_field):
                k2 = _normalize_key(row[alt_norm_field])
                if k2 and k2 not in idx:
                    idx[k2] = (lat, lon)
    return idx

def coord_from_csv(localidad: str) -> Optional[Tuple[float,float]]:
    global _LOCALIDADES_MAP
    if _LOCALIDADES_MAP is None:
        _LOCALIDADES_MAP = _build_map_from_csv(
            LOCALIDADES_CSV, name_field="Localidad", lat_field="Latitud", lon_field="Longitud", alt_norm_field="name_normalized"
        )
    k = _normalize_key(localidad)
    if k in _LOCALIDADES_MAP: return _LOCALIDADES_MAP[k]
    for v in _name_variants(localidad):
        if v in _LOCALIDADES_MAP: return _LOCALIDADES_MAP[v]
    return None

def coord_from_comuna(nombre: str) -> Optional[Tuple[float,float]]:
    global _COMUNAS_MAP
    if _COMUNAS_MAP is None:
        _COMUNAS_MAP = _build_map_from_csv(
            COMUNAS_CSV, name_field="Comuna", lat_field="Latitud", lon_field="Longitud", alt_norm_field="Comuna_normalized"
        )
    k = _normalize_key(nombre)
    if k in _COMUNAS_MAP: return _COMUNAS_MAP[k]
    for v in _name_variants(nombre):
        if v in _COMUNAS_MAP: return _COMUNAS_MAP[v]
    return None

def get_coords(localidad_o_comuna: str) -> Tuple[Optional[Tuple[float,float]], Optional[str]]:
    p = coord_from_csv(localidad_o_comuna)
    if p: return p, "place"
    c = coord_from_comuna(localidad_o_comuna)
    if c: return c, "comuna"
    return None, None

# ========= Utilidades scraping / parseo =========

def backoff_get(url: str, max_tries: int = 5, sleep_base: float = 1.0) -> requests.Response:
    last_exc = None
    for i in range(max_tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r
            log.warning(f"GET {url} -> {r.status_code}")
        except requests.RequestException as e:
            last_exc = e
            log.warning(f"GET {url} error: {e}")
        time.sleep(sleep_base * (2 ** i))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"No se pudo obtener {url}")

def clean_num(s: str) -> float:
    s = (s or "").replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in {"", "-", "."}:
        raise ValueError(f"Número inválido: '{s}'")
    return float(s)

def haversine(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return float(np.round(6371.0 * c, 0))
    except Exception:
        return None

def url_dia(fecha: datetime) -> str:
    y = fecha.year
    m = f"{fecha.month:02d}"
    d = f"{fecha.day:02d}"
    return f"https://www.sismologia.cl/sismicidad/catalogo/{y}/{m}/{y}{m}{d}.html"

def extrae_id_sismo_desde_url(url: str) -> Optional[int]:
    m = re.search(r"/(\d{6})\.html$", url)
    return int(m.group(1)) if m else None

def parse_info_sismo(soup: BeautifulSoup) -> Dict[str,str]:
    data = {}
    table = None
    articles = soup.select("main article")
    if len(articles) >= 2:
        table = articles[1].select_one("table")
    if not table:
        table = soup.select_one("article table")

    def set_k(k, v):
        if v is not None:
            data[k] = v

    if table:
        for tr in table.select("tr"):
            tds = tr.select("td")
            if len(tds) >= 2:
                k = tds[0].get_text(strip=True)
                v = tds[1].get_text(strip=True)
                if "Referencia" in k:
                    set_k("Referencia", v)
                elif "Fecha" in k and "Local" in k:
                    set_k("Fecha_Local_raw", v)
                elif "Fecha" in k and "UTC" in k:
                    set_k("Fecha_UTC_raw", v)
                elif "Latitud" in k:
                    set_k("Latitud", v)
                elif "Longitud" in k:
                    set_k("Longitud", v)
                elif "Profundidad" in k:
                    set_k("Profundidad", v)
                elif "Magnitud" in k:
                    set_k("Magnitud_raw", v)

    # Normaliza fechas
    def to_iso(raw: str) -> Optional[str]:
        if not raw: return None
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

    # Magnitud número + unidad
    if "Magnitud_raw" in data:
        m = re.search(r"([0-9]+(?:[\.,][0-9]+)?)\s*([A-Za-z]+)", data["Magnitud_raw"].replace(",", "."))
        if m:
            data["Magnitud"] = m.group(1)
            data["Unidad"]   = m.group(2)
        else:
            data["Magnitud"] = data["Magnitud_raw"]

    return data

def encuentra_link_reporte_intensidades(soup: BeautifulSoup) -> Optional[str]:
    # heurística por texto
    for a in soup.select("a[href]"):
        href = a["href"]
        txt = a.get_text(strip=True).lower()
        if ("reporte" in txt or "intens" in txt) and href:
            if href.startswith("http"):
                return href
            if href.startswith("/"):
                return "https://www.sismologia.cl" + href
            return href
    return None

def parse_intensidades(soup: BeautifulSoup) -> List[Tuple[str,int]]:
    resultados: List[Tuple[str,int]] = []
    tablas = soup.select("#publishView table")
    if not tablas:
        tablas = soup.select("table")
    for table in tablas:
        for tr in table.select("tr"):
            tds = [td.get_text(strip=True) for td in tr.select("td")]
            if not tds or "Región" in tds[0]:
                continue
            if len(tds) == 2:
                comuna = tds[0].replace(":", "").strip()
                rom = (tds[1] or "").strip().upper()
                val = ROMANOS.get(rom)
                if val:
                    resultados.append((comuna, val))
    return resultados

# ========= Scraper principal =========

def carga_estado_inicial(engine: Engine):
    with engine.connect() as conn:
        id_sismo_0 = conn.execute(text("SELECT MAX(Id_sismo) FROM Almacen_Intensidades")).scalar()
        fecha_str  = conn.execute(text("SELECT MAX(Fecha_sismo) FROM Almacen_Intensidades")).scalar()
        count_loc0 = conn.execute(text("SELECT COUNT(*) FROM Almacen_Intensidades")).scalar() or 0
        count_sis0 = conn.execute(text("SELECT COUNT(*) FROM Almacen_Magnitudes")).scalar() or 0
    return (id_sismo_0 or 0), fecha_str, count_loc0, count_sis0

def run_scraper(desde_fecha: Optional[str] = None, fecha_actual_str: Optional[str] = None):
    engine = build_engine()

    # Modelo
    model_path = ensure_model_available()
    modelo = joblib.load(model_path)
    id_modelo = os.path.basename(model_path).replace(".pkl", "")
    log.info(f"Modelo: {id_modelo}")

    # Estado inicial
    id_sismo_0, fecha_str, count_loc0, count_sis0 = carga_estado_inicial(engine)
    log.info(f"Último Id_sismo en BD: {id_sismo_0}")
    log.info(f"Última fecha en BD:   {fecha_str}")

    # Fecha actual tz-aware (America/Santiago), override con FECHA_ACTUAL si quieres
    tz = ZoneInfo("America/Santiago")
    if fecha_actual_str:
        fa = datetime.strptime(fecha_actual_str[:10], "%Y-%m-%d")
        fecha_actual = datetime(fa.year, fa.month, fa.day, tzinfo=tz)
    else:
        now_cl = datetime.now(tz)
        fecha_actual = now_cl.replace(hour=0, minute=0, second=0, microsecond=0)

    # Fecha inicial
    if desde_fecha:
        fecha_inicial = datetime.strptime(desde_fecha[:10], "%Y-%m-%d").replace(tzinfo=tz)
    elif fecha_str:
        fecha_inicial = datetime.strptime(str(fecha_str)[:10], "%Y-%m-%d").replace(tzinfo=tz)
    else:
        fecha_inicial = (fecha_actual - timedelta(days=2))

    log.info(f"Rango de días: {fecha_inicial:%Y-%m-%d} → {fecha_actual:%Y-%m-%d}")

    num_loc_tot = 0
    num_sis_tot = 0

    # Loop por días
    dias = (fecha_actual.date() - fecha_inicial.date()).days
    for i in range(dias + 1):
        f = (fecha_inicial + timedelta(days=i)).astimezone(tz)
        url = url_dia(f)
        log.info(f"[DÍA] {url}")
        try:
            r = backoff_get(url)
        except Exception as e:
            log.warning(f"Saltando {url}: {e}")
            continue

        soup = BeautifulSoup(r.text, "lxml")
        items = soup.select(".percibido a[href]")
        if not items:
            # fallback por texto
            items = [a for a in soup.select("a[href]") if "percib" in a.get_text(strip=True).lower()]
        log.info(f"[DÍA] sismos percibidos: {len(items)}")
        if not items:
            continue

        # Loop por sismo
        for a in items:
            try:
                href = a.get("href") or ""
                if not href:
                    continue
                link = href if href.startswith("http") else ("https://www.sismologia.cl" + href if href.startswith("/") else href)
                id_sismo = extrae_id_sismo_desde_url(link)
                if not id_sismo:
                    log.warning(f"No pude extraer Id_sismo en {link}")
                    continue

                if id_sismo <= id_sismo_0:
                    continue

                # Detalle del sismo
                sismo_resp = backoff_get(link)
                sismo_soup = BeautifulSoup(sismo_resp.text, "lxml")
                meta = parse_info_sismo(sismo_soup)

                try:
                    lat_sis = clean_num(meta.get("Latitud"))
                    lon_sis = clean_num(meta.get("Longitud"))
                    prof_km = clean_num(meta.get("Profundidad"))
                except Exception as e:
                    log.warning(f"Sismo {id_sismo} con números inválidos: {e}")
                    continue

                mag_raw = meta.get("Magnitud")
                unidad = meta.get("Unidad", "")
                if mag_raw is None:
                    # intenta caer del raw
                    mraw = meta.get("Magnitud_raw")
                    mag_raw = clean_num(mraw) if mraw else None
                magnitud = float(str(mag_raw).replace(",", ".")) if mag_raw is not None else None

                ref = meta.get("Referencia")
                fecha_local = meta.get("Fecha_Local")
                fecha_utc   = meta.get("Fecha_UTC")

                # Link reporte intensidades
                rep_link = encuentra_link_reporte_intensidades(sismo_soup)
                if not rep_link:
                    log.warning(f"No hallé reporte intensidades para {id_sismo} ({link})")
                    continue

                rep_resp = backoff_get(rep_link)
                rep_soup = BeautifulSoup(rep_resp.text, "lxml")
                pares = parse_intensidades(rep_soup)
                log.info(f"[SISMO {id_sismo}] intensidades parseadas: {len(pares)}")

                if not pares:
                    continue

                nuevos_int = []
                for comuna, y in pares:
                    par, origen = get_coords(comuna)
                    if not par:
                        log.warning(f"Sin coordenadas para {comuna} (sismo {id_sismo})")
                        continue
                    lat_loc, lon_loc = par
                    dist = haversine(lat_sis, lon_sis, lat_loc, lon_loc)
                    X = [[lat_sis, lon_sis, prof_km, magnitud or 0.0, lat_loc, lon_loc, dist or 0.0]]
                    ypred = int(np.round(modelo.predict(X))[0])

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
                        "Id_modelo": os.path.basename(model_path).replace(".pkl", "")
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
                        "Profundidad": prof_km,
                        "Magnitud": magnitud,
                        "Unidad": unidad,
                        "Referencia": ref
                    }]).drop_duplicates(subset=["Id_sismo"])

                    try:
                        with engine.begin() as conn:
                            df_mag.to_sql("Almacen_Magnitudes", con=conn, if_exists="append", index=False, chunksize=1000, method="multi")
                            df_int.to_sql("Almacen_Intensidades", con=conn, if_exists="append", index=False, chunksize=1000, method="multi")
                        num_sis_tot += 1
                        num_loc_tot += len(df_int)
                        log.info(f"[SISMO {id_sismo}] INSERTADOS -> mag:1, int:{len(df_int)}")
                    except Exception as e:
                        log.error(f"Fallo al insertar sismo {id_sismo}: {e}")

                time.sleep(0.8)  # respeto al sitio

            except Exception as e:
                log.error(f"Error en sismo {id_sismo if 'id_sismo' in locals() else 'desconocido'}: {e}")

        time.sleep(0.8)

    # Deduplicación
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                DELETE t1 FROM Almacen_Intensidades t1
                JOIN Almacen_Intensidades t2
                ON 
                    t1.Id_sismo = t2.Id_sismo AND
                    t1.Fecha_sismo = t2.Fecha_sismo AND
                    t1.Localidad = t2.Localidad AND
                    t1.Latitud_localidad = t2.Latitud_localidad AND
                    t1.Longitud_localidad = t2.Longitud_localidad AND
                    t1.distancia_epicentro = t2.distancia_epicentro AND
                    t1.y = t2.y AND
                    t1.ypred = t2.ypred AND
                    t1.Etapa = t2.Etapa AND
                    t1.Id_modelo = t2.Id_modelo AND
                    t1.id > t2.id
            """))
            conn.execute(text("""
                DELETE t1 FROM Almacen_Magnitudes t1
                JOIN Almacen_Magnitudes t2
                ON 
                    t1.Id_sismo = t2.Id_sismo AND
                    t1.Fecha_Local = t2.Fecha_Local AND
                    t1.Fecha_UTC = t2.Fecha_UTC AND
                    t1.Latitud = t2.Latitud AND
                    t1.Longitud = t2.Longitud AND
                    t1.Profundidad = t2.Profundidad AND
                    t1.Magnitud = t2.Magnitud AND
                    t1.Unidad = t2.Unidad AND
                    t1.Referencia = t2.Referencia AND
                    t1.id > t2.id
            """))
    except SQLAlchemyError as e:
        log.warning(f"Deduplicación con warning: {e}")

    with engine.connect() as conn:
        count_loc = conn.execute(text("SELECT COUNT(*) FROM Almacen_Intensidades")).scalar() or 0
        count_sis = conn.execute(text("SELECT COUNT(*) FROM Almacen_Magnitudes")).scalar() or 0

    log.info(f"Total sismos nuevos:       {count_sis - count_sis0} de {num_sis_tot}")
    log.info(f"Total localidades nuevas:  {count_loc - count_loc0} de {num_loc_tot}")

# ========= Entrada CLI / Scheduler =========

if __name__ == "__main__":
    # Permite forzar fechas desde variables si deseas:
    # DESDE_FECHA=YYYY-MM-DD  FECHA_ACTUAL=YYYY-MM-DD
    desde = os.getenv("DESDE_FECHA")
    fecha_actual_override = os.getenv("FECHA_ACTUAL")
    run_scraper(desde_fecha=desde, fecha_actual_str=fecha_actual_override)
