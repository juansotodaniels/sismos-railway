import os
import re
import time
import math
import json
import joblib
import logging
import pathlib
import hashlib
import warnings
from io import StringIO
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse, quote_plus

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# -----------------------------
# Config por env vars
# -----------------------------
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

MODEL_PATH = os.getenv("MODEL_PATH", "/data/Sismos_17-01-2025.pkl")
MODEL_URL  = os.getenv("MODEL_URL")  # opcional: descarga si no existe
LOCALIDADES_CSV = os.getenv("LOCALIDADES_CSV", "Localidades_CL_osm_places.csv")
COMUNAS_CSV     = os.getenv("COMUNAS_CSV", "Comunas_CL_osm_centroids.csv")

# Fecha límite (override para pruebas). Si viene vacío, usa hoy (America/Santiago).
FECHA_ACTUAL_OVERRIDE = os.getenv("FECHA_ACTUAL", "").strip()  # "2025-07-01 00:00" por ejemplo

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-CL,es;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION_TIMEOUT = 20

ROMAN_MAP = {
    'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 'VI':6,
    'VII':7, 'VIII':8, 'IX':9, 'X':10, 'XI':11, 'XII':12
}

# -----------------------------
# Utilidades
# -----------------------------
def ensure_model(path: str, url: str | None):
    p = pathlib.Path(path)
    if p.exists():
        return str(p)
    if not url:
        raise RuntimeError(f"No existe el modelo en {path} y no se entregó MODEL_URL.")
    p.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Descargando modelo desde {url} ...")
    r = SESSION.get(url, timeout=120)
    r.raise_for_status()
    p.write_bytes(r.content)
    log.info(f"Modelo guardado en {p}")
    return str(p)

def build_engine():
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
        raise RuntimeError("Faltan variables de entorno de BD (DB_HOST, DB_NAME, DB_USER, DB_PASS)")
    conn_str = f"mysql+mysqlconnector://{quote_plus(DB_USER)}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(conn_str, connect_args={"connect_timeout": 120}, pool_pre_ping=True)
    return engine

def to_float_safe(x: str):
    if isinstance(x, (int, float)):
        return float(x)
    if x is None:
        return None
    y = str(x).strip().replace(",", ".")
    try:
        return float(re.findall(r"-?\d+\.?\d*", y)[0])
    except:
        return None

def haversine(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        return round(6371.0 * c, 0)
    except Exception:
        return None

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    # normaliza tildes simples
    rep = {
        "á":"a","é":"e","í":"i","ó":"o","ú":"u",
        "ü":"u","ñ":"n","’":"'", "´":"'", "”":'"', "“":'"'
    }
    for k,v in rep.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s)

# -----------------------------
# Carga diccionarios de localidades
# -----------------------------
def load_localidades(archivo_localidades: str, archivo_comunas: str):
    def _load_csv(path):
        if not pathlib.Path(path).exists():
            log.warning(f"CSV no encontrado: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        # columnas esperadas: nombre, lat, lon (flexibles)
        cols = {c.lower(): c for c in df.columns}
        # intenta mapear nombres comunes
        name_col = next((cols[k] for k in cols if k in ("nombre","name","localidad","place","comuna")), None)
        lat_col  = next((cols[k] for k in cols if k in ("lat","latitude","latitud")), None)
        lon_col  = next((cols[k] for k in cols if k in ("lon","long","lng","longitud","longitude")), None)
        if not name_col or not lat_col or not lon_col:
            # intenta deducir por posición
            if len(df.columns) >= 3:
                name_col, lat_col, lon_col = df.columns[:3]
            else:
                return pd.DataFrame()
        out = df[[name_col, lat_col, lon_col]].copy()
        out.columns = ["nombre", "lat", "lon"]
        out["key"] = out["nombre"].astype(str).map(normalize_text)
        out = out.dropna(subset=["lat","lon"])
        return out

    df1 = _load_csv(archivo_localidades)
    df2 = _load_csv(archivo_comunas)
    base = pd.concat([df1, df2], ignore_index=True)
    base = base.drop_duplicates(subset=["key"])
    return base

LOCALIDADES_BASE = load_localidades(LOCALIDADES_CSV, COMUNAS_CSV)

def lookup_coords(localidad: str):
    if not isinstance(localidad, str) or localidad.strip() == "":
        return None, None
    key = normalize_text(localidad)
    row = LOCALIDADES_BASE.loc[LOCALIDADES_BASE["key"] == key]
    if not row.empty:
        r = row.iloc[0]
        return float(r["lat"]), float(r["lon"])
    # fallback: intenta quitar sufijos/prefijos comunes (ej. "Comuna de", ":" etc.)
    key2 = normalize_text(re.sub(r"(^comuna de\s+|:)", "", localidad))
    row = LOCALIDADES_BASE.loc[LOCALIDADES_BASE["key"] == key2]
    if not row.empty:
        r = row.iloc[0]
        return float(r["lat"]), float(r["lon"])
    return None, None

# -----------------------------
# Fetch helpers
# -----------------------------
def get_html(url: str, allow_redirects=True, timeout=SESSION_TIMEOUT) -> str | None:
    try:
        r = SESSION.get(url, allow_redirects=allow_redirects, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r.text
    except Exception as e:
        log.debug(f"[HTTP] error get {url}: {e}")
        return None

def senapred_to_web(url: str) -> str:
    """senapred.cl → web.senapred.cl (misma ruta); también maneja www."""
    try:
        u = urlparse(url)
        host = u.netloc.replace("www.", "")
        if host == "senapred.cl":
            u = u._replace(netloc="web.senapred.cl")
        return urlunparse(u)
    except:
        return url

def strip_trailing_number_slug(slug: str) -> str:
    # ...-83 → ...  (quita "-<número>" final)
    return re.sub(r"-\d+/?$", "/", slug)

def find_first_informate_conte_from_search(query_terms: str, max_candidates: int = 5) -> list[str]:
    """
    Busca en el buscador de WP: /?s=...
    Devuelve hasta max_candidates enlaces a /informate_conte/.
    """
    q = "+".join([t for t in re.split(r"[\s\-_/]+", query_terms) if t and not t.isdigit()])
    search_url = f"https://web.senapred.cl/?s={q}"
    html = get_html(search_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select('a[href*="/informate_conte/"]'):
        href = a.get("href")
        if not href:
            continue
        if href not in links:
            links.append(href)
        if len(links) >= max_candidates:
            break
    return links

def parse_intensidades_from_html(html: str) -> list[dict]:
    """
    Intenta extraer intensidades desde:
      - Tablas con cabecera 'Mercalli'
      - Listas / párrafos con 'Localidad  ROMANO'
    Retorna: [{'region': str|None, 'localidad': str, 'mercalli_rom': 'III', 'mercalli': 3}, ...]
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    # 1) Tablas con 3 columnas (Región, Localidad, Mercalli)
    for table in soup.find_all("table"):
        # Busca cabeceras
        headers = [normalize_text(th.get_text(" ", strip=True)) for th in table.find_all("th")]
        if not headers:
            # puede venir sin <th>, intentar mirar la primera fila
            first_row = table.find("tr")
            if first_row:
                headers = [normalize_text(td.get_text(" ", strip=True)) for td in first_row.find_all("td")]
        has_mercalli = any("mercalli" in h for h in headers)
        if not has_mercalli:
            continue

        # filas
        rows = table.find_all("tr")
        # si primera fila era header sin <th>, saltarla
        start_idx = 1 if rows and ("mercalli" in " ".join(normalize_text(td.get_text()) for td in rows[0].find_all("td"))) else 0
        for tr in rows[start_idx:]:
            tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(tds) < 2:
                continue
            if len(tds) == 3:
                region, loc, rom = tds
            else:
                # algunas tablas tienen 2 columnas (Localidad, Rom)
                region, loc, rom = None, tds[0], tds[1]
            rom = normalize_text(rom).upper().replace(" ", "")
            rom = rom.replace("IIX", "VIII")  # sanity
            rom = re.sub(r"[^IVXLCDM]", "", rom)
            if rom in ROMAN_MAP and loc:
                results.append({
                    "region": region,
                    "localidad": loc.replace(":", "").strip(),
                    "mercalli_rom": rom,
                    "mercalli": ROMAN_MAP[rom],
                })

    if results:
        return results

    # 2) Párrafos/listas con "Localidad  ROMANO"
    texto = soup.get_text("\n", strip=True)
    # a veces viene la sección: "Las intensidades en escala de Mercalli son:"
    m = re.search(r"intensidades\s+en\s+escala\s+de\s+mercalli\s+son\s*:\s*(.+)", texto, flags=re.I|re.S)
    bloque = texto if not m else m.group(1)
    # extrae pares "Palabras ...  ROMANO"
    for linea in bloque.splitlines():
        linea = linea.strip()
        if not linea:
            continue
        # Muchos vienen tipo "Illapel II", "Talagante III", "Región de XX: ..."
        # Saca región si aparece explícita
        linea_clean = re.sub(r"^regi[oó]n\s+de\s+[^\:]+:\s*", "", linea, flags=re.I)
        m2 = re.findall(r"([A-Za-zÁÉÍÓÚÜÑ'´\-\s\.\(\)]+?)\s+\b(II{0,3}|IV|V|VI{0,2}|IX|X|XI|XII)\b", linea_clean, flags=re.I)
        for (loc, rom) in m2:
            rom = rom.upper()
            loc = loc.replace(":", "").strip()
            if rom in ROMAN_MAP and loc:
                results.append({
                    "region": None,
                    "localidad": loc,
                    "mercalli_rom": rom,
                    "mercalli": ROMAN_MAP[rom],
                })
    return results

def fetch_senapred_intensidades(rep_link: str) -> tuple[list[dict], str | None]:
    """
    Dado el link (posiblemente SPA) de SENAPRED, intenta encontrar una página SSR
    de donde extraer intensidades. Devuelve (lista_intensidades, url_usada_o_none).
    """
    cand_urls = []
    # 1) Cambiar host a web.senapred.cl
    rep_web = senapred_to_web(rep_link)
    cand_urls.append(rep_web)

    # 2) AMP (con y sin slash final)
    if not rep_web.endswith("/"):
        cand_urls.append(rep_web + "/amp/")
    cand_urls.append(rep_web + "?output=amp")

    # 3) Reescribir informate/evento → informate_conte
    try:
        u = urlparse(rep_web)
        path = u.path
        if "/informate/evento/" in path:
            slug = path.split("/informate/evento/", 1)[1]
            if not slug.endswith("/"):
                slug += "/"
            cand_urls.append(urlunparse(u._replace(path="/informate_conte/" + slug)))
            # 3.b sin número final
            stripped = strip_trailing_number_slug(slug)
            cand_urls.append(urlunparse(u._replace(path="/informate_conte/" + stripped)))
    except:
        pass

    tried = set()
    # Intento directo de candidatos
    for url in cand_urls:
        if url in tried:
            continue
        tried.add(url)
        html = get_html(url)
        if not html:
            continue
        if "You need to enable JavaScript to run this app." in html:
            continue
        intens = parse_intensidades_from_html(html)
        if intens:
            return intens, url

    # 4) Búsqueda interna en web.senapred.cl
    # Derivar términos desde el slug original (quita guiones/números)
    slug_terms = re.sub(r"https?://", "", rep_link)
    slug_terms = re.sub(r"[-_/]+", " ", slug_terms)
    slug_terms = re.sub(r"\b\d+\b", " ", slug_terms)
    slug_terms = normalize_text(slug_terms)

    for url in find_first_informate_conte_from_search(slug_terms, max_candidates=6):
        if url in tried:
            continue
        tried.add(url)
        html = get_html(url)
        if not html:
            continue
        intens = parse_intensidades_from_html(html)
        if intens:
            return intens, url

    return [], None

# -----------------------------
# CSN parsers
# -----------------------------
def parse_csn_day_list(day_url: str) -> list[dict]:
    """
    Lee la página del día y devuelve [{id, detalle_url}, ...] de sismos percibidos.
    """
    html = get_html(day_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    out = []
    # Filas de sismos con clase percibido (estructura estable de CSN)
    for div in soup.select(".percibido"):
        a = div.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        # id al final del href (termina en .html, y antes viene el id)
        m = re.search(r"/(\d{5,})\.html$", href)
        if not m:
            # fallback: intenta texto
            m = re.search(r"(\d{5,})", href)
        if not m:
            continue
        sismo_id = int(m.group(1))
        detalle_url = href if href.startswith("http") else f"https://www.sismologia.cl{href if href.startswith('/') else '/' + href}"
        out.append({"id": sismo_id, "url": detalle_url})
    return out

def parse_csn_event(detalle_url: str) -> tuple[dict, str | None]:
    """
    Devuelve (mag_info, rep_link) donde:
    mag_info = {
        'Fecha_Local','Fecha_UTC','Latitud','Longitud','Profundidad','Magnitud','Unidad','Referencia'
    }
    rep_link = url de reporte de intensidades (SENAPRED) o None
    """
    html = get_html(detalle_url)
    if not html:
        return {}, None
    soup = BeautifulSoup(html, "lxml")

    # La tabla de "Hipocentro" suele venir como pares <tr><td>Campo</td><td>Valor</td>
    campos = {}
    for tr in soup.select("article table tr"):
        tds = tr.find_all("td")
        if len(tds) == 2:
            k = tds[0].get_text(" ", strip=True)
            v = tds[1].get_text(" ", strip=True)
            campos[normalize_text(k)] = v

    referencia = campos.get("referencia")
    fecha_local = campos.get("hora local")
    fecha_utc = campos.get("hora utc")
    lat = campos.get("latitud")
    lon = campos.get("longitud")
    prof = campos.get("profundidad")
    magtxt = campos.get("magnitud")

    unidad = None
    magnitud = None
    if magtxt:
        # ej: "5.8 Mw" o "4.3 Ml"
        m = re.match(r"\s*([0-9\.,]+)\s*([A-Za-z]+)\s*", magtxt)
        if m:
            magnitud = to_float_safe(m.group(1))
            unidad = m.group(2)

    # Link a "Reporte de intensidades"
    rep_link = None
    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True).lower()
        if "reporte de intensidades" in txt:
            rep_link = a["href"]
            # asegurar esquema absoluto
            if rep_link and not rep_link.startswith("http"):
                rep_link = f"https://{rep_link.lstrip('/')}"
            break

    # Normaliza fechas a "YYYY-MM-DD HH:MM"
    def norm_fecha(s):
        if not s:
            return None
        s = s.strip()
        try:
            dt = datetime.strptime(s, "%H:%M:%S %d/%m/%Y")
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return s

    mag_info = {
        "Referencia": referencia,
        "Fecha_Local": norm_fecha(fecha_local),
        "Fecha_UTC": norm_fecha(fecha_utc),
        "Latitud": to_float_safe(lat),
        "Longitud": to_float_safe(lon),
        "Profundidad": to_float_safe(prof),
        "Magnitud": magnitud,
        "Unidad": unidad
    }
    return mag_info, rep_link

# -----------------------------
# DB helpers
# -----------------------------
def carga_estado_inicial(engine):
    with engine.connect() as conn:
        r = conn.execute(text("SELECT MAX(Id_sismo) FROM Almacen_Intensidades"))
        id_sismo_0 = r.scalar() or 0
        r = conn.execute(text("SELECT MAX(Fecha_sismo) FROM Almacen_Intensidades"))
        fecha_str = r.scalar() or "2020-01-01 00:00"
        r = conn.execute(text("SELECT COUNT(*) FROM Almacen_Intensidades"))
        count_loc0 = r.scalar() or 0
        r = conn.execute(text("SELECT COUNT(*) FROM Almacen_Magnitudes"))
        count_sis0 = r.scalar() or 0
    return id_sismo_0, str(fecha_str), count_loc0, count_sis0

def insertar_dfs(engine, intens_df: pd.DataFrame, mags_df: pd.DataFrame):
    if not intens_df.empty:
        intens_df = intens_df.drop_duplicates(subset=["Id_sismo", "Localidad"])
        intens_df.to_sql("Almacen_Intensidades", con=engine, if_exists="append", index=False, chunksize=5000, method="multi")
    if not mags_df.empty:
        mags_df = mags_df.drop_duplicates(subset=["Id_sismo"])
        mags_df.to_sql("Almacen_Magnitudes", con=engine, if_exists="append", index=False, chunksize=5000, method="multi")

def dedup_bd(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE t1 FROM Almacen_Intensidades t1
            JOIN Almacen_Intensidades t2
            ON t1.Id_sismo = t2.Id_sismo
            AND t1.Localidad = t2.Localidad
            AND t1.Fecha_sismo = t2.Fecha_sismo
            AND t1.id > t2.id
        """))
        conn.execute(text("""
            DELETE t1 FROM Almacen_Magnitudes t1
            JOIN Almacen_Magnitudes t2
            ON t1.Id_sismo = t2.Id_sismo
            AND t1.Fecha_Local = t2.Fecha_Local
            AND t1.id > t2.id
        """))

# -----------------------------
# Main
# -----------------------------
def run_scraper(desde_fecha: str | None = None, fecha_actual_str: str | None = None):
    # Modelo
    model_path = ensure_model(MODEL_PATH, MODEL_URL)
    try:
        modelo = joblib.load(model_path)
    except Exception as e:
        log.warning(f"No se pudo cargar el modelo ({e}). Se continuará sin predicción.")
        modelo = None

    id_modelo = pathlib.Path(model_path).stem

    # BD
    engine = build_engine()
    id_sismo_0, fecha_str, count_loc0, count_sis0 = carga_estado_inicial(engine)
    log.info(f"Último Id_sismo en BD: {id_sismo_0}")
    log.info(f"Última fecha en BD:   {fecha_str}")

    # Fechas
    if fecha_actual_str and fecha_actual_str.strip():
        fecha_actual_dt = datetime.strptime(fecha_actual_str[:10], "%Y-%m-%d")
    elif FECHA_ACTUAL_OVERRIDE:
        fecha_actual_dt = datetime.strptime(FECHA_ACTUAL_OVERRIDE[:10], "%Y-%m-%d")
    else:
        # “hoy” (sin hora)
        now = datetime.now()
        fecha_actual_dt = datetime(now.year, now.month, now.day)

    fecha_inicio_dt = datetime.strptime(fecha_str[:10], "%Y-%m-%d")
    if desde_fecha:
        try:
            fecha_inicio_dt = datetime.strptime(desde_fecha[:10], "%Y-%m-%d")
        except:
            pass

    log.info(f"Rango de días: {fecha_inicio_dt.date()} \u2192 {fecha_actual_dt.date()}")

    # DataFrames acumuladores
    intensidades_df = pd.DataFrame(columns=[
        "Id_sismo","Fecha_sismo","Localidad","Latitud_localidad","Longitud_localidad",
        "distancia_epicentro","y","ypred","Etapa","Id_modelo"
    ])
    magnitudes_df = pd.DataFrame(columns=[
        "Id_sismo","Url","Fecha_Local","Fecha_UTC","Latitud","Longitud",
        "Profundidad","Magnitud","Unidad","Referencia"
    ])

    num_sis_tot = 0
    num_loc_tot = 0

    # Loop de días
    days = (fecha_actual_dt - fecha_inicio_dt).days + 1
    for i in range(days):
        f = fecha_inicio_dt + timedelta(days=i)
        dia = f.strftime("%d")
        mes = f.strftime("%m")
        anio = f.strftime("%Y")
        day_url = f"https://www.sismologia.cl/sismicidad/catalogo/{anio}/{mes}/{anio}{mes}{dia}.html"

        html = get_html(day_url)
        if not html:
            continue

        # Cuenta "percibidos"
        soup = BeautifulSoup(html, "lxml")
        percibidos = soup.select(".percibido")
        log.info(f"[DÍA] {day_url}")
        log.info(f"[DÍA] sismos percibidos: {len(percibidos)}")

        # parse lista
        sismos = parse_csn_day_list(day_url)
        for it in sismos:
            sid = it["id"]
            if sid and sid <= (id_sismo_0 or 0):
                continue

            mag_info, rep_link = parse_csn_event(it["url"])
            if not mag_info or not mag_info.get("Fecha_Local"):
                # aún así inserta magnitud básica si hay lat/lon/mag
                pass

            # Inserta SIEMPRE el registro de magnitud (aunque no haya intensidades)
            magnitudes_df = pd.concat([magnitudes_df, pd.DataFrame([{
                "Id_sismo": sid,
                "Url": it["url"],
                **{k: mag_info.get(k) for k in ["Fecha_Local","Fecha_UTC","Latitud","Longitud","Profundidad","Magnitud","Unidad","Referencia"]},
            }])], ignore_index=True)
            num_sis_tot += 1

            if rep_link:
                # Resolver intensidades vía SSR
                intens_list, used_url = fetch_senapred_intensidades(rep_link)
                log.info(f"[SISMO {sid}] reporte: {rep_link}")
                if used_url and used_url != rep_link:
                    log.info(f"[SISMO {sid}] SSR usado: {used_url}")
                log.info(f"[SISMO {sid}] intensidades parseadas: {len(intens_list)}")

                # Agregar intensidades
                for item in intens_list:
                    loc = item["localidad"]
                    lat_loc, lon_loc = lookup_coords(loc)
                    dist = None
                    if lat_loc is not None and lon_loc is not None and mag_info.get("Latitud") is not None and mag_info.get("Longitud") is not None:
                        dist = haversine(mag_info["Latitud"], mag_info["Longitud"], lat_loc, lon_loc)

                    ypred = None
                    if modelo is not None and all(v is not None for v in [mag_info.get("Latitud"), mag_info.get("Longitud"), mag_info.get("Profundidad"), mag_info.get("Magnitud"), lat_loc, lon_loc, dist]):
                        feats = [[
                            float(mag_info["Latitud"]),
                            float(mag_info["Longitud"]),
                            float(mag_info["Profundidad"]),
                            float(mag_info["Magnitud"]),
                            float(lat_loc),
                            float(lon_loc),
                            float(dist)
                        ]]
                        try:
                            ypred = int(np.round(modelo.predict(feats))[0])
                        except Exception:
                            ypred = None

                    intensidades_df = pd.concat([intensidades_df, pd.DataFrame([{
                        "Id_sismo": sid,
                        "Fecha_sismo": mag_info.get("Fecha_Local") or mag_info.get("Fecha_UTC"),
                        "Localidad": loc,
                        "Latitud_localidad": lat_loc,
                        "Longitud_localidad": lon_loc,
                        "distancia_epicentro": dist,
                        "y": int(item["mercalli"]),
                        "ypred": (int(ypred) if ypred is not None else None),
                        "Etapa": "Producción",
                        "Id_modelo": id_modelo
                    }])], ignore_index=True)
                    num_loc_tot += 1
            else:
                log.info(f"[SISMO {sid}] sin enlace de intensidades en CSN ({it['url']})")

        # Inserta por día para no acumular de más
        insertar_dfs(engine, intensidades_df, magnitudes_df)
        intensidades_df = intensidades_df.iloc[0:0]
        magnitudes_df = magnitudes_df.iloc[0:0]
        time.sleep(0.2)

    # Dedup y resumen
    dedup_bd(engine)
    with engine.connect() as conn:
        result_loc = conn.execute(text("SELECT COUNT(*) FROM Almacen_Intensidades"))
        count_loc = result_loc.scalar() or 0
        result_sis = conn.execute(text("SELECT COUNT(*) FROM Almacen_Magnitudes"))
        count_sis = result_sis.scalar() or 0

    log.info(f"Total sismos nuevos:       {count_sis - count_sis0} de {num_sis_tot}")
    log.info(f"Total localidades nuevas:  {count_loc - count_loc0} de {num_loc_tot}")

if __name__ == "__main__":
    # Puedes fijar DESDE (inclusive) y FECHA_ACTUAL por env vars si quieres
    desde = os.getenv("DESDE_FECHA")  # p.ej. "2025-07-01 00:00"
    fecha_actual_override = os.getenv("FECHA_ACTUAL")  # p.ej. "2025-09-21 00:00"
    run_scraper(desde_fecha=desde, fecha_actual_str=fecha_actual_override)
