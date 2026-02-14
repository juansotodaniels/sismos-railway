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
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# CONFIGURACIÓN
# ============================================================
MIN_EVENT_MAGNITUDE = float(os.getenv("MIN_EVENT_MAGNITUDE", "4"))
MIN_INTENSITY_TO_SHOW = int(os.getenv("MIN_INTENSITY_TO_SHOW", "3"))
DEFAULT_TABLE_ROWS = int(os.getenv("DEFAULT_TABLE_ROWS", "200"))

PRELOAD_MODEL_ON_STARTUP = os.getenv("PRELOAD_MODEL_ON_STARTUP", "1") == "1"

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "25"))
MODEL_DOWNLOAD_TIMEOUT = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "600"))

ALERTA_TOP_DEFAULT = int(os.getenv("ALERTA_TOP_DEFAULT", "200"))

XOR_API_URL = os.getenv("XOR_API_URL", "https://api.xor.cl/sismo/recent")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RailwayBot/1.0; +https://railway.app)"
}

CSV_PATH = os.getenv("LOCALIDADES_CSV", "Localidades_Enero_2026_con_coords.csv")

MODEL_PATH = os.getenv("MODEL_PATH", "Sismos_RF_joblib_Ene_2026.pkl")
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/juansotodaniels/sismos-railway/releases/download/v1.0/Sismos_RF_joblib_Ene_2026.pkl"
)

LOGO_VERSION = "20260203"
MERCALLI_VERSION = "20260214"

app = FastAPI(title="YATI — Predicción de Intensidad Sísmica")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# UTILIDADES
# ============================================================
def _to_float(s):
    return float(str(s).strip().replace(",", "."))

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def distancia_critica_km(M):
    return max(0.0, 11.5220 * M**2 - 8.1164*M + 137.5910)

# ============================================================
# MODELO
# ============================================================
MODEL = None
MODEL_LOCK = threading.Lock()

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    r = requests.get(MODEL_URL, stream=True, timeout=MODEL_DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk)

def load_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            ensure_model()
            MODEL = joblib.load(MODEL_PATH)
    return MODEL

# ============================================================
# HEADER
# ============================================================
def render_header_html():
    return f"""
    <div style="display:flex; align-items:center; gap:18px; margin-bottom:18px;">
      <img src="/static/logo.png?v={LOGO_VERSION}" style="height:140px;">
      <div>
        <h1 style="margin:0; font-size:48px;">
          Y<span style="color:#f57c00;">A</span>T<span style="color:#f57c00;">I</span>
        </h1>
        <h2 style="margin:6px 0 0 0; font-size:22px;">
          Sistema de predicción de intensidad sísmica (Chile)
        </h2>
      </div>
    </div>
    """

# ============================================================
# HOME
# ============================================================
@app.get("/", response_class=HTMLResponse)
def home():

    try:
        r = requests.get(XOR_API_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        evento = data[0] if isinstance(data, list) else data["events"][0]

        lat = float(evento["latitude"])
        lon = float(evento["longitude"])
        mag = float(evento["magnitude"]["value"])
        depth = float(evento["depth"])
        fecha = evento.get("local_date", "No disponible")
        ref = evento.get("geo_reference", "No disponible")

    except Exception:
        return HTMLResponse("<h2>No se pudo obtener información del sismo</h2>")

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>YATI</title>
    </head>
    <body style="font-family: Arial; padding: 24px;">

        {render_header_html()}

        <div style="
            display:flex;
            justify-content:flex-start;
            align-items:flex-start;
            gap:14px;
            flex-wrap:wrap;
        ">

            <div style="flex:1 1 520px; min-width:360px;">
                <h2>Último sismo ≥ {MIN_EVENT_MAGNITUDE} en 48 hrs</h2>
                <ul>
                    <li><b>Fecha:</b> {fecha}</li>
                    <li><b>Latitud:</b> {lat}</li>
                    <li><b>Longitud:</b> {lon}</li>
                    <li><b>Profundidad:</b> {depth} km</li>
                    <li><b>Magnitud:</b> {mag}</li>
                    <li><b>Referencia:</b> {ref}</li>
                </ul>
                <div>
                    <b>Fuente:</b>
                    <a href="https://www.sismologia.cl/" target="_blank">
                        www.sismologia.cl
                    </a>
                </div>
            </div>

            <div style="flex:0 0 190px;">
                <div style="text-align:center; font-weight:600;">
                    Escala Mercalli (MMI)
                </div>
                <img src="/static/mercalli_mmi.jpg?v={MERCALLI_VERSION}"
                     style="width:100%; border-radius:8px; border:1px solid #ddd;">
            </div>

        </div>

    </body>
    </html>
    """

    return HTMLResponse(html)


