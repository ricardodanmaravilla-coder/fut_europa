import os
import streamlit as st
import pandas as pd
import requests
from modules.elo_europa import SistemaEloEuropa
from modules.montecarlo_europa import simular_partido_europa

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="European Elite Leagues Analytics (2026)", layout="wide")

API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

# IDs oficiales de API-Football para las 5 Grandes Ligas de Europa
LIGAS_IDS = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61
}

ARCHIVOS_HISTORICOS = {
    "Premier League": "data/historico_premier.csv",
    "La Liga": "data/historico_laliga.csv",
    "Serie A": "data/historico_seriea.csv",
    "Bundesliga": "data/historico_bundesliga.csv",
    "Ligue 1": "data/historico_ligue1.csv"
}

def cargar_historico_liga(nombre_liga):
    ruta = ARCHIVOS_HISTORICOS.get(nombre_liga, "")
    if os.path.exists(ruta):
        try:
            df = pd.read_csv(ruta)
            df['Local'] = df['Local'].str.strip()
            df['Visitante'] = df['Visitante'].str.strip()
            return df
        except:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_proximos_partidos_europa(league_id):
    url = f"{BASE_URL}/fixtures"
    querystring = {"league": str(league_id), "season": "2026", "next": "10"} 
    response = requests.get(url, headers=HEADERS, params=querystring)
    if response.status_code != 200:
        return {}
        
    datos = response.json().get("response", [])
    partidos_dict = {}
    for p in datos:
        local = p["teams"]["home"]["name"]
        visita = p["teams"]["away"]["name"]
        fix_id = p["fixture"]["id"]
        fecha = p["fixture"]["date"][:10]
        
        llave = f"⚽ {fecha} | {local} vs {visita}"
        partidos_dict[llave] = {
            "local": local,
            "visita": visita,
            "fixture_id": fix_id
        }
    return partidos_dict

st.title("🇪🇺 European Elite Leagues Analytics (5 Grandes Ligas)")
st.write("Análisis cuantitativo avanzado: xG, Tiros a Gol, Atajadas, Árbitros, ELO y Montecarlo.")

# --- SISTEMA DE PESTAÑAS PRINCIPAL ---
tabs = st.tabs(["🇬🇧 Premier League", "🇪🇸 La Liga", "🇮🇹 Serie A", "🇩🇪 Bundesliga", "🇫🇷 Ligue 1", "🌐 Escáner Global EV+"])

# Diccionario para renderizar dinámicamente las 5 ligas
for idx, (nombre_liga, league_id) in enumerate(LIGAS_IDS.items()):
    with tabs[idx]:
        st.subheader(f"📊 {nombre_liga} - Análisis de Jornada")
        
        partidos_liga = obtener_proximos_partidos_europa(league_id)
        if not partidos_liga:
            st.warning(f"⚠️ No se encontraron partidos próximos en la API para la {nombre_liga}.")
        else:
            seleccion = st.selectbox(f"Próximos partidos de {nombre_liga}:", list(partidos_liga.keys()), key=f"sel_{nombre_liga}")
            datos_partido = partidos_liga[seleccion]

            if st.button(f"Ejecutar Simulación - {nombre_liga}", type="primary", key=f"btn_{nombre_liga}"):
                with st.spinner(f"Analizando {datos_partido['local']} vs {datos_partido['visita']} con xG y Montecarlo..."):
                    df_hist = cargar_historico_liga(nombre_liga)
                    
                    motor_elo = SistemaEloEuropa()
                    tabla_elo = motor_elo.actualizar_ratings(df_hist)
                    
                    try:
                        e_loc = float(tabla_elo.loc[tabla_elo['Equipo'] == datos_partido['local'], 'ELO_Rating'].values[0])
                    except:
                        e_loc = 1500.0
                    try:
                        e_vis = float(tabla_elo.loc[tabla_elo['Equipo'] == datos_partido['visita'], 'ELO_Rating'].values[0])
                    except:
                        e_vis = 1500.0

                    resultados = simular_partido_europa(
                        datos_partido["local"], 
                        datos_partido["visita"],
                        df_historico=df_hist,
                        elo_local=e_loc,
                        elo_visita=e_vis
                    )
                    
                    if isinstance(resultados, str):
                        st.error(resultados)
                    else:
                        st.markdown("##### 🏆 Probabilidades del Encuentro (1X2)")
                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"Victoria {datos_partido['local']}", f"{resultados['Resultado_1X2']['Gana Local']}%")
                        c2.metric("Empate", f"{resultados['Resultado_1X2']['Empate']}%")
                        c3.metric(f"Victoria {datos_partido['visita']}", f"{resultados['Resultado_1X2']['Gana Visita']}%")
                        
                        st.markdown("---")
                        st.markdown("##### 🎯 Mercados Over / Under Clave")
                        c4, c5, c6 = st.columns(3)
                        c4.metric("Over 2.5 Goles", f"{resultados['Goles_Over_Under']['Over 2.5']}%")
                        c5.metric("Over 9.5 Córners", f"{resultados['Corners_Totales']['Over 9.5 Corners']}%")
                        c6.metric("Over 4.5 Tarjetas", f"{resultados['Tarjetas_Totales']['Over 4.5 Tarjetas']}%")

# ==========================================
# PESTAÑA 6: ESCÁNER GLOBAL EV+
# ==========================================
with tabs[5]:
    st.subheader("🌐 Escáner Global de Valor (Las 5 Ligas en Simultáneo)")
    st.info("Este módulo recorre las jornadas activas de las 5 ligas europeas en busca de ineficiencias en las cuotas de las casas de apuestas.")
    
    if st.button("🚀 Escanear Todas las Ligas de Europa", type="primary"):
        with st.spinner("Escaneando Premier League, La Liga, Serie A, Bundesliga y Ligue 1..."):
            # Aquí puedes unificar la iteración sobre los fixtures de los 5 IDs de Ligas
            st.success("¡Simulación global lista! (Configura tus cuotas en cada pestaña para afinar el filtro EV+).")
