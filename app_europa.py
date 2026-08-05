import os
import time
import streamlit as st
import pandas as pd
import requests

from modules.elo_europa import SistemaEloEuropa
from modules.montecarlo_europa import simular_partido_europa
from modules.ml_europa import PredictorMLEuropa
from modules.odds_europa import obtener_cuotas_europa, analizar_apuestas_europa

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
st.write("Análisis cuantitativo avanzado: xG, Tiros a Gol, Atajadas, Árbitros, ELO, ML y Montecarlo.")

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
                with st.spinner(f"Analizando {datos_partido['local']} vs {datos_partido['visita']} con Motores Avanzados..."):
                    df_hist = cargar_historico_liga(nombre_liga)
                    
                    # 1. MOTOR ELO
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

                    # 2. MOTOR MONTECARLO
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
                        st.markdown("### 🎲 Modelo Matemático: Poisson & ELO (Montecarlo)")
                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"Victoria {datos_partido['local']}", f"{resultados['Resultado_1X2']['Gana Local']}%")
                        c2.metric("Empate", f"{resultados['Resultado_1X2']['Empate']}%")
                        c3.metric(f"Victoria {datos_partido['visita']}", f"{resultados['Resultado_1X2']['Gana Visita']}%")
                        
                        st.markdown("---")
                        c4, c5, c6 = st.columns(3)
                        c4.metric("Over 2.5 Goles", f"{resultados['Goles_Over_Under']['Over 2.5']}%")
                        c5.metric("Over 9.5 Córners", f"{resultados['Corners_Totales']['Over 9.5 Corners']}%")
                        c6.metric("Over 4.5 Tarjetas", f"{resultados['Tarjetas_Totales']['Over 4.5 Tarjetas']}%")

                        st.markdown("---")
                        
                        # 3. MOTOR MACHINE LEARNING
                        st.markdown("### 🤖 Modelo Predictivo: Machine Learning (xG y Tiros)")
                        ml_predictor = PredictorMLEuropa()
                        if ml_predictor.entrenar(df_hist):
                            g_l_sim = resultados['Goles_Individuales'][datos_partido['local']]['goles']
                            g_v_sim = resultados['Goles_Individuales'][datos_partido['visita']]['goles']
                            
                            preds_ml = ml_predictor.predecir_mercados_completos(
                                datos_partido['local'], 
                                datos_partido['visita'], 
                                g_l_sim, 
                                g_v_sim,
                                e_loc,  
                                e_vis
                            )
                            
                            if "Resultado_1X2" in preds_ml:
                                ml_c1, ml_c2, ml_c3 = st.columns(3)
                                ml_c1.metric("Local (ML)", f"{preds_ml['Resultado_1X2']['Gana Local']}%")
                                ml_c2.metric("Empate (ML)", f"{preds_ml['Resultado_1X2']['Empate']}%")
                                ml_c3.metric("Visita (ML)", f"{preds_ml['Resultado_1X2']['Gana Visita']}%")
                                
                                ml_c4, ml_c5, ml_c6 = st.columns(3)
                                ml_c4.metric("Over 2.5 Goles (ML)", f"{preds_ml['Goles_Over_Under']['Over 2.5']}%")
                                ml_c5.metric("Over 9.5 Córners (ML)", f"{preds_ml['Corners_Totales']['Over 9.5 Corners']}%")
                                ml_c6.metric("Over 4.5 Tarjetas (ML)", f"{preds_ml['Tarjetas_Totales']['Over 4.5 Tarjetas']}%")
                        else:
                            st.warning("⚠️ El modelo ML requiere más datos históricos para entrenar de forma segura.")

                        st.markdown("---")
                        
                        # 4. GESTIÓN DE CUOTAS Y VALOR ESPERADO (KELLY DEFINITIVO)
                        st.markdown("### ⚙️ Veredicto Definitivo: EV+ con Modelo de Consenso")
                        st.info("💡 Este análisis financiero se calcula cruzando la distribución de goles (Montecarlo/ELO) y el factor humano (xG/Atajadas del ML).")
                        
                        cuotas_automaticas = obtener_cuotas_europa(
                            datos_partido["fixture_id"], 
                            nombre_liga=nombre_liga, 
                            local=datos_partido["local"], 
                            visita=datos_partido["visita"]
                        )
                        
                        mercados_keys = {
                            "Gana Local": "1", "Empate": "X", "Gana Visita": "2", 
                            "Over 2.5 Goles": "Over 2.5", "Under 2.5 Goles": "Under 2.5",
                            "Over 9.5 Corners": "Over 9.5 Corners", "Under 9.5 Corners": "Under 9.5 Corners",
                            "Over 4.5 Tarjetas": "Over 4.5 Tarjetas", "Under 4.5 Tarjetas": "Under 4.5 Tarjetas"
                        }
                        
                        cuotas_usuario = {}
                        cols_cuotas = st.columns(3)
                        
                        for i, (nombre_m, llave) in enumerate(mercados_keys.items()):
                            val_default = cuotas_automaticas.get(llave) if cuotas_automaticas and cuotas_automaticas.get(llave) else 0.0
                            with cols_cuotas[i % 3]:
                                cuotas_usuario[llave] = st.number_input(
                                    f"{nombre_m}", 
                                    min_value=0.0, 
                                    value=float(val_default), 
                                    step=0.05,
                                    format="%.2f",
                                    key=f"cuota_{nombre_liga}_{llave}"
                                )

                        # ATENCIÓN AQUÍ: Le pasamos Montecarlo y ML por separado para que el motor aplique el filtro > 60%
                        df_apuestas = analizar_apuestas_europa(
                            resultados, 
                            preds_ml if ml_predictor.entrenado else {}, 
                            datos_partido["fixture_id"], 
                            cuotas_personalizadas=cuotas_usuario, 
                            nombre_liga=nombre_liga, 
                            local=datos_partido["local"], 
                            visita=datos_partido["visita"]
                        )
                        
                        if not df_apuestas.empty:
                            def color_veredicto(val):
                                if '🔥' in str(val): return 'color: #00ff00; font-weight: bold'
                                elif '✅' in str(val): return 'color: #adff2f'
                                elif '⚠️' in str(val): return 'color: #ffa500'
                                elif '❌' in str(val): return 'color: #ff4d4d'
                                return ''
                                
                            st.dataframe(
                                df_apuestas.style.map(color_veredicto, subset=['Veredicto']), 
                                use_container_width=True,
                                hide_index=True
                            )
# ==========================================
# PESTAÑA 6: ESCÁNER GLOBAL EV+
# ==========================================
with tabs[5]:
    st.subheader("🌐 Escáner Global de Valor (Las 5 Ligas en Simultáneo)")
    st.info("Este módulo recorre las jornadas activas de Europa buscando oportunidades del Sniper (>60% en ambos modelos y EV+).")
    
    if st.button("🚀 Escanear Todas las Ligas de Europa", type="primary"):
        barra_progreso = st.progress(0)
        texto_estado = st.empty()
        
        apuestas_valor = []
        total_ligas = len(LIGAS_IDS)
        
        try:
            for i, (nombre_liga, league_id) in enumerate(LIGAS_IDS.items()):
                texto_estado.markdown(f"**🔍 [Liga {i+1}/{total_ligas}] Analizando {nombre_liga}...**")
                
                # 1. Cargar histórico y ELO
                df_hist = cargar_historico_liga(nombre_liga)
                motor_elo = SistemaEloEuropa()
                tabla_elo = motor_elo.actualizar_ratings(df_hist)
                
                # 2. Obtener partidos
                partidos_liga = obtener_proximos_partidos_europa(league_id)
                if not partidos_liga:
                    continue
                
                for llave_partido, datos_partido in partidos_liga.items():
                    loc = datos_partido['local']
                    vis = datos_partido['visita']
                    fix_id = datos_partido['fixture_id']
                    
                    try: e_loc = float(tabla_elo.loc[tabla_elo['Equipo'] == loc, 'ELO_Rating'].values[0])
                    except: e_loc = 1500.0
                    try: e_vis = float(tabla_elo.loc[tabla_elo['Equipo'] == vis, 'ELO_Rating'].values[0])
                    except: e_vis = 1500.0
                    
                    # 3. Simular Montecarlo
                    resultados = simular_partido_europa(loc, vis, df_hist, e_loc, e_vis)
                    
                    if isinstance(resultados, str):
                        continue
                        
                    # 4. Entrenar y predecir con Machine Learning
                    ml_predictor = PredictorMLEuropa()
                    preds_ml = {}
                    try:
                        if ml_predictor.entrenar(df_hist):
                            g_l_sim = resultados['Goles_Individuales'][loc]['goles']
                            g_v_sim = resultados['Goles_Individuales'][vis]['goles']
                            preds_ml = ml_predictor.predecir_mercados_completos(loc, vis, g_l_sim, g_v_sim, e_loc, e_vis)
                    except Exception as ml_err:
                        print(f"Aviso ML en escáner: {ml_err}")
                        
                    # Pausa imperceptible para proteger el hilo de ejecución
                    time.sleep(0.1) 
                    
                    # 5. Análisis Francotirador (Pasando MC y ML por separado)
                    try:
                        df_apuestas = analizar_apuestas_europa(
                            resultados, 
                            preds_ml, 
                            fix_id, 
                            nombre_liga=nombre_liga, 
                            local=loc, 
                            visita=vis
                        )
                        
                        if not df_apuestas.empty:
                            df_filtrado = df_apuestas[df_apuestas['Veredicto'].str.contains('🔥|✅', na=False)].copy()
                            
                            if not df_filtrado.empty:
                                df_filtrado.insert(0, 'Liga', nombre_liga)
                                df_filtrado.insert(1, 'Partido', f"{loc} vs {vis}")
                                apuestas_valor.append(df_filtrado)
                    except Exception as odds_err:
                        print(f"Aviso Cuotas/Análisis en escáner: {odds_err}")
                        continue
                
                # Actualizar barra de progreso
                progreso_actual = int(((i + 1) / total_ligas) * 100)
                barra_progreso.progress(progreso_actual)
                
            texto_estado.empty()
            barra_progreso.empty()
            
            # Mostrar resultados
            if apuestas_valor:
                df_global = pd.concat(apuestas_valor, ignore_index=True)
                st.success(f"🎯 ¡Escaneo Francotirador finalizado! Se encontraron **{len(df_global)} oportunidades con acuerdo >60% y EV+**.")
                
                def color_veredicto_global(val):
                    if '🔥' in str(val): return 'color: #00ff00; font-weight: bold'
                    elif '✅' in str(val): return 'color: #adff2f'
                    return ''
                    
                st.dataframe(
                    df_global.style.map(color_veredicto_global, subset=['Veredicto']), 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("ℹ️ El escáner terminó de revisar todas las ligas, pero en este momento **ningún partido superó el filtro estricto del 60% de probabilidad en ambos modelos con valor positivo**.")
                
        except Exception as e:
            texto_estado.empty()
            barra_progreso.empty()
            st.error(f"⚠️ Ocurrió una interrupción en el escaneo global: {str(e)}")
