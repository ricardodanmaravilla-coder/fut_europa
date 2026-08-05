import os
import time
import requests
import pandas as pd
import unicodedata

API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

LIGAS_A_DESCARGAR = {
    "Premier League": {"id": 39, "espn_code": "eng.1", "archivo": "data/historico_premier.csv"},
    "La Liga": {"id": 140, "espn_code": "esp.1", "archivo": "data/historico_laliga.csv"},
    "Serie A": {"id": 135, "espn_code": "ita.1", "archivo": "data/historico_seriea.csv"},
    "Bundesliga": {"id": 78, "espn_code": "ger.1", "archivo": "data/historico_bundesliga.csv"},
    "Ligue 1": {"id": 61, "espn_code": "fra.1", "archivo": "data/historico_ligue1.csv"}
}

TEMPORADA_ACTUAL = 2026

def normalizar_nombre(nombre):
    return unicodedata.normalize('NFKD', nombre).encode('ASCII', 'ignore').decode('utf-8').strip()

def obtener_partidos_desde_espn(espn_code):
    """Extrae los partidos directamente desde la cartelera pública de ESPN"""
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard"
    partidos_espn = []
    
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            for event in data.get('events', []):
                fecha = event.get('date', '')[:10]
                competencia = event.get('competitions', [{}])[0]
                
                # Equipos
                competitors = competencia.get('competitors', [])
                local, visita, g_loc, g_vis = "", "", None, None
                
                for comp in competitors:
                    team_name = comp.get('team', {}).get('displayName', '')
                    score = comp.get('score')
                    if comp.get('homeAway') == 'home':
                        local = normalizar_nombre(team_name)
                        try: g_loc = int(score) if score is not None else None
                        except: pass
                    else:
                        visita = normalizar_nombre(team_name)
                        try: g_vis = int(score) if score is not None else None
                        except: pass
                
                # Solo agregamos si el partido ya se jugó (tiene marcador)
                status_type = competencia.get('status', {}).get('type', {}).get('completed', False)
                if status_type and g_loc is not None and g_vis is not None and local and visita:
                    partidos_espn.append({
                        "Fecha": fecha,
                        "Local": local,
                        "Visitante": visita,
                        "Goles_Local": g_loc,
                        "Goles_Visita": g_vis,
                        "Corners_Local": 5.0, # Valores estimados estándar para respaldo de ESPN
                        "Corners_Visita": 4.5,
                        "Tarjetas_Local": 2.0,
                        "Tarjetas_Visita": 2.0,
                        "xG_Local": round(g_loc * 0.9, 2),
                        "xG_Visita": round(g_vis * 0.9, 2),
                        "TirosGol_Local": 4.0,
                        "TirosGol_Visita": 3.5,
                        "Atajadas_Local": 3.0,
                        "Atajadas_Visita": 3.0,
                        "Arbitro": "Desconocido"
                    })
    except Exception as e:
        print(f"⚠️ Error conectando a ESPN: {e}")
        
    return partidos_espn

def actualizar_historicos_liga(nombre_liga, league_id, espn_code, archivo_salida):
    print(f"\n[INICIO] Actualizando datos para: {nombre_liga}...")
    
    df_existente = pd.DataFrame()
    fechas_existentes = set()
    if os.path.exists(archivo_salida):
        try:
            df_existente = pd.read_csv(archivo_salida)
            if not df_existente.empty and 'Fecha' in df_existente.columns:
                fechas_existentes = set(zip(df_existente['Fecha'], df_existente['Local'], df_existente['Visitante']))
        except:
            pass

    nuevos_partidos = []

    # 1. INTENTO PRINCIPAL: API-Sports (Si tiene llamadas disponibles)
    url = f"{BASE_URL}/fixtures"
    querystring = {"league": str(league_id), "season": str(TEMPORADA_ACTUAL)}
    
    try:
        response = requests.get(url, headers=HEADERS, params=querystring, timeout=5)
        if response.status_code == 200:
            data = response.json().get("response", [])
            for p in data:
                fixture = p.get("fixture", {})
                fecha = fixture.get("date", "")[:10]
                teams = p.get("teams", {})
                local = normalizar_nombre(teams.get("home", {}).get("name", ""))
                visita = normalizar_nombre(teams.get("away", {}).get("name", ""))
                goals = p.get("goals", {})
                g_loc = goals.get("home")
                g_vis = goals.get("away")

                if g_loc is None or g_vis is None: continue
                if (fecha, local, visita) in fechas_existentes: continue

                # Recolectar estadísticas detalladas si es posible
                fixture_id = fixture.get("id")
                stats_url = f"{BASE_URL}/fixtures/statistics"
                time.sleep(0.2)
                
                try:
                    stats_res = requests.get(stats_url, headers=HEADERS, params={"fixture": fixture_id}, timeout=3)
                except:
                    stats_res = None

                c_loc, c_v, t_loc, t_v = 5.0, 4.5, 1.8, 2.0
                xg_l, xg_v = round(g_loc * 0.9, 2), round(g_vis * 0.9, 2)
                tiros_l, tiros_v, atajadas_l, atajadas_v = 4.0, 3.5, 3.0, 3.0
                arbitro = fixture.get("referee", "Desconocido")

                if stats_res and stats_res.status_code == 200:
                    stats_data = stats_res.json().get("response", [])
                    for team_stat in stats_data:
                        is_home = (normalizar_nombre(team_stat.get("team", {}).get("name", "")) == local)
                        st_list = team_stat.get("statistics", [])
                        stat_dict = {s.get("type"): s.get("value") for s in st_list if s.get("value") is not None}
                        
                        corners = float(stat_dict.get("Corner Kicks", 5))
                        yellows = float(stat_dict.get("Yellow Cards", 1))
                        reds = float(stat_dict.get("Red Cards", 0))
                        t_puntos = yellows + (reds * 2)
                        s_goal = float(stat_dict.get("Shots on Goal", 4))
                        saves = float(stat_dict.get("Goalkeeper Saves", 3))
                        
                        raw_xg = stat_dict.get("expected_goals")
                        if raw_xg is not None:
                            try: xg_val = float(str(raw_xg).replace(",", "."))
                            except: xg_val = round(s_goal * 0.32, 2)
                        else:
                            xg_val = round(s_goal * 0.32, 2)

                        if is_home:
                            c_loc, t_loc, tiros_l, atajadas_l, xg_l = corners, t_puntos, s_goal, saves, xg_val
                        else:
                            c_v, t_v, tiros_v, atajadas_v, xg_v = corners, t_puntos, s_goal, saves, xg_val

                nuevos_partidos.append({
                    "Fecha": fecha, "Local": local, "Visitante": visita,
                    "Goles_Local": g_loc, "Goles_Visita": g_vis,
                    "Corners_Local": c_loc, "Corners_Visita": c_v,
                    "Tarjetas_Local": t_loc, "Tarjetas_Visita": t_v,
                    "xG_Local": xg_l, "xG_Visita": xg_v,
                    "TirosGol_Local": tiros_l, "TirosGol_Visita": tiros_v,
                    "Atajadas_Local": atajadas_l, "Atajadas_Visita": atajadas_v,
                    "Arbitro": arbitro
                })
    except Exception as e:
        print(f"⚠️ API-Sports no disponible o sin llamadas: {e}")

    # 2. RESPALDO AUTOMÁTICO: Si la API oficial falló o no dio resultados, jalamos de ESPN
    if not nuevos_partidos:
        print(f"🔄 Activando respaldo de emergencia: Consultando cartelera de ESPN para {nombre_liga}...")
        partidos_espn = obtener_partidos_desde_espn(espn_code)
        for p in partidos_espn:
            if (p["Fecha"], p["Local"], p["Visitante"]) not in fechas_existentes:
                nuevos_partidos.append(p)

    # Guardar resultados combinados
    if nuevos_partidos:
        df_nuevos = pd.DataFrame(nuevos_partidos)
        df_final = pd.concat([df_existente, df_nuevos], ignore_index=True)
        if 'Fecha' in df_final.columns:
            df_final = df_final.sort_values(by='Fecha').reset_index(drop=True)
            
        os.makedirs("data", exist_ok=True)
        df_final.to_csv(archivo_salida, index=False)
        print(f"✅ [ÉXITO] Se actualizaron {len(nuevos_partidos)} partidos nuevos. Total en CSV: {len(df_final)}.\n")
    else:
        print(f"ℹ️ [INFO] No hay partidos nuevos por registrar para {nombre_liga}.\n")

if __name__ == "__main__":
    for liga, info in LIGAS_A_DESCARGAR.items():
        actualizar_historicos_liga(liga, info["id"], info["espn_code"], info["archivo"])
