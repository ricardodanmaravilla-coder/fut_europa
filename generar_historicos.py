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

# Definimos el rango exacto de temporadas que solicitaste
TEMPORADAS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

def normalizar_nombre(nombre):
    return unicodedata.normalize('NFKD', nombre).encode('ASCII', 'ignore').decode('utf-8').strip()

def descargar_temporada_liga(nombre_liga, league_id, temporada):
    url = f"{BASE_URL}/fixtures"
    querystring = {"league": str(league_id), "season": str(temporada)}
    partidos_temporada = []
    
    try:
        response = requests.get(url, headers=HEADERS, params=querystring, timeout=15)
        if response.status_code == 200:
            data = response.json().get("response", [])
            print(f"   ↳ Temporada {temporada}: {len(data)} partidos encontrados en la API.")
            
            for p in data:
                fixture = p.get("fixture", {})
                fecha = fixture.get("date", "")[:10]
                
                # Validar que el partido haya finalizado
                short_status = fixture.get("status", {}).get("short", "")
                if short_status not in ["FT", "AET", "PEN"]:
                    continue 

                teams = p.get("teams", {})
                local = normalizar_nombre(teams.get("home", {}).get("name", ""))
                visita = normalizar_nombre(teams.get("away", {}).get("name", ""))
                goals = p.get("goals", {})
                g_loc = goals.get("home")
                g_vis = goals.get("away")

                if g_loc is None or g_vis is None: continue

                fixture_id = fixture.get("id")
                stats_url = f"{BASE_URL}/fixtures/statistics"
                time.sleep(0.12) # Pausa ligera para respetar el límite de la API
                
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

                partidos_temporada.append({
                    "Fecha": fecha, "Local": local, "Visitante": visita,
                    "Goles_Local": g_loc, "Goles_Visita": g_vis,
                    "Corners_Local": c_loc, "Corners_Visita": c_v,
                    "Tarjetas_Local": t_loc, "Tarjetas_Visita": t_v,
                    "xG_Local": xg_l, "xG_Visita": xg_v,
                    "TirosGol_Local": tiros_l, "TirosGol_Visita": tiros_v,
                    "Atajadas_Local": atajadas_l, "Atajadas_Visita": atajadas_v,
                    "Arbitro": arbitro
                })
        else:
            print(f"   ⚠️ Temporada {temporada} no disponible o error HTTP {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Error de conexión en temporada {temporada}: {e}")

    return partidos_temporada

def procesar_liga_completa(nombre_liga, league_id, archivo_salida):
    print(f"\n========================================")
    print(f"🏆 Descargando histórico completo para: {nombre_liga}")
    print(f"========================================")
    
    todos_los_partidos = []

    for temporada in TEMPORADAS:
        print(f"⏳ Consultando temporada {temporada}...")
        partidos_temp = descargar_temporada_liga(nombre_liga, league_id, temporada)
        todos_los_partidos.extend(partidos_temp)
        time.sleep(1) # Pausa de cortesía entre peticiones de temporadas

    if todos_los_partidos:
        df_final = pd.DataFrame(todos_los_partidos)
        
        # Eliminar posibles duplicados exactos por seguridad
        if 'Fecha' in df_final.columns and 'Local' in df_final.columns and 'Visitante' in df_final.columns:
            df_final = df_final.drop_duplicates(subset=['Fecha', 'Local', 'Visitante'])
            df_final = df_final.sort_values(by='Fecha').reset_index(drop=True)
            
        os.makedirs("data", exist_ok=True)
        df_final.to_csv(archivo_salida, index=False)
        print(f"✅ [ÉXITO] Archivo generado: {archivo_salida} con un total de {len(df_final)} partidos históricos (2020-2026).\n")
    else:
        print(f"❌ [AVISO] No se pudieron recolectar datos para {nombre_liga}.\n")

if __name__ == "__main__":
    if not API_KEY:
        print("❌ ATENCIÓN: No se detectó la variable de entorno 'API_SPORTS_KEY'. Configúrala antes de ejecutar.")
    else:
        for liga, info in LIGAS_A_DESCARGAR.items():
            procesar_liga_completa(liga, info["id"], info["archivo"])
