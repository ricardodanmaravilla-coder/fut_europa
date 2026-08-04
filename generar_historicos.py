import os
import time
import requests
import pandas as pd

# Tu API Key (asegúrate de que esté configurada como variable de entorno o reemplázala aquí)
API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

# Ligas y sus IDs oficiales
LIGAS_A_DESCARGAR = {
    "Premier League": {"id": 39, "archivo": "data/historico_premier.csv"},
    "La Liga": {"id": 140, "archivo": "data/historico_laliga.csv"},
    "Serie A": {"id": 135, "archivo": "data/historico_seriea.csv"},
    "Bundesliga": {"id": 78, "archivo": "data/historico_bundesliga.csv"},
    "Ligue 1": {"id": 61, "archivo": "data/historico_ligue1.csv"}
}

# Probamos con la temporada actual (2026) o la anterior (2025)
TEMPORADAS = [2026, 2025]

def descargar_liga_segura(nombre_liga, league_id, archivo_salida):
    print(f"\n[INICIO] Conectando con API-Sports para: {nombre_liga} (ID: {league_id})...")
    partidos_totales = []

    for temporada in TEMPORADAS:
        url = f"{BASE_URL}/fixtures"
        querystring = {"league": str(league_id), "season": str(temporada)}
        
        print(f"-> Consultando temporada {temporada}...")
        response = requests.get(url, headers=HEADERS, params=querystring)
        
        if response.status_code != 200:
            print(f"⚠️ Error HTTP {response.status_code} para {nombre_liga}: {response.text}")
            continue
            
        data = response.json().get("response", [])
        print(f"-> ¡Éxito! Se encontraron {len(data)} partidos en la respuesta de la API.")

        if len(data) == 0:
            continue

        for idx, p in enumerate(data):
            fixture = p.get("fixture", {})
            fecha = fixture.get("date", "")[:10]
            
            teams = p.get("teams", {})
            local = teams.get("home", {}).get("name", "")
            visita = teams.get("away", {}).get("name", "")
            
            goals = p.get("goals", {})
            g_loc = goals.get("home")
            g_vis = goals.get("away")

            # Omitir partidos que aún no se juegan (sin goles registrados)
            if g_loc is None or g_vis is None:
                continue

            fixture_id = fixture.get("id")
            stats_url = f"{BASE_URL}/fixtures/statistics"
            
            # Pausa ligera para evitar saturar el límite de velocidad de la API
            time.sleep(0.25) 
            stats_res = requests.get(stats_url, headers=HEADERS, params={"fixture": fixture_id})
            
            # Valores por defecto estándar de respaldo
            c_loc, c_v = 5.0, 4.5
            t_loc, t_v = 1.8, 2.0
            xg_l, xg_v = 1.4, 1.1
            tiros_l, tiros_v = 4.5, 3.8
            atajadas_l, atajadas_v = 3.0, 3.2
            arbitro = fixture.get("referee", "Desconocido")

            if stats_res.status_code == 200:
                stats_data = stats_res.json().get("response", [])
                for team_stat in stats_data:
                    is_home = (team_stat.get("team", {}).get("name") == local)
                    st_list = team_stat.get("statistics", [])
                    stat_dict = {s.get("type"): s.get("value") for s in st_list}
                    
                    corners = float(stat_dict.get("Corner Kicks", 5) or 5)
                    yellows = float(stat_dict.get("Yellow Cards", 1) or 1)
                    reds = float(stat_dict.get("Red Cards", 0) or 0)
                    tarjetas_puntos = yellows + (reds * 2)
                    
                    shots_on_goal = float(stat_dict.get("Shots on Goal", 4) or 4)
                    saves = float(stat_dict.get("Goalkeeper Saves", 3) or 3)
                    expected_goals = float(stat_dict.get("expected_goals", 1.3) or 1.3)

                    if is_home:
                        c_loc, t_loc, tiros_l, atajadas_l, xg_l = corners, tarjetas_puntos, shots_on_goal, saves, expected_goals
                    else:
                        c_v, t_v, tiros_v, atajadas_v, xg_v = corners, tarjetas_puntos, shots_on_goal, saves, expected_goals

            partidos_totales.append({
                "Fecha": fecha,
                "Local": local,
                "Visitante": visita,
                "Goles_Local": g_loc,
                "Goles_Visita": g_vis,
                "Corners_Local": c_loc,
                "Corners_Visita": c_v,
                "Tarjetas_Local": t_loc,
                "Tarjetas_Visita": t_v,
                "xG_Local": xg_l,
                "xG_Visita": xg_v,
                "TirosGol_Local": tiros_l,
                "TirosGol_Visita": tiros_v,
                "Atajadas_Local": atajadas_l,
                "Atajadas_Visita": atajadas_v,
                "Arbitro": arbitro
            })

        # Si ya encontramos partidos con goles en esta temporada, rompemos el ciclo de temporadas para esta liga
        if len(partidos_totales) > 0:
            break

    if len(partidos_totales) > 0:
        # Asegurar de forma absoluta que la carpeta data exista en el entorno
        os.makedirs("data", exist_ok=True)
        
        df_final = pd.DataFrame(partidos_totales)
        df_final.to_csv(archivo_salida, index=False)
        print(f"✅ [ÉXITO] Archivo guardado correctamente en: {archivo_salida} con {len(df_final)} registros.\n")
    else:
        print(f"⚠️ [AVISO] No se recolectaron partidos jugados para {nombre_liga}.\n")

if __name__ == "__main__":
    # Probamos primero con la Premier League para verificar la respuesta
    for liga, info in list(LIGAS_A_DESCARGAR.items())[:1]: 
        descargar_liga_segura(liga, info["id"], info["archivo"])
