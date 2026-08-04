import os
import requests
import pandas as pd

# Configuración de tu API Key de API-Sports
API_KEY = os.environ.get("API_SPORTS_KEY") # O puedes poner tu llave directamente en texto: "TU_API_KEY"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

# Diccionario de Ligas con sus IDs oficiales y el nombre del archivo CSV que generará
LIGAS_A_DESCARGAR = {
    "Premier League": {"id": 39, "archivo": "data/historico_premier.csv"},
    "La Liga": {"id": 140, "archivo": "data/historico_laliga.csv"},
    "Serie A": {"id": 135, "archivo": "data/historico_seriea.csv"},
    "Bundesliga": {"id": 78, "archivo": "data/historico_bundesliga.csv"},
    "Ligue 1": {"id": 61, "archivo": "data/historico_ligue1.csv"}
}

# Temporadas que deseas extraer
TEMPORADAS = [2020, 2021]

def descargar_y_formatear_liga(nombre_liga, league_id, archivo_salida):
    print(f"📥 Descargando histórico para {nombre_liga}...")
    partidos_totales = []

    for temporada in TEMPORADAS:
        url = f"{BASE_URL}/fixtures"
        querystring = {"league": str(league_id), "season": str(temporada)}
        
        response = requests.get(url, headers=HEADERS, params=querystring)
        if response.status_code != 200:
            print(f"⚠️ Error al conectar con la API para {nombre_liga} en la temporada {temporada}")
            continue
            
        data = response.json().get("response", [])
        print(f"-> Temporada {temporada}: {len(data)} partidos encontrados.")

        for p in data:
            # Extraer datos básicos del partido
            fixture = p.get("fixture", {})
            fecha = fixture.get("date", "")[:10]
            
            teams = p.get("teams", {})
            local = teams.get("home", {}).get("name", "")
            visita = teams.get("away", {}).get("name", "")
            
            goals = p.get("goals", {})
            g_loc = goals.get("home", 0)
            g_vis = goals.get("away", 0)

            # Si el partido no se ha jugado (no tiene goles), lo saltamos
            if g_loc is None or g_vis is None:
                continue

            # Estadísticas detalladas del partido (Statistics endpoint o metadatos)
            # Para simplificar y asegurar compatibilidad masiva, asignamos valores base o extraemos si la API los trae
            # (Nota: La API-Sports de fútbol desglosa tiros, córners y tarjetas en el endpoint /fixtures/statistics)
            fixture_id = fixture.get("id")
            stats_url = f"{BASE_URL}/fixtures/statistics"
            stats_res = requests.get(stats_url, headers=HEADERS, params={"fixture": fixture_id})
            
            # Valores por defecto estándar de fútbol europeo por si falla una consulta individual
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
                    
                    # Mapear las estadísticas oficiales de API-Sports
                    stat_dict = {s.get("type"): s.get("value") for s in st_list}
                    
                    corners = float(stat_dict.get("Corner Kicks", 5) or 5)
                    yellows = float(stat_dict.get("Yellow Cards", 1) or 1)
                    reds = float(stat_dict.get("Red Cards", 0) or 0)
                    tarjetas_puntos = yellows + (reds * 2) # Ponderación de tarjetas
                    
                    shots_on_goal = float(stat_dict.get("Shots on Goal", 4) or 4)
                    saves = float(stat_dict.get("Goalkeeper Saves", 3) or 3)
                    expected_goals = float(stat_dict.get("expected_goals", 1.3) or 1.3) # Si la cuenta tiene xG disponible

                    if is_home:
                        c_loc = corners
                        t_loc = tarjetas_puntos
                        tiros_l = shots_on_goal
                        atajadas_l = saves
                        xg_l = expected_goals
                    else:
                        c_v = corners
                        t_v = tarjetas_puntos
                        tiros_v = shots_on_goal
                        atajadas_v = saves
                        xg_v = expected_goals

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

    if partidos_totales:
        os.makedirs("data", exist_ok=True)
        df_final = pd.DataFrame(partidos_totales)
        df_final.to_csv(archivo_salida, index=False)
        print(f"✅ ¡Archivo creado con éxito: {archivo_salida} ({len(df_final)} partidos guardados)!\n")
    else:
        print(f"⚠️ No se pudieron recolectar datos para {nombre_liga}.\n")

if __name__ == "__main__":
    for liga, info in LIGAS_A_DESCARGAR.items():
        descargar_y_formatear_liga(liga, info["id"], info["archivo"])
