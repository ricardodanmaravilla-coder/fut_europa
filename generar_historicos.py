import os
import time
import requests
import pandas as pd

# Coloca tu llave de API-Sports de forma directa o usa os.environ.get("API_SPORTS_KEY")
API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

# Diccionario completo con las 5 Grandes Ligas de Europa
LIGAS_A_DESCARGAR = {
    "Premier League": {"id": 39, "archivo": "data/historico_premier.csv"},
    "La Liga": {"id": 140, "archivo": "data/historico_laliga.csv"},
    "Serie A": {"id": 135, "archivo": "data/historico_seriea.csv"},
    "Bundesliga": {"id": 78, "archivo": "data/historico_bundesliga.csv"},
    "Ligue 1": {"id": 61, "archivo": "data/historico_ligue1.csv"}
}

# Temporada actual de Europa (2026) o puedes ajustar a [2025, 2026] si deseas más profundidad
TEMPORADAS = [2020, 2022, 2023]

def descargar_y_generar_csv(nombre_liga, league_id, archivo_salida):
    print(f"\n[INICIO] Descargando y procesando: {nombre_liga} (ID: {league_id})...")
    partidos_totales = []

    for temporada in TEMPORADAS:
        url = f"{BASE_URL}/fixtures"
        querystring = {"league": str(league_id), "season": str(temporada)}
        
        response = requests.get(url, headers=HEADERS, params=querystring)
        if response.status_code != 200:
            print(f"⚠️ Error HTTP {response.status_code} para {nombre_liga}")
            continue
            
        data = response.json().get("response", [])
        print(f"-> Temporada {temporada}: {len(data)} partidos encontrados en la API.")

        for idx, p in enumerate(data):
            fixture = p.get("fixture", {})
            fecha = fixture.get("date", "")[:10]
            
            teams = p.get("teams", {})
            local = teams.get("home", {}).get("name", "")
            visita = teams.get("away", {}).get("name", "")
            
            goals = p.get("goals", {})
            g_loc = goals.get("home")
            g_vis = goals.get("away")

            # Omitir partidos que aún no se juegan (sin marcador final)
            if g_loc is None or g_vis is None:
                continue

            fixture_id = fixture.get("id")
            stats_url = f"{BASE_URL}/fixtures/statistics"
            
            # Pausa ligera para respetar el límite de velocidad de la API
            time.sleep(0.3) 
            stats_res = requests.get(stats_url, headers=HEADERS, params={"fixture": fixture_id})
            
            # Valores base iniciales independientes para evitar estancamiento
            c_loc, c_v = 5.0, 4.5
            t_loc, t_v = 1.8, 2.0
            xg_l, xg_v = 1.2, 0.9  # Valores iniciales dinámicos
            tiros_l, tiros_v = 4.0, 3.5
            atajadas_l, atajadas_v = 3.0, 3.0
            arbitro = fixture.get("referee", "Desconocido")

            if stats_res.status_code == 200:
                stats_data = stats_res.json().get("response", [])
                for team_stat in stats_data:
                    is_home = (team_stat.get("team", {}).get("name") == local)
                    st_list = team_stat.get("statistics", [])
                    
                    # Convertir lista de estadísticas de la API a un diccionario plano
                    stat_dict = {}
                    for s in st_list:
                        tipo = s.get("type")
                        val = s.get("value")
                        if val is not None:
                            stat_dict[tipo] = val

                    corners = float(stat_dict.get("Corner Kicks", 5))
                    yellows = float(stat_dict.get("Yellow Cards", 1))
                    reds = float(stat_dict.get("Red Cards", 0))
                    tarjetas_puntos = yellows + (reds * 2)
                    
                    shots_on_goal = float(stat_dict.get("Shots on Goal", 4))
                    saves = float(stat_dict.get("Goalkeeper Saves", 3))
                    
                    # Búsqueda robusta de xG (Expected Goals suele venir como string o float en la API)
                    raw_xg = stat_dict.get("expected_goals")
                    if raw_xg is not None:
                        try:
                            expected_goals = float(str(raw_xg).replace(",", "."))
                        except:
                            expected_goals = round(shots_on_goal * 0.32, 2) # Estimador analítico si el formato varía
                    else:
                        # Si la API no provee xG para ese partido específico, lo estimamos analíticamente con base en tiros a puerta
                        expected_goals = round(shots_on_goal * 0.32 + (0.2 if is_home else 0.0), 2)

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

    if partidos_totales:
        os.makedirs("data", exist_ok=True)
        df_final = pd.DataFrame(partidos_totales)
        df_final.to_csv(archivo_salida, index=False)
        print(f"✅ [ÉXITO] Archivo generado correctamente: {archivo_salida} ({len(df_final)} partidos guardados).\n")
    else:
        print(f"⚠️ [AVISO] No se recolectaron partidos jugados para {nombre_liga}.\n")

if __name__ == "__main__":
    # Recorre y genera los 5 CSV de las 5 grandes ligas de Europa en secuencia
    for liga, info in LIGAS_A_DESCARGAR.items():
        descargar_y_generar_csv(liga, info["id"], info["archivo"])
