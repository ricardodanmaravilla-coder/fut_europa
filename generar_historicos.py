import os
import time
import requests
import pandas as pd

API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

LIGAS_A_DESCARGAR = {
    "Premier League": {"id": 39, "archivo": "data/historico_premier.csv"},
    "La Liga": {"id": 140, "archivo": "data/historico_laliga.csv"},
    "Serie A": {"id": 135, "archivo": "data/historico_seriea.csv"},
    "Bundesliga": {"id": 78, "archivo": "data/historico_bundesliga.csv"},
    "Ligue 1": {"id": 61, "archivo": "data/historico_ligue1.csv"}
}

# Nos enfocamos principalmente en la temporada actual (2026) para actualizar resultados recientes
TEMPORADA_ACTUAL = 2026

def actualizar_historicos_liga(nombre_liga, league_id, archivo_salida):
    print(f"\n[INICIO] Actualizando datos para: {nombre_liga} (ID: {league_id})...")
    
    # 1. Cargar CSV existente si ya lo tenemos guardado
    df_existente = pd.DataFrame()
    fechas_existentes = set()
    if os.path.exists(archivo_salida):
        try:
            df_existente = pd.read_csv(archivo_salida)
            # Creamos una llave única combinando Fecha + Local + Visitante para evitar duplicados
            if not df_existente.empty and 'Fecha' in df_existente.columns:
                fechas_existentes = set(zip(df_existente['Fecha'], df_existente['Local'], df_existente['Visitante']))
            print(f"-> Histórico local cargado: {len(df_existente)} partidos previos registrados.")
        except Exception as e:
            print(f"⚠️ No se pudo leer el CSV existente, se creará uno nuevo. Detalle: {e}")

    # 2. Consultar solo la temporada actual a la API (Mucho más rápido y sin bloqueos)
    url = f"{BASE_URL}/fixtures"
    querystring = {"league": str(league_id), "season": str(TEMPORADA_ACTUAL)}
    
    response = requests.get(url, headers=HEADERS, params=querystring, timeout=10)
    if response.status_code != 200:
        print(f"⚠️ Error HTTP {response.status_code} para {nombre_liga}")
        return
        
    data = response.json().get("response", [])
    print(f"-> Temporada {TEMPORADA_ACTUAL}: {len(data)} partidos totales en la cartelera de la API.")

    nuevos_partidos = []

    for p in data:
        fixture = p.get("fixture", {})
        fecha = fixture.get("date", "")[:10]
        
        teams = p.get("teams", {})
        local = teams.get("home", {}).get("name", "").strip()
        visita = teams.get("away", {}).get("name", "").strip()
        
        goals = p.get("goals", {})
        g_loc = goals.get("home")
        g_vis = goals.get("away")

        # Omitir partidos que aún no se juegan (sin marcador final)
        if g_loc is None or g_vis is None:
            continue

        # Si este partido exacto ya está guardado en nuestro CSV, lo saltamos para ahorrar tiempo y peticiones
        if (fecha, local, visita) in fechas_existentes:
            continue

        fixture_id = fixture.get("id")
        stats_url = f"{BASE_URL}/fixtures/statistics"
        
        time.sleep(0.2) # Pausa ligera de seguridad
        try:
            stats_res = requests.get(stats_url, headers=HEADERS, params={"fixture": fixture_id}, timeout=5)
        except:
            stats_res = None
        
        c_loc, c_v = 5.0, 4.5
        t_loc, t_v = 1.8, 2.0
        xg_l, xg_v = 1.2, 0.9
        tiros_l, tiros_v = 4.0, 3.5
        atajadas_l, atajadas_v = 3.0, 3.0
        arbitro = fixture.get("referee", "Desconocido")

        if stats_res and stats_res.status_code == 200:
            stats_data = stats_res.json().get("response", [])
            for team_stat in stats_data:
                is_home = (team_stat.get("team", {}).get("name", "").strip() == local)
                st_list = team_stat.get("statistics", [])
                
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
                
                raw_xg = stat_dict.get("expected_goals")
                if raw_xg is not None:
                    try:
                        expected_goals = float(str(raw_xg).replace(",", "."))
                    except:
                        expected_goals = round(shots_on_goal * 0.32, 2)
                else:
                    expected_goals = round(shots_on_goal * 0.32 + (0.2 if is_home else 0.0), 2)

                if is_home:
                    c_loc, t_loc, tiros_l, atajadas_l, xg_l = corners, tarjetas_puntos, shots_on_goal, saves, expected_goals
                else:
                    c_v, t_v, tiros_v, atajadas_v, xg_v = corners, tarjetas_puntos, shots_on_goal, saves, expected_goals

        nuevos_partidos.append({
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

    if nuevos_partidos:
        df_nuevos = pd.DataFrame(nuevos_partidos)
        # Combinar el histórico viejo con los partidos nuevos recién jugados
        df_final = pd.concat([df_existente, df_nuevos], ignore_index=True)
        # Ordenar cronológicamente por fecha
        if 'Fecha' in df_final.columns:
            df_final = df_final.sort_values(by='Fecha').reset_index(drop=True)
            
        os.makedirs("data", exist_ok=True)
        df_final.to_csv(archivo_salida, index=False)
        print(f"✅ [ÉXITO] Se añadieron {len(df_nuevos)} partidos nuevos. Total en archivo: {len(df_final)} partidos.\n")
    else:
        print(f"ℹ️ [INFO] No hay partidos nuevos por agregar para {nombre_liga}. El archivo ya está actualizado.\n")

if __name__ == "__main__":
    for liga, info in LIGAS_A_DESCARGAR.items():
        actualizar_historicos_liga(liga, info["id"], info["archivo"])
