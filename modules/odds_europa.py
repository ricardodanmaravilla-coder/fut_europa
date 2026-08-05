import os
import requests
import pandas as pd
import unicodedata

API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

def american_to_decimal(american):
    """Convierte momios americanos de ESPN (+150 / -120) a decimales europeos (2.50 / 1.83)"""
    if american == 0: return 0.0
    if american > 0:
        return round((american / 100.0) + 1.0, 2)
    else:
        return round((100.0 / abs(american)) + 1.0, 2)

def normalizar_nombre(nombre):
    """Limpia tildes y caracteres raros para comparar nombres de equipos"""
    return unicodedata.normalize('NFKD', nombre).encode('ASCII', 'ignore').decode('utf-8').lower()

def extraer_cuotas_espn(nombre_liga, local, visita):
    """Se conecta a la API oculta de ESPN para robar las cuotas reales si API-Sports falla"""
    espn_leagues = {
        "Premier League": "eng.1",
        "La Liga": "esp.1",
        "Serie A": "ita.1",
        "Bundesliga": "ger.1",
        "Ligue 1": "fra.1"
    }
    
    codigo_espn = espn_leagues.get(nombre_liga)
    if not codigo_espn: return {}

    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{codigo_espn}/scoreboard"
    cuotas = {}
    
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            for event in data.get('events', []):
                name_event = normalizar_nombre(event.get('name', ''))
                # Tomamos solo las primeras 4 letras para evitar errores tipográficos entre ESPN y API-Sports
                loc_norm = normalizar_nombre(local)[:4] 
                vis_norm = normalizar_nombre(visita)[:4]

                if loc_norm in name_event and vis_norm in name_event:
                    competencia = event.get('competitions', [{}])[0]
                    odds_list = competencia.get('odds', [])
                    if odds_list:
                        odds_data = odds_list[0]
                        
                        # Extraer 1X2 desde ESPN
                        if 'homeTeamOdds' in odds_data:
                            cuotas["1"] = american_to_decimal(odds_data['homeTeamOdds'].get('moneyLine', 0))
                        if 'drawOdds' in odds_data:
                            cuotas["X"] = american_to_decimal(odds_data['drawOdds'].get('moneyLine', 0))
                        if 'awayTeamOdds' in odds_data:
                            cuotas["2"] = american_to_decimal(odds_data['awayTeamOdds'].get('moneyLine', 0))
                            
                        # Extraer Over/Under (A veces ESPN solo pone la línea 2.5, asumimos cuota 1.90 de casino estándar)
                        if 'overUnder' in odds_data:
                            linea = odds_data['overUnder']
                            if linea == 2.5:
                                cuotas["Over 2.5"] = 1.90
                                cuotas["Under 2.5"] = 1.90
                    break # Salimos del loop al encontrar el partido
    except Exception as e:
        print(f"Fallo conexión con ESPN: {e}")
        
    return cuotas

def obtener_cuotas_europa(fixture_id, nombre_liga=None, local=None, visita=None):
    """Intenta API-Sports. Si está vacío, inyecta datos reales de ESPN."""
    cuotas = {
        "1": 0.0, "X": 0.0, "2": 0.0, 
        "Over 2.5": 0.0, "Under 2.5": 0.0,
        "Over 9.5 Corners": 0.0, "Under 9.5 Corners": 0.0,
        "Over 4.5 Tarjetas": 0.0, "Under 4.5 Tarjetas": 0.0
    }
    
    # 1. INTENTO OFICIAL (API-Sports)
    url = f"{BASE_URL}/odds"
    querystring = {"fixture": str(fixture_id)}
    exito_api = False
    
    try:
        response = requests.get(url, headers=HEADERS, params=querystring, timeout=5)
        if response.status_code == 200:
            data = response.json().get("response", [])
            if data and len(data) > 0:
                bookmakers = data[0].get("bookmakers", [])
                if bookmakers:
                    bm_elegido = next((bm for bm in bookmakers if bm["id"] == 8), bookmakers[0])
                    mercados = bm_elegido.get("bets", [])
                    
                    for mercado in mercados:
                        if mercado["id"] == 1:
                            for valor in mercado["values"]:
                                if valor["value"] == "Home": cuotas["1"] = float(valor["odd"]); exito_api = True
                                elif valor["value"] == "Draw": cuotas["X"] = float(valor["odd"])
                                elif valor["value"] == "Away": cuotas["2"] = float(valor["odd"])
                        elif mercado["id"] == 5:
                            for valor in mercado["values"]:
                                if valor["value"] == "Over 2.5": cuotas["Over 2.5"] = float(valor["odd"])
                                elif valor["value"] == "Under 2.5": cuotas["Under 2.5"] = float(valor["odd"])
                        elif mercado["id"] == 45: 
                            for valor in mercado["values"]:
                                if valor["value"] == "Over 9.5": cuotas["Over 9.5 Corners"] = float(valor["odd"])
                                elif valor["value"] == "Under 9.5": cuotas["Under 9.5 Corners"] = float(valor["odd"])
    except:
        pass

    # 2. SISTEMA DE RESPALDO: EXTRACCIÓN DE ESPN
    if not exito_api and nombre_liga and local and visita:
        cuotas_espn = extraer_cuotas_espn(nombre_liga, local, visita)
        for k, v in cuotas_espn.items():
            if v > 0:
                cuotas[k] = v

    return cuotas

def calcular_kelly_fraccional(prob_modelo_decimal, cuota_decimal, fraccion=0.25):
    if cuota_decimal <= 1.0 or prob_modelo_decimal <= 0 or prob_modelo_decimal >= 1: return 0.0
    q = 1.0 - prob_modelo_decimal
    b = cuota_decimal - 1.0
    kelly_puro = ( (b * prob_modelo_decimal) - q ) / b
    if kelly_puro <= 0: return 0.0
    return round((kelly_puro * fraccion) * 100, 2)

def analizar_apuestas_europa(resultados_mc, preds_ml, fixture_id, cuotas_personalizadas=None, nombre_liga=None, local=None, visita=None):
    if cuotas_personalizadas:
        cuotas_finales = cuotas_personalizadas
    else:
        cuotas_finales = obtener_cuotas_europa(fixture_id, nombre_liga, local, visita)

    if not cuotas_finales: return pd.DataFrame()

    analisis = []
    
    # Función auxiliar para extraer datos de forma segura
    def get_prob(d, cat, key):
        if not d or cat not in d: return 0.0
        return float(d[cat].get(key, 0.0))

    mercados_lista = [
        ("Gana Local", "Resultado_1X2", "Gana Local", "1"),
        ("Empate", "Resultado_1X2", "Empate", "X"),
        ("Gana Visita", "Resultado_1X2", "Gana Visita", "2"),
        ("Over 2.5 Goles", "Goles_Over_Under", "Over 2.5", "Over 2.5"),
        ("Under 2.5 Goles", "Goles_Over_Under", "Under 2.5", "Under 2.5"),
        ("Over 9.5 Corners", "Corners_Totales", "Over 9.5 Corners", "Over 9.5 Corners"),
        ("Under 9.5 Corners", "Corners_Totales", "Under 9.5 Corners", "Under 9.5 Corners"),
        ("Over 4.5 Tarjetas", "Tarjetas_Totales", "Over 4.5 Tarjetas", "Over 4.5 Tarjetas"),
        ("Under 4.5 Tarjetas", "Tarjetas_Totales", "Under 4.5 Tarjetas", "Under 4.5 Tarjetas")
    ]

    for nombre_mercado, cat, llave_dict, llave_cuota in mercados_lista:
        cuota = float(cuotas_finales.get(llave_cuota, 0.0))
        if cuota > 1.01:
            # Extraemos lo que dice cada modelo
            prob_mc = get_prob(resultados_mc, cat, llave_dict)
            prob_ml = get_prob(preds_ml, cat, llave_dict) if preds_ml else prob_mc
            
            prob_consenso = round((prob_mc + prob_ml) / 2.0, 1)
            prob_modelo = prob_consenso / 100.0
            prob_implicita = (1 / cuota) * 100
            
            ev = (prob_modelo * (cuota - 1)) - (1 - prob_modelo)
            ev_pct = round(ev * 100, 2)
            kelly_rec = calcular_kelly_fraccional(prob_modelo, cuota)
            
            # =======================================================
            # 🛡️ FILTRO FRANCOTIRADOR: Ambos modelos > 60%
            # =======================================================
            if prob_mc >= 60.0 and prob_ml >= 60.0:
                if ev_pct > 10 and kelly_rec > 1.5: veredicto = "🔥 Value Fuerte (Apostar)"
                elif ev_pct > 3 and kelly_rec > 0.5: veredicto = "✅ Value Moderado"
                elif ev_pct > 0: veredicto = "⚠️ EV Positivo Marginal"
                else: veredicto = "❌ EV Negativo"
            else:
                veredicto = "❌ Descartado (No superan el 60%)"
                kelly_rec = 0.0
                
            analisis.append({
                "Mercado": nombre_mercado,
                "Prob. MC": f"{prob_mc}%",
                "Prob. ML": f"{prob_ml}%",
                "Consenso": f"{prob_consenso}%",
                "Cuota Casino": cuota,
                "EV (Valor)": f"{ev_pct}%",
                "Stake Recomendado": f"{kelly_rec}% Bank",
                "Veredicto": veredicto
            })

    return pd.DataFrame(analisis)
