import os
import requests
import pandas as pd

API_KEY = os.environ.get("API_SPORTS_KEY") 
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {'x-apisports-key': API_KEY}

def obtener_cuotas_europa(fixture_id, bookmaker_id=8):
    """
    Extrae cuotas reales usando la API de API-Sports.
    bookmaker_id=8 suele ser Bet365 (muy estándar para Europa).
    """
    url = f"{BASE_URL}/odds"
    querystring = {"fixture": str(fixture_id), "bookmaker": str(bookmaker_id)}
    
    cuotas = {}
    try:
        response = requests.get(url, headers=HEADERS, params=querystring, timeout=5)
        if response.status_code == 200:
            data = response.json().get("response", [])
            if data and len(data) > 0:
                bookmakers = data[0].get("bookmakers", [])
                if bookmakers:
                    mercados = bookmakers[0].get("bets", [])
                    for mercado in mercados:
                        # Mercado 1: Ganador del Partido (1X2)
                        if mercado["id"] == 1:
                            for valor in mercado["values"]:
                                if valor["value"] == "Home": cuotas["1"] = float(valor["odd"])
                                elif valor["value"] == "Draw": cuotas["X"] = float(valor["odd"])
                                elif valor["value"] == "Away": cuotas["2"] = float(valor["odd"])
                        # Mercado 5: Over/Under Goles (Tomamos la línea de 2.5)
                        elif mercado["id"] == 5:
                            for valor in mercado["values"]:
                                if valor["value"] == "Over 2.5": cuotas["Over 2.5"] = float(valor["odd"])
                                elif valor["value"] == "Under 2.5": cuotas["Under 2.5"] = float(valor["odd"])
    except Exception as e:
        print(f"Error extrayendo cuotas europeas: {e}")
        
    return cuotas

def calcular_kelly_fraccional(prob_modelo_decimal, cuota_decimal, fraccion=0.25):
    """
    Criterio de Kelly Fraccional (1/4) para gestionar el bankroll en ligas europeas.
    Reduce la varianza protegiendo el capital a largo plazo.
    """
    if cuota_decimal <= 1.0 or prob_modelo_decimal <= 0 or prob_modelo_decimal >= 1:
        return 0.0
    
    q = 1.0 - prob_modelo_decimal
    b = cuota_decimal - 1.0
    kelly_puro = ( (b * prob_modelo_decimal) - q ) / b
    
    if kelly_puro <= 0:
        return 0.0
        
    return round((kelly_puro * fraccion) * 100, 2)

def analizar_apuestas_europa(resultados_modelo, fixture_id, cuotas_personalizadas=None):
    """
    Cruza las probabilidades del modelo (Montecarlo/ML) con las cuotas para buscar EV+.
    """
    cuotas_api = obtener_cuotas_europa(fixture_id) if not cuotas_personalizadas else {}
    cuotas_finales = cuotas_personalizadas if cuotas_personalizadas else cuotas_api

    if not cuotas_finales:
        return pd.DataFrame()

    analisis = []
    
    # Mapeo de los mercados del modelo hacia las llaves de cuotas
    mapeo_mercados = [
        ("Gana Local", resultados_modelo['Resultado_1X2']['Gana Local'], "1"),
        ("Empate", resultados_modelo['Resultado_1X2']['Empate'], "X"),
        ("Gana Visita", resultados_modelo['Resultado_1X2']['Gana Visita'], "2"),
        ("Over 2.5 Goles", resultados_modelo['Goles_Over_Under']['Over 2.5'], "Over 2.5"),
        ("Under 2.5 Goles", resultados_modelo['Goles_Over_Under']['Under 2.5'], "Under 2.5"),
        ("Over 9.5 Corners", resultados_modelo['Corners_Totales']['Over 9.5 Corners'], "Over 9.5 Corners"),
        ("Under 9.5 Corners", resultados_modelo['Corners_Totales'].get('Under 9.5 Corners', 0), "Under 9.5 Corners"),
        ("Over 4.5 Tarjetas", resultados_modelo['Tarjetas_Totales']['Over 4.5 Tarjetas'], "Over 4.5 Tarjetas"),
        ("Under 4.5 Tarjetas", resultados_modelo['Tarjetas_Totales'].get('Under 4.5 Tarjetas', 0), "Under 4.5 Tarjetas")
    ]

    for nombre_mercado, prob_modelo_pct, llave_cuota in mapeo_mercados:
        cuota = float(cuotas_finales.get(llave_cuota, 0.0))
        
        if cuota > 1.01:
            prob_implicita = (1 / cuota) * 100
            prob_modelo = prob_modelo_pct / 100.0
            
            # Valor Esperado (EV)
            ev = (prob_modelo * (cuota - 1)) - (1 - prob_modelo)
            ev_pct = round(ev * 100, 2)
            
            # Gestión de Bankroll
            kelly_rec = calcular_kelly_fraccional(prob_modelo, cuota)
            
            # Filtro de Veredicto Institucional
            if ev_pct > 15 and kelly_rec > 2.0:
                veredicto = "🔥 Value Fuerte (Apostar)"
            elif ev_pct > 3 and kelly_rec > 0.5:
                veredicto = "✅ Value Moderado"
            elif ev_pct > 0:
                veredicto = "⚠️ EV Positivo Marginal (Opcional)"
            else:
                veredicto = "❌ EV Negativo (No Apostar)"
                
            analisis.append({
                "Mercado": nombre_mercado,
                "Prob. Modelo": f"{prob_modelo_pct}%",
                "Cuota Casino": cuota,
                "Prob. Implícita": f"{round(prob_implicita, 1)}%",
                "EV (Valor)": f"{ev_pct}%",
                "Stake Recomendado": f"{kelly_rec}% Bank",
                "Veredicto": veredicto
            })

    return pd.DataFrame(analisis)
