import numpy as np
import pandas as pd

def simular_partido_europa(local, visita, df_historico, elo_local, elo_visita, n_simulaciones=100000):
    """
    Simulador Montecarlo especializado para fútbol europeo.
    Utiliza xG históricos, tiros a gol, atajadas y ajuste por ELO.
    """
    if df_historico is None or df_historico.empty:
        # Valores por defecto de emergencia si el CSV está vacío
        xg_l, xg_v = 1.4, 1.1
        corners_l, corners_v = 5.2, 4.5
        tarjetas_l, tarjetas_v = 2.0, 2.2
    else:
        # Filtrar datos de los equipos
        df_loc = df_historico[df_historico['Local'] == local]
        df_vis = df_historico[df_historico['Visitante'] == visita]
        
        # Extracción de métricas avanzadas (xG y Tiros a Gol)
        xg_l = df_loc['xG_Local'].mean() if not df_loc.empty and 'xG_Local' in df_loc.columns else 1.4
        xg_v = df_vis['xG_Visita'].mean() if not df_vis.empty and 'xG_Visita' in df_vis.columns else 1.1
        
        corners_l = df_loc['Corners_Local'].mean() if not df_loc.empty and 'Corners_Local' in df_loc.columns else 5.2
        corners_v = df_vis['Corners_Visita'].mean() if not df_vis.empty and 'Corners_Visita' in df_vis.columns else 4.5
        
        tarjetas_l = df_loc['Tarjetas_Local'].mean() if not df_loc.empty and 'Tarjetas_Local' in df_loc.columns else 2.0
        tarjetas_v = df_vis['Tarjetas_Visita'].mean() if not df_vis.empty and 'Tarjetas_Visita' in df_vis.columns else 2.2

    # Ajuste matemático por diferencia de ELO
    factor_elo = (elo_local - elo_visita) / 400.0
    lambda_l = max(0.4, xg_l + (factor_elo * 0.25))
    lambda_v = max(0.4, xg_v - (factor_elo * 0.25))

    # Simulaciones de Poisson (Goles)
    goles_l = np.random.poisson(lam=lambda_l, size=n_simulaciones)
    goles_v = np.random.poisson(lam=lambda_v, size=n_simulaciones)

    # Simulaciones para Córners y Tarjetas
    sim_corners_l = np.random.normal(loc=corners_l, scale=1.6, size=n_simulaciones).clip(0, 16)
    sim_corners_v = np.random.normal(loc=corners_v, scale=1.5, size=n_simulaciones).clip(0, 16)
    tot_corners = sim_corners_l + sim_corners_v

    sim_tarjetas_l = np.random.poisson(lam=tarjetas_l, size=n_simulaciones)
    sim_tarjetas_v = np.random.poisson(lam=tarjetas_v, size=n_simulaciones)
    tot_tarjetas = sim_tarjetas_l + sim_tarjetas_v

    # Cálculo de Probabilidades Porcentuales
    ganan_local = np.sum(goles_l > goles_v) / n_simulaciones * 100
    empates = np.sum(goles_l == goles_v) / n_simulaciones * 100
    ganan_visita = np.sum(goles_l < goles_v) / n_simulaciones * 100

    over_25 = np.sum((goles_l + goles_v) > 2.5) / n_simulaciones * 100
    under_25 = 100.0 - over_25
    
    over_corners = np.sum(tot_corners > 9.5) / n_simulaciones * 100
    over_tarjetas = np.sum(tot_tarjetas > 4.5) / n_simulaciones * 100

    return {
        "Resultado_1X2": {
            "Gana Local": round(ganan_local, 1), 
            "Empate": round(empates, 1), 
            "Gana Visita": round(ganan_visita, 1)
        },
        "Goles_Over_Under": {
            "Over 2.5": round(over_25, 1), 
            "Under 2.5": round(under_25, 1)
        },
        "Corners_Totales": {
            "Over 9.5 Corners": round(over_corners, 1),
            "Under 9.5 Corners": round(100.0 - over_corners, 1)
        },
        "Tarjetas_Totales": {
            "Over 4.5 Tarjetas": round(over_tarjetas, 1),
            "Under 4.5 Tarjetas": round(100.0 - over_tarjetas, 1)
        },
        "Goles_Individuales": {
            local: {"goles": round(lambda_l, 2)}, 
            visita: {"goles": round(lambda_v, 2)}
        },
        "Corners_Individuales": {
            local: {"corners": round(corners_l, 1)}, 
            visita: {"corners": round(corners_v, 1)}
        },
        "Tarjetas_Individuales": {
            local: {"tarjetas": round(tarjetas_l, 1)}, 
            visita: {"tarjetas": round(tarjetas_v, 1)}
        }
    }
