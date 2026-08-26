import numpy as np
import pandas as pd


def _safe_mean(s, default):
    try:
        v = float(pd.to_numeric(s, errors="coerce").mean())
        return default if np.isnan(v) else v
    except Exception:
        return default


def _team_recent(df, team, n=12):
    if df is None or df.empty:
        return None
    x = df[(df["Local"] == team) | (df["Visitante"] == team)].copy()
    if x.empty:
        return None
    if "Fecha" in x.columns:
        x["_fecha"] = pd.to_datetime(x["Fecha"], errors="coerce", dayfirst=True)
        x = x.sort_values("_fecha")
    x = x.tail(n)

    rows = []
    for _, r in x.iterrows():
        home = r["Local"] == team
        rows.append({
            "xgf": r.get("xG_Local" if home else "xG_Visita"),
            "xga": r.get("xG_Visita" if home else "xG_Local"),
            "cf": r.get("Corners_Local" if home else "Corners_Visita"),
            "ca": r.get("Corners_Visita" if home else "Corners_Local"),
            "cards": r.get("Tarjetas_Local" if home else "Tarjetas_Visita"),
        })
    z = pd.DataFrame(rows)
    return {
        "xgf": _safe_mean(z["xgf"], 1.25),
        "xga": _safe_mean(z["xga"], 1.25),
        "cf": _safe_mean(z["cf"], 4.8),
        "ca": _safe_mean(z["ca"], 4.8),
        "cards": _safe_mean(z["cards"], 2.0),
        "n": len(z),
    }


def simular_partido_europa(local, visita, df_historico, elo_local, elo_visita, n_simulaciones=50000, seed=42):
    """Simulación prepartido con ataque/defensa rival + forma reciente + Elo.

    No usa estadísticas del partido objetivo. `df_historico` debe contener sólo
    partidos ya finalizados disponibles al momento del pronóstico.
    """
    pl = _team_recent(df_historico, local) or {"xgf": 1.35, "xga": 1.25, "cf": 5.0, "ca": 4.8, "cards": 2.0, "n": 0}
    pv = _team_recent(df_historico, visita) or {"xgf": 1.15, "xga": 1.35, "cf": 4.5, "ca": 5.0, "cards": 2.1, "n": 0}

    # Ataque propio combinado con lo que concede el rival.
    base_l = 0.55 * pl["xgf"] + 0.45 * pv["xga"]
    base_v = 0.55 * pv["xgf"] + 0.45 * pl["xga"]

    # Ajuste Elo acotado para impedir que domine el modelo de goles.
    elo_diff = float(np.clip((float(elo_local) - float(elo_visita)) / 400.0, -1.0, 1.0))
    home_adv = 0.10
    lambda_l = float(np.clip(base_l * np.exp(0.10 * elo_diff) + home_adv, 0.25, 3.50))
    lambda_v = float(np.clip(base_v * np.exp(-0.10 * elo_diff), 0.20, 3.20))

    corners_l = float(np.clip(0.60 * pl["cf"] + 0.40 * pv["ca"], 1.5, 9.0))
    corners_v = float(np.clip(0.60 * pv["cf"] + 0.40 * pl["ca"], 1.5, 9.0))
    cards_l = float(np.clip(pl["cards"], 0.5, 5.0))
    cards_v = float(np.clip(pv["cards"], 0.5, 5.0))

    rng = np.random.default_rng(seed)
    goles_l = rng.poisson(lambda_l, n_simulaciones)
    goles_v = rng.poisson(lambda_v, n_simulaciones)
    # Córners son discretos y sobredispersos; Poisson es más coherente que normal continua.
    sim_corners_l = rng.poisson(corners_l, n_simulaciones)
    sim_corners_v = rng.poisson(corners_v, n_simulaciones)
    sim_cards_l = rng.poisson(cards_l, n_simulaciones)
    sim_cards_v = rng.poisson(cards_v, n_simulaciones)

    total_g = goles_l + goles_v
    total_c = sim_corners_l + sim_corners_v
    total_t = sim_cards_l + sim_cards_v

    p_home = 100.0 * np.mean(goles_l > goles_v)
    p_draw = 100.0 * np.mean(goles_l == goles_v)
    p_away = 100.0 * np.mean(goles_l < goles_v)
    p_o25 = 100.0 * np.mean(total_g > 2.5)
    p_c95 = 100.0 * np.mean(total_c > 9.5)
    p_t45 = 100.0 * np.mean(total_t > 4.5)

    return {
        "Resultado_1X2": {"Gana Local": round(p_home, 1), "Empate": round(p_draw, 1), "Gana Visita": round(p_away, 1)},
        "Goles_Over_Under": {"Over 2.5": round(p_o25, 1), "Under 2.5": round(100-p_o25, 1)},
        "Corners_Totales": {"Over 9.5 Corners": round(p_c95, 1), "Under 9.5 Corners": round(100-p_c95, 1)},
        "Tarjetas_Totales": {"Over 4.5 Tarjetas": round(p_t45, 1), "Under 4.5 Tarjetas": round(100-p_t45, 1)},
        "Goles_Individuales": {local: {"goles": round(lambda_l, 2)}, visita: {"goles": round(lambda_v, 2)}},
        "Corners_Individuales": {local: {"corners": round(corners_l, 2)}, visita: {"corners": round(corners_v, 2)}},
        "Tarjetas_Individuales": {local: {"tarjetas": round(cards_l, 2)}, visita: {"tarjetas": round(cards_v, 2)}},
        "Meta": {"recent_local": pl["n"], "recent_away": pv["n"], "elo_diff": round(elo_diff, 3)},
    }
