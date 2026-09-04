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
        "xgf": _safe_mean(z["xgf"], 1.25), "xga": _safe_mean(z["xga"], 1.25),
        "cf": _safe_mean(z["cf"], 4.8), "ca": _safe_mean(z["ca"], 4.8),
        "cards": _safe_mean(z["cards"], 2.0), "n": len(z),
    }


def _ou_probs(values, line):
    line = float(line)
    over = 100.0 * np.mean(values > line)
    under = 100.0 * np.mean(values < line)
    push = 100.0 * np.mean(values == line) if float(line).is_integer() else 0.0
    return round(over, 1), round(under, 1), round(push, 1)


def simular_partido_europa(local, visita, df_historico, elo_local, elo_visita,
                            n_simulaciones=50000, seed=42, lineas_casino=None):
    pl = _team_recent(df_historico, local) or {"xgf": 1.35, "xga": 1.25, "cf": 5.0, "ca": 4.8, "cards": 2.0, "n": 0}
    pv = _team_recent(df_historico, visita) or {"xgf": 1.15, "xga": 1.35, "cf": 4.5, "ca": 5.0, "cards": 2.1, "n": 0}

    base_l = 0.55 * pl["xgf"] + 0.45 * pv["xga"]
    base_v = 0.55 * pv["xgf"] + 0.45 * pl["xga"]
    elo_diff = float(np.clip((float(elo_local) - float(elo_visita)) / 400.0, -1.0, 1.0))
    lambda_l = float(np.clip(base_l * np.exp(0.10 * elo_diff) + 0.10, 0.25, 3.50))
    lambda_v = float(np.clip(base_v * np.exp(-0.10 * elo_diff), 0.20, 3.20))
    corners_l = float(np.clip(0.60 * pl["cf"] + 0.40 * pv["ca"], 1.5, 9.0))
    corners_v = float(np.clip(0.60 * pv["cf"] + 0.40 * pl["ca"], 1.5, 9.0))
    cards_l = float(np.clip(pl["cards"], 0.5, 5.0))
    cards_v = float(np.clip(pv["cards"], 0.5, 5.0))

    rng = np.random.default_rng(seed)
    goles_l = rng.poisson(lambda_l, n_simulaciones)
    goles_v = rng.poisson(lambda_v, n_simulaciones)
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
    g_over, g_under, _ = _ou_probs(total_g, 2.5)
    c_over, c_under, _ = _ou_probs(total_c, 9.5)
    t_over, t_under, _ = _ou_probs(total_t, 4.5)

    out = {
        "Resultado_1X2": {"Gana Local": round(p_home, 1), "Empate": round(p_draw, 1), "Gana Visita": round(p_away, 1)},
        "Goles_Over_Under": {"Over 2.5": g_over, "Under 2.5": g_under},
        "Corners_Totales": {"Over 9.5 Corners": c_over, "Under 9.5 Corners": c_under},
        "Tarjetas_Totales": {"Over 4.5 Tarjetas": t_over, "Under 4.5 Tarjetas": t_under},
        "Goles_Individuales": {local: {"goles": round(lambda_l, 2)}, visita: {"goles": round(lambda_v, 2)}},
        "Corners_Individuales": {local: {"corners": round(corners_l, 2)}, visita: {"corners": round(corners_v, 2)}},
        "Tarjetas_Individuales": {local: {"tarjetas": round(cards_l, 2)}, visita: {"tarjetas": round(cards_v, 2)}},
        "Lineas_Casino": {},
        "Meta": {"recent_local": pl["n"], "recent_away": pv["n"], "elo_diff": round(elo_diff, 3)},
    }

    specs = {
        "goles": (total_g, "Goles", np.arange(0.5, 6.01, 0.25)),
        "corners": (total_c, "Corners", np.arange(5.0, 16.01, 0.5)),
        "tarjetas": (total_t, "Tarjetas", np.arange(1.0, 10.01, 0.5)),
    }
    requested = lineas_casino if isinstance(lineas_casino, dict) else {}
    for tipo, (values, label, grid) in specs.items():
        lines = {round(float(x), 2) for x in grid}
        if requested.get(tipo) is not None:
            try:
                lines.add(round(float(requested[tipo]), 2))
            except Exception:
                pass
        markets = {}
        for line in sorted(lines):
            over, under, push = _ou_probs(values, line)
            markets[f"Over {line:g} {label}"] = over
            markets[f"Under {line:g} {label}"] = under
            markets[f"Push {line:g} {label}"] = push
        out["Lineas_Casino"][tipo] = markets

    return out
