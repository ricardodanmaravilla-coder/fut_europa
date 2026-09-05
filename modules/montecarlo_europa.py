import numpy as np
import pandas as pd


def _safe_mean(s, default):
    try:
        v = float(pd.to_numeric(s, errors="coerce").mean())
        return default if np.isnan(v) else v
    except Exception:
        return default


def _rows_profile(x, team):
    rows = []
    for _, r in x.iterrows():
        home = str(r.get("Local", "")) == team
        rows.append({
            "xgf": r.get("xG_Local" if home else "xG_Visita"),
            "xga": r.get("xG_Visita" if home else "xG_Local"),
            "cf": r.get("Corners_Local" if home else "Corners_Visita"),
            "ca": r.get("Corners_Visita" if home else "Corners_Local"),
            "cards": r.get("Tarjetas_Local" if home else "Tarjetas_Visita"),
        })
    z = pd.DataFrame(rows)
    if z.empty:
        return None
    return {"xgf": _safe_mean(z["xgf"], 1.25), "xga": _safe_mean(z["xga"], 1.25),
            "cf": _safe_mean(z["cf"], 4.8), "ca": _safe_mean(z["ca"], 4.8),
            "cards": _safe_mean(z["cards"], 2.0), "n": len(z)}


def _team_recent(df, team, venue=None, n=12):
    if df is None or df.empty:
        return None
    x = df[(df["Local"] == team) | (df["Visitante"] == team)].copy()
    if x.empty:
        return None
    if "Fecha" in x.columns:
        x["_fecha"] = pd.to_datetime(x["Fecha"], errors="coerce", format="%Y-%m-%d")
        x = x.sort_values("_fecha", kind="stable")
    overall = _rows_profile(x.tail(n), team)
    if venue == "home": vx = x[x["Local"] == team].tail(max(6, n // 2))
    elif venue == "away": vx = x[x["Visitante"] == team].tail(max(6, n // 2))
    else: vx = x.iloc[0:0]
    venue_p = _rows_profile(vx, team)
    if overall is None: return venue_p
    if venue_p is None or venue_p["n"] < 3: return overall
    out = {k: 0.65 * venue_p[k] + 0.35 * overall[k] for k in ("xgf", "xga", "cf", "ca", "cards")}
    out["n"] = overall["n"]; out["venue_n"] = venue_p["n"]
    return out


def _ou_probs(values, line):
    line = float(line)
    over = 100.0 * np.mean(values > line); under = 100.0 * np.mean(values < line)
    push = 100.0 * np.mean(values == line) if line.is_integer() else 0.0
    return round(over, 1), round(under, 1), round(push, 1)


def simular_partido_europa(local, visita, df_historico, elo_local, elo_visita,
                            n_simulaciones=50000, seed=42, lineas_casino=None):
    pl = _team_recent(df_historico, local, "home") or {"xgf":1.35,"xga":1.25,"cf":5.0,"ca":4.8,"cards":2.0,"n":0}
    pv = _team_recent(df_historico, visita, "away") or {"xgf":1.15,"xga":1.35,"cf":4.5,"ca":5.0,"cards":2.1,"n":0}
    base_l = 0.55*pl["xgf"] + 0.45*pv["xga"]; base_v = 0.55*pv["xgf"] + 0.45*pl["xga"]
    elo_diff = float(np.clip((float(elo_local)+55.0-float(elo_visita))/400.0,-1.0,1.0))
    lambda_l=float(np.clip(base_l*np.exp(0.10*elo_diff),0.25,3.50)); lambda_v=float(np.clip(base_v*np.exp(-0.10*elo_diff),0.20,3.20))
    corners_l=float(np.clip((0.60*pl["cf"]+0.40*pv["ca"])*np.exp(0.025*elo_diff),1.5,9.0)); corners_v=float(np.clip((0.60*pv["cf"]+0.40*pl["ca"])*np.exp(-0.025*elo_diff),1.5,9.0))
    cards_l=float(np.clip(pl["cards"],0.5,5.0)); cards_v=float(np.clip(pv["cards"],0.5,5.0))
    rng=np.random.default_rng(seed)
    goles_l=rng.poisson(lambda_l,n_simulaciones); goles_v=rng.poisson(lambda_v,n_simulaciones)
    total_g=goles_l+goles_v; total_c=rng.poisson(corners_l,n_simulaciones)+rng.poisson(corners_v,n_simulaciones); total_t=rng.poisson(cards_l,n_simulaciones)+rng.poisson(cards_v,n_simulaciones)
    p_home=100*np.mean(goles_l>goles_v); p_draw=100*np.mean(goles_l==goles_v); p_away=100*np.mean(goles_l<goles_v)
    go,gu,_=_ou_probs(total_g,2.5); co,cu,_=_ou_probs(total_c,9.5); to,tu,_=_ou_probs(total_t,4.5)
    requested=lineas_casino if isinstance(lineas_casino,dict) else {}
    out={"Resultado_1X2":{"Gana Local":round(p_home,1),"Empate":round(p_draw,1),"Gana Visita":round(p_away,1)},"Goles_Over_Under":{"Over 2.5":go,"Under 2.5":gu},"Corners_Totales":{"Over 9.5 Corners":co,"Under 9.5 Corners":cu},"Tarjetas_Totales":{"Over 4.5 Tarjetas":to,"Under 4.5 Tarjetas":tu},"Goles_Individuales":{local:{"goles":round(lambda_l,2)},visita:{"goles":round(lambda_v,2)}},"Corners_Individuales":{local:{"corners":round(corners_l,2)},visita:{"corners":round(corners_v,2)}},"Tarjetas_Individuales":{local:{"tarjetas":round(cards_l,2)},visita:{"tarjetas":round(cards_v,2)}},"Lineas_Casino":{"goles":{},"corners":{},"tarjetas":{}},"Meta":{"recent_local":pl["n"],"recent_away":pv["n"],"venue_local":pl.get("venue_n",0),"venue_away":pv.get("venue_n",0),"elo_diff":round(elo_diff,3),"n_simulaciones":int(n_simulaciones),"sportsbook_lines_only":True,"line_lock":"sportsbook_requested_only","requested_lines":requested.copy(),"lineas_modeladas":{}}}
    for tipo,(values,label) in {"goles":(total_g,"Goles"),"corners":(total_c,"Corners"),"tarjetas":(total_t,"Tarjetas")}.items():
        raw=requested.get(tipo)
        if raw is None: continue
        try: line=round(float(raw),2)
        except Exception: continue
        over,under,push=_ou_probs(values,line)
        out["Lineas_Casino"][tipo]={f"Over {line:g} {label}":over,f"Under {line:g} {label}":under,f"Push {line:g} {label}":push}
        out["Meta"]["lineas_modeladas"][tipo]=line
    return out
