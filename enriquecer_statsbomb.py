"""Capa avanzada gratuita con StatsBomb Open Data.

Descarga únicamente competiciones abiertas de las cinco ligas objetivo, conserva un
Parquet de eventos mínimo, alineaciones y un agregado por partido con xG real,
passes, pressures, possessions, PPDA y xThreat. Si existe el grid aprendido por
wyscout_event_pipeline.py, lo transfiere a coordenadas StatsBomb normalizadas.
La cobertura es parcial por diseño.
"""
import json
import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = "data/statsbomb_xg_matches.csv"
EVENTS_OUT = "data/statsbomb_events.parquet"
LINEUPS_OUT = "data/statsbomb_lineups.parquet"
FEATURES_OUT = "data/statsbomb_match_features.parquet"
WYSCOUT_GRID = "data/wyscout_xt_grid.json"
TARGET_NAMES = {
    "Premier League": ["premier league"], "La Liga": ["la liga"],
    "Serie A": ["serie a"], "Bundesliga": ["bundesliga"], "Ligue 1": ["ligue 1"],
}
MIN_DATE = pd.Timestamp("2020-07-01")
MAX_WORKERS = 8

# Fallback histórico 12x8. Sólo se usa si no existe el grid Wyscout aprendido.
XT_FALLBACK = np.array([
 [0.0064,0.0078,0.0084,0.0092,0.0113,0.0121,0.0148,0.0170,0.0222,0.0307,0.0433,0.0745],
 [0.0070,0.0082,0.0090,0.0100,0.0120,0.0132,0.0160,0.0184,0.0244,0.0340,0.0500,0.0960],
 [0.0075,0.0088,0.0097,0.0108,0.0130,0.0145,0.0178,0.0208,0.0280,0.0405,0.0630,0.1320],
 [0.0078,0.0092,0.0102,0.0115,0.0138,0.0155,0.0195,0.0230,0.0318,0.0470,0.0780,0.1800],
 [0.0078,0.0092,0.0102,0.0115,0.0138,0.0155,0.0195,0.0230,0.0318,0.0470,0.0780,0.1800],
 [0.0075,0.0088,0.0097,0.0108,0.0130,0.0145,0.0178,0.0208,0.0280,0.0405,0.0630,0.1320],
 [0.0070,0.0082,0.0090,0.0100,0.0120,0.0132,0.0160,0.0184,0.0244,0.0340,0.0500,0.0960],
 [0.0064,0.0078,0.0084,0.0092,0.0113,0.0121,0.0148,0.0170,0.0222,0.0307,0.0433,0.0745],
])


def load_xt_grid():
    if os.path.exists(WYSCOUT_GRID):
        try:
            with open(WYSCOUT_GRID, "r", encoding="utf-8") as f:
                grid = np.asarray(json.load(f), dtype=float)
            if grid.ndim == 2 and grid.shape[0] >= 4 and grid.shape[1] >= 4 and np.isfinite(grid).all():
                return grid, "Wyscout public 2017/18 learned grid"
        except Exception as exc:
            print("WARN Wyscout grid", type(exc).__name__, exc)
    return XT_FALLBACK, "Legacy fixed xThreat grid"


XT, XT_SOURCE = load_xt_grid()


def norm(x):
    s = unicodedata.normalize("NFKD", str(x or "")).encode("ascii", "ignore").decode("utf-8")
    return "".join(ch.lower() for ch in s if ch.isalnum())


def get_json(url, timeout=30):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "fut-europa-statsbomb/3.0"})
    if r.status_code != 200: return None
    try: return r.json()
    except Exception: return None


def target_league(name):
    low = str(name or "").lower().strip()
    for canon, aliases in TARGET_NAMES.items():
        if any(a in low for a in aliases): return canon
    return None


def xt_value(loc):
    """Mapea StatsBomb 120x80 al grid aprendido, sin asumir dimensiones fijas."""
    if not isinstance(loc, (list, tuple)) or len(loc) < 2: return np.nan
    try:
        x, y = float(loc[0]), float(loc[1])
        rows, cols = XT.shape
        cx = int(np.clip(x / 120.0 * cols, 0, cols - 1))
        cy = int(np.clip(y / 80.0 * rows, 0, rows - 1))
        return float(XT[cy, cx])
    except Exception: return np.nan


def discover_open_seasons():
    comps = get_json(f"{RAW}/competitions.json") or []
    rows=[]
    for c in comps:
        league=target_league(c.get("competition_name"))
        if league:
            rows.append({"Liga":league,"competition_id":c.get("competition_id"),"season_id":c.get("season_id"),"season_name":c.get("season_name")})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def load_matches(comp_id, season_id):
    return get_json(f"{RAW}/matches/{int(comp_id)}/{int(season_id)}.json") or []


def process_match(match):
    mid=match.get("match_id")
    if mid is None: return None, [], []
    date=pd.to_datetime(match.get("match_date"),errors="coerce")
    home=match.get("home_team",{}).get("home_team_name"); away=match.get("away_team",{}).get("away_team_name")
    if pd.isna(date) or date<MIN_DATE or not home or not away: return None, [], []
    events=get_json(f"{RAW}/events/{int(mid)}.json") or []
    if not events: return None, [], []
    lineups=get_json(f"{RAW}/lineups/{int(mid)}.json") or []
    teams=[home,away]; stats={norm(t):{"team":t,"xg":0.,"shots":0,"sot":0,"passes":0,"pressures":0,"poss":set(),"def_actions":0,"opp_passes_high":0,"xt":0.} for t in teams}
    event_rows=[]
    for e in events:
        team=e.get("team",{}).get("name"); tk=norm(team); typ=e.get("type",{}).get("name")
        if tk not in stats: continue
        loc=e.get("location"); end=None; xg=np.nan; successful=True
        if typ=="Pass":
            end=(e.get("pass") or {}).get("end_location"); successful=(e.get("pass") or {}).get("outcome") is None
            stats[tk]["passes"] += 1
        elif typ=="Carry": end=(e.get("carry") or {}).get("end_location")
        elif typ=="Shot":
            shot=e.get("shot") or {}; end=shot.get("end_location"); xg=float(shot.get("statsbomb_xg",0.) or 0.)
            stats[tk]["xg"]+=xg; stats[tk]["shots"]+=1
            if str((shot.get("outcome") or {}).get("name","")) in {"Goal","Saved","Saved to Post"}: stats[tk]["sot"]+=1
        elif typ=="Pressure": stats[tk]["pressures"]+=1
        if typ in {"Pressure","Duel","Interception","Block","Foul Committed","Ball Recovery","Clearance"}:
            try:
                if loc and float(loc[0])>=48: stats[tk]["def_actions"]+=1
            except Exception: pass
        poss=e.get("possession")
        if poss is not None: stats[tk]["poss"].add(poss)
        xt_move=np.nan
        if typ in {"Pass","Carry"} and successful:
            v0,v1=xt_value(loc),xt_value(end)
            if np.isfinite(v0) and np.isfinite(v1):
                xt_move=max(0.0, v1-v0)
                stats[tk]["xt"] += xt_move
        event_rows.append({"match_id":int(mid),"Fecha":date.strftime("%Y-%m-%d"),"team":team,"type":typ,"minute":e.get("minute"),"player":(e.get("player") or {}).get("name"),"x":loc[0] if loc else np.nan,"y":loc[1] if loc else np.nan,"end_x":end[0] if end else np.nan,"end_y":end[1] if end and len(end)>1 else np.nan,"xg":xg,"xT_transfer":xt_move,"possession":poss})
    for e in events:
        if e.get("type",{}).get("name")!="Pass": continue
        team=norm(e.get("team",{}).get("name")); loc=e.get("location")
        try: high=loc and float(loc[0])<=72
        except Exception: high=False
        if high:
            for other in stats:
                if other!=team: stats[other]["opp_passes_high"]+=1
    lineup_rows=[]
    for l in lineups:
        team=l.get("team_name")
        for p in l.get("lineup",[]) or []:
            lineup_rows.append({"match_id":int(mid),"Fecha":date.strftime("%Y-%m-%d"),"team":team,"player_id":p.get("player_id"),"player_name":p.get("player_name"),"country":(p.get("country") or {}).get("name") if isinstance(p.get("country"),dict) else p.get("country")})
    h,a=stats[norm(home)],stats[norm(away)]
    def pp(s): return float(s["opp_passes_high"])/max(float(s["def_actions"]),1.0)
    row={"Fecha":date.strftime("%Y-%m-%d"),"Local_SB":home,"Visitante_SB":away,"Local_norm":norm(home),"Visitante_norm":norm(away),"StatsBomb_match_id":int(mid),
         "xG_Real_Local":round(h["xg"],4),"xG_Real_Visita":round(a["xg"],4),"Tiros_SB_Local":h["shots"],"Tiros_SB_Visita":a["shots"],"TirosGol_SB_Local":h["sot"],"TirosGol_SB_Visita":a["sot"],
         "Pases_SB_Local":h["passes"],"Pases_SB_Visita":a["passes"],"Presiones_SB_Local":h["pressures"],"Presiones_SB_Visita":a["pressures"],"Posesiones_SB_Local":len(h["poss"]),"Posesiones_SB_Visita":len(a["poss"]),
         "PPDA_Local":round(pp(h),3),"PPDA_Visita":round(pp(a),3),"xThreat_Local":round(h["xt"],6),"xThreat_Visita":round(a["xt"],6),"xThreat_Source":XT_SOURCE,"Fuente_xG_Real":"StatsBomb Open Data"}
    return row,event_rows,lineup_rows


def main():
    print("xThreat source:", XT_SOURCE, "grid_shape=", XT.shape)
    coverage=discover_open_seasons()
    if coverage.empty: print("StatsBomb: sin temporadas objetivo"); return
    matches=[]
    for _,c in coverage.iterrows():
        for m in load_matches(c["competition_id"],c["season_id"]):
            dt=pd.to_datetime(m.get("match_date"),errors="coerce")
            if pd.notna(dt) and dt>=MIN_DATE: matches.append(m)
    unique={int(m["match_id"]):m for m in matches if m.get("match_id") is not None}
    print(f"StatsBomb: {len(unique)} partidos abiertos desde {MIN_DATE.date()}")
    rows=[]; evs=[]; lineups=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs={ex.submit(process_match,m):mid for mid,m in unique.items()}
        for i,fut in enumerate(as_completed(futs),1):
            try:
                row,e,l=fut.result()
                if row: rows.append(row); evs.extend(e); lineups.extend(l)
            except Exception as exc: print("WARN",futs[fut],type(exc).__name__,exc)
            if i%100==0: print(f"StatsBomb procesados {i}/{len(futs)}"); time.sleep(.05)
    if not rows: print("StatsBomb: sin eventos compatibles"); return
    os.makedirs("data",exist_ok=True)
    out=pd.DataFrame(rows).drop_duplicates(subset=["Fecha","Local_norm","Visitante_norm"]).sort_values(["Fecha","Local_norm","Visitante_norm"])
    out.to_csv(OUT,index=False); out.to_parquet(FEATURES_OUT,index=False,compression="zstd")
    pd.DataFrame(evs).to_parquet(EVENTS_OUT,index=False,compression="zstd")
    pd.DataFrame(lineups).drop_duplicates().to_parquet(LINEUPS_OUT,index=False,compression="zstd")
    print(f"STATSBOMB_ADVANCED_OK matches={len(out)} events={len(evs)} lineups={len(lineups)} xt_source={XT_SOURCE}")

if __name__=="__main__": main()
