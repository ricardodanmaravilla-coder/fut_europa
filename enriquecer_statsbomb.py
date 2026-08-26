"""Enriquecimiento gratuito y opcional con StatsBomb Open Data.

Descubre automáticamente competiciones/temporadas abiertas de las cinco ligas
objetivo, descarga partidos y eventos, y agrega xG real por partido a un cache
local. No fuerza cobertura: sólo escribe coincidencias que StatsBomb publica.

Fuente: https://github.com/statsbomb/open-data
Al compartir análisis derivados, atribuir los datos a StatsBomb según su licencia.
"""
import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = "data/statsbomb_xg_matches.csv"
TARGET_NAMES = {
    "Premier League": ["premier league"],
    "La Liga": ["la liga"],
    "Serie A": ["serie a"],
    "Bundesliga": ["bundesliga"],
    "Ligue 1": ["ligue 1"],
}
MIN_DATE = pd.Timestamp("2020-07-01")
MAX_WORKERS = 8


def norm(x):
    s = unicodedata.normalize("NFKD", str(x or "")).encode("ascii", "ignore").decode("utf-8")
    return "".join(ch.lower() for ch in s if ch.isalnum())


def get_json(url, timeout=25):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "fut-europa-statsbomb/1.0"})
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def target_league(name):
    low = str(name or "").lower().strip()
    for canon, aliases in TARGET_NAMES.items():
        if any(a in low for a in aliases):
            return canon
    return None


def discover_open_seasons():
    comps = get_json(f"{RAW}/competitions.json") or []
    rows = []
    for c in comps:
        league = target_league(c.get("competition_name"))
        if not league:
            continue
        rows.append({
            "Liga": league,
            "competition_id": c.get("competition_id"),
            "season_id": c.get("season_id"),
            "season_name": c.get("season_name"),
        })
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def load_matches(comp_id, season_id):
    return get_json(f"{RAW}/matches/{int(comp_id)}/{int(season_id)}.json") or []


def aggregate_event_match(match):
    mid = match.get("match_id")
    if mid is None:
        return None
    events = get_json(f"{RAW}/events/{int(mid)}.json")
    if not events:
        return None

    home = match.get("home_team", {}).get("home_team_name")
    away = match.get("away_team", {}).get("away_team_name")
    date = pd.to_datetime(match.get("match_date"), errors="coerce")
    if pd.isna(date) or date < MIN_DATE or not home or not away:
        return None

    agg = {
        norm(home): {"xg": 0.0, "shots": 0, "sot": 0},
        norm(away): {"xg": 0.0, "shots": 0, "sot": 0},
    }
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        team = norm(e.get("team", {}).get("name"))
        if team not in agg:
            continue
        shot = e.get("shot", {}) or {}
        try:
            xg = float(shot.get("statsbomb_xg", 0.0) or 0.0)
        except Exception:
            xg = 0.0
        agg[team]["xg"] += xg
        agg[team]["shots"] += 1
        outcome = str((shot.get("outcome") or {}).get("name", ""))
        if outcome in {"Goal", "Saved", "Saved to Post"}:
            agg[team]["sot"] += 1

    h = agg[norm(home)]
    a = agg[norm(away)]
    return {
        "Fecha": date.strftime("%Y-%m-%d"),
        "Local_SB": home,
        "Visitante_SB": away,
        "Local_norm": norm(home),
        "Visitante_norm": norm(away),
        "StatsBomb_match_id": int(mid),
        "xG_Real_Local": round(h["xg"], 4),
        "xG_Real_Visita": round(a["xg"], 4),
        "Tiros_SB_Local": h["shots"],
        "Tiros_SB_Visita": a["shots"],
        "TirosGol_SB_Local": h["sot"],
        "TirosGol_SB_Visita": a["sot"],
        "Fuente_xG_Real": "StatsBomb Open Data",
    }


def main():
    coverage = discover_open_seasons()
    if coverage.empty:
        print("StatsBomb: no se encontraron temporadas abiertas de ligas objetivo")
        return
    print("StatsBomb temporadas candidatas:")
    print(coverage.to_string(index=False))

    matches = []
    for _, c in coverage.iterrows():
        for m in load_matches(c["competition_id"], c["season_id"]):
            dt = pd.to_datetime(m.get("match_date"), errors="coerce")
            if pd.notna(dt) and dt >= MIN_DATE:
                matches.append(m)

    # Evita duplicados si la misma temporada aparece más de una vez en metadata.
    unique = {int(m["match_id"]): m for m in matches if m.get("match_id") is not None}
    print(f"StatsBomb: {len(unique)} partidos abiertos desde {MIN_DATE.date()}")

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(aggregate_event_match, m): mid for mid, m in unique.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
                if row:
                    rows.append(row)
            except Exception:
                pass
            if i % 100 == 0:
                print(f"StatsBomb eventos procesados: {i}/{len(futs)}")
                time.sleep(0.05)

    if not rows:
        print("StatsBomb: no hubo eventos compatibles para enriquecer")
        return

    out = pd.DataFrame(rows).drop_duplicates(subset=["Fecha", "Local_norm", "Visitante_norm"])
    out = out.sort_values(["Fecha", "Local_norm", "Visitante_norm"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"StatsBomb cache OK: {OUT} | {len(out)} partidos con xG real")


if __name__ == "__main__":
    main()
