"""Build an offline Wyscout 2017/18 event representation for the five big European leagues.

The public dataset is CC BY 4.0 and is downloaded by socceraction's
PublicWyscoutLoader.  This script intentionally keeps 2017/18 separated from the
modern prediction history: it learns reusable action-value representations and
writes aggregate team/game features, but does not append old matches to the
2021+ walk-forward dataset.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = {"England", "Spain", "Italy", "Germany", "France"}
ROOT = Path("data/wyscout_public")
OUT = Path("data")


def _league_name(row: pd.Series) -> str:
    text = " ".join(str(row.get(c, "")) for c in ["competition_name", "country_name", "name"])
    low = text.lower()
    if "england" in low or "premier league" in low: return "Premier League"
    if "spain" in low or "la liga" in low: return "La Liga"
    if "italy" in low or "serie a" in low: return "Serie A"
    if "germany" in low or "bundesliga" in low: return "Bundesliga"
    if "france" in low or "ligue 1" in low: return "Ligue 1"
    return ""


def _ppda(events: pd.DataFrame, team_id: int) -> float:
    """Event-based PPDA proxy on the opponent's first 60% of pitch.

    Wyscout coordinates are 0-100 from the acting team's attacking direction.
    We count opponent passes in their build-up zone and defensive actions by the
    evaluated team in that same broad pressing zone.  This is stored as a
    research feature, not injected into production until validated.
    """
    if events.empty: return np.nan
    ev = events.copy()
    def sx(v):
        try:
            if isinstance(v, list) and v: return float(v[0].get("x", np.nan))
        except Exception: pass
        return np.nan
    ev["sx"] = ev.get("positions", pd.Series([None]*len(ev))).map(sx)
    opp_passes = ev[(ev["team_id"] != team_id) & (ev["type_name"].str.lower().eq("pass")) & (ev["sx"] <= 60)]
    defensive = ev[(ev["team_id"] == team_id) & ev["type_name"].str.lower().isin({"duel","foul","interruption","others on the ball"}) & (ev["sx"] >= 40)]
    return float(len(opp_passes) / max(len(defensive), 1))


def main() -> None:
    from socceraction.data.wyscout import PublicWyscoutLoader
    import socceraction.spadl as spadl
    import socceraction.xthreat as xthreat
    from socceraction.vaep import VAEP

    OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    api = PublicWyscoutLoader(root=str(ROOT), download=not any(ROOT.iterdir()))
    comps = api.competitions().copy()
    comps["Liga"] = comps.apply(_league_name, axis=1)
    comps = comps[comps["Liga"].ne("")]
    if comps.empty:
        raise RuntimeError("No se encontraron las cinco ligas Wyscout abiertas")

    games_all = []
    actions_by_game: list[tuple[pd.Series, pd.DataFrame, pd.DataFrame]] = []
    events_rows = []

    for comp in comps.to_dict(orient="records"):
        cid = comp.get("competition_id")
        sid = comp.get("season_id")
        league = comp["Liga"]
        games = api.games(cid, sid).copy()
        if games.empty: continue
        games["Liga"] = league
        games_all.append(games)
        for _, game in games.iterrows():
            gid = int(game["game_id"])
            ev = api.events(gid)
            if ev.empty: continue
            actions = spadl.wyscout.convert_to_actions(ev, int(game["home_team_id"]))
            actions = spadl.play_left_to_right(actions, int(game["home_team_id"]))
            actions = spadl.add_names(actions)
            actions["Liga"] = league
            actions_by_game.append((game, actions, ev))
            # Keep a compact standardized event sample in Parquet.
            keep = [c for c in ["game_id","event_id","period_id","milliseconds","team_id","player_id","type_name","subtype_name","positions"] if c in ev.columns]
            mini = ev[keep].copy()
            mini["Liga"] = league
            events_rows.append(mini)

    if not actions_by_game:
        raise RuntimeError("Wyscout no produjo acciones SPADL")

    actions_all = pd.concat([a for _, a, _ in actions_by_game], ignore_index=True)
    actions_all.to_parquet(OUT / "wyscout_spadl_actions.parquet", index=False, compression="zstd")
    if events_rows:
        pd.concat(events_rows, ignore_index=True).to_parquet(OUT / "wyscout_events_compact.parquet", index=False, compression="zstd")

    # Learn xThreat purely on the 2017/18 event universe.
    xt_model = xthreat.ExpectedThreat(l=16, w=12)
    xt_model.fit(actions_all)
    move_actions = xthreat.get_successful_move_actions(actions_all).copy()
    move_actions["xT_value"] = xt_model.rate(move_actions)
    move_actions.to_parquet(OUT / "wyscout_xt_actions.parquet", index=False, compression="zstd")
    with open(OUT / "wyscout_xt_grid.json", "w", encoding="utf-8") as f:
        json.dump(np.asarray(xt_model.xT).tolist(), f)

    # VAEP is learned on Wyscout itself. We fit on chronologically earlier games
    # and rate later games to avoid learning and evaluating the same actions.
    game_ids = [int(g["game_id"]) for g, _, _ in actions_by_game]
    split = max(1, int(len(game_ids) * 0.8))
    train_ids = set(game_ids[:split])
    test_ids = set(game_ids[split:])
    vaep = VAEP()
    Xs, Ys = [], []
    for game, actions, _ in actions_by_game:
        if int(game["game_id"]) not in train_ids: continue
        try:
            Xs.append(vaep.compute_features(game, actions))
            Ys.append(vaep.compute_labels(game, actions))
        except Exception:
            continue
    vaep_fitted = False
    if Xs and Ys:
        Xtr = pd.concat(Xs, ignore_index=True)
        Ytr = pd.concat(Ys, ignore_index=True)
        try:
            vaep.fit(Xtr, Ytr)
            vaep_fitted = True
        except Exception:
            vaep_fitted = False

    rows = []
    for game, actions, ev in actions_by_game:
        gid = int(game["game_id"])
        home_id, away_id = int(game["home_team_id"]), int(game["away_team_id"])
        ma = move_actions[move_actions["game_id"] == gid]
        xt_home = float(ma.loc[ma["team_id"] == home_id, "xT_value"].sum()) if not ma.empty else 0.0
        xt_away = float(ma.loc[ma["team_id"] == away_id, "xT_value"].sum()) if not ma.empty else 0.0
        vaep_home = vaep_away = np.nan
        if vaep_fitted and gid in test_ids:
            try:
                Xg = vaep.compute_features(game, actions)
                rated = vaep.rate(game, actions, Xg)
                vals = rated if isinstance(rated, pd.DataFrame) else pd.DataFrame(rated)
                if "vaep_value" in vals.columns:
                    tmp = actions.reset_index(drop=True).join(vals[["vaep_value"]].reset_index(drop=True))
                    vaep_home = float(tmp.loc[tmp.team_id == home_id, "vaep_value"].sum())
                    vaep_away = float(tmp.loc[tmp.team_id == away_id, "vaep_value"].sum())
            except Exception:
                pass
        rows.append({
            "game_id": gid,
            "Liga": actions["Liga"].iloc[0],
            "home_team_id": home_id,
            "away_team_id": away_id,
            "xThreat_Local": xt_home,
            "xThreat_Visita": xt_away,
            "PPDA_Local": _ppda(ev, home_id),
            "PPDA_Visita": _ppda(ev, away_id),
            "VAEP_Local_OOS": vaep_home,
            "VAEP_Visita_OOS": vaep_away,
            "Acciones_SPADL": int(len(actions)),
        })

    features = pd.DataFrame(rows)
    features.to_parquet(OUT / "wyscout_match_features.parquet", index=False, compression="zstd")
    pd.concat(games_all, ignore_index=True).to_parquet(OUT / "wyscout_games.parquet", index=False, compression="zstd")
    print("WYSCOUT_ADVANCED_OK", {
        "games": len(features),
        "spadl_actions": len(actions_all),
        "xt_actions": len(move_actions),
        "vaep_oos_games": int(features[["VAEP_Local_OOS","VAEP_Visita_OOS"]].notna().all(axis=1).sum()),
        "leagues": sorted(features["Liga"].unique().tolist()),
    })


if __name__ == "__main__":
    main()
