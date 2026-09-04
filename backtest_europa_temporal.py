"""Walk-forward temporal para las cinco ligas.

Cada bloque se predice con un modelo entrenado sólo con partidos anteriores.
Compara modelo puro, mercado de apertura, mercado de cierre (benchmark) y
probabilidad final calibrada con APERTURA. El cierre nunca se usa para construir
una predicción retrospectiva temprana: sólo mide cuánta información incorporó
el mercado hasta el kickoff.

Modo normal: walk-forward completo (3 bloques por liga).
Modo CI (FUT_BACKTEST_FAST=1): una ventana temporal reciente por cada liga,
con hasta 120 partidos de prueba. Conserva la separación estricta train/test,
pero evita reentrenamientos redundantes en GitHub Actions.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from modules.elo_europa import SistemaEloEuropa
from modules.ml_europa import PredictorMLEuropa

FILES = {
    "Premier": "data/historico_premier.csv",
    "LaLiga": "data/historico_laliga.csv",
    "SerieA": "data/historico_seriea.csv",
    "Bundesliga": "data/historico_bundesliga.csv",
    "Ligue1": "data/historico_ligue1.csv",
}

FAST_MODE = os.getenv("FUT_BACKTEST_FAST", "0").strip().lower() in {"1", "true", "yes", "on"}
FAST_TEST_ROWS = max(80, int(os.getenv("FUT_BACKTEST_FAST_ROWS", "120")))


def rating(tabla, team):
    try:
        return float(tabla.loc[tabla["Equipo"] == team, "ELO_Rating"].iloc[0])
    except Exception:
        return 1500.0


def no_vig_from_cols(r, hcol, dcol, acol):
    try:
        odds = np.array([float(r[acol]), float(r[dcol]), float(r[hcol])], dtype=float)
        if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
            return None
        q = 1.0 / odds
        return q / q.sum()
    except Exception:
        return None


def market_open_vec(r):
    if {"Apertura_1", "Apertura_X", "Apertura_2"}.issubset(r.index):
        v = no_vig_from_cols(r, "Apertura_1", "Apertura_X", "Apertura_2")
        if v is not None:
            return v
    return no_vig_from_cols(r, "Cuota_1", "Cuota_X", "Cuota_2")


def market_close_vec(r):
    if {"Cierre_1", "Cierre_X", "Cierre_2"}.issubset(r.index):
        return no_vig_from_cols(r, "Cierre_1", "Cierre_X", "Cierre_2")
    return None


def opening_odds_dict(r):
    if {"Apertura_1", "Apertura_X", "Apertura_2"}.issubset(r.index):
        return {"1": r.get("Apertura_1"), "X": r.get("Apertura_X"), "2": r.get("Apertura_2")}
    return {"1": r.get("Cuota_1"), "X": r.get("Cuota_X"), "2": r.get("Cuota_2")}


def pred_vec(ml, loc, vis, el, ev, fecha, odds=None):
    p = ml.predecir_mercados_completos(
        loc, vis, elo_local=el, elo_visita=ev,
        cuotas_1x2=odds, fecha_partido=fecha,
    )["Resultado_1X2"]
    v = np.array([p["Gana Visita"], p["Empate"], p["Gana Local"]], dtype=float) / 100.0
    return v / v.sum()


def temporal_windows(df, n_blocks=3, fast=False):
    """Devuelve ventanas (train_end, test_end) sin fuga temporal."""
    if fast:
        # Una ventana reciente por liga. Se deja al menos 400 partidos para train.
        test_rows = min(FAST_TEST_ROWS, max(80, len(df) - 400))
        bstart = max(400, len(df) - test_rows)
        if len(df) - bstart < 20:
            return []
        return [(bstart, len(df))]

    start = max(400, int(len(df) * 0.70))
    remaining = len(df) - start
    block = max(80, int(np.ceil(remaining / n_blocks)))
    windows = []
    for bstart in range(start, len(df), block):
        bend = min(len(df), bstart + block)
        if bend - bstart >= 20:
            windows.append((bstart, bend))
    return windows


def evaluate(path, n_blocks=3, fast=False):
    df = pd.read_csv(path)
    df["_fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", format="%Y-%m-%d")
    df = df.dropna(subset=["_fecha"]).sort_values("_fecha", kind="stable").reset_index(drop=True)

    y, pure, final = [], [], []
    opening, opening_y = [], []
    closing, closing_y = [], []
    move_abs_pp = []
    learned = []

    windows = temporal_windows(df, n_blocks=n_blocks, fast=fast)
    assert windows, (path, "sin ventanas temporales")

    for bstart, bend in windows:
        train = df.iloc[:bstart].copy()
        test = df.iloc[bstart:bend].copy()

        ml = PredictorMLEuropa()
        assert ml.entrenar(train), path
        tabla = SistemaEloEuropa().actualizar_ratings(train)
        learned.append((ml.market_model_weight, ml.temp_1x2))

        for _, r in test.iterrows():
            loc, vis = str(r["Local"]), str(r["Visitante"])
            el, ev = rating(tabla, loc), rating(tabla, vis)
            gl, gv = float(r["Goles_Local"]), float(r["Goles_Visita"])
            target = 2 if gl > gv else (1 if gl == gv else 0)
            fecha = r.get("Fecha")

            pv = pred_vec(ml, loc, vis, el, ev, fecha, None)
            fv = pred_vec(ml, loc, vis, el, ev, fecha, opening_odds_dict(r))
            if not np.all(np.isfinite(pv)) or not np.all(np.isfinite(fv)):
                continue

            y.append(target)
            pure.append(pv)
            final.append(fv)

            ov = market_open_vec(r)
            cv = market_close_vec(r)
            if ov is not None:
                opening.append(ov)
                opening_y.append(target)
            if cv is not None:
                closing.append(cv)
                closing_y.append(target)
            if ov is not None and cv is not None:
                move_abs_pp.append(float(np.mean(np.abs(cv - ov)) * 100.0))

    y = np.asarray(y, dtype=int)
    pure = np.asarray(pure)
    final = np.asarray(final)
    min_required = 60 if fast else 100
    assert len(y) >= min_required, (path, len(y))
    onehot = np.eye(3)[y]

    out = {
        "n": int(len(y)),
        "accuracy_pure": float(accuracy_score(y, pure.argmax(axis=1))),
        "accuracy_final": float(accuracy_score(y, final.argmax(axis=1))),
        "logloss_pure": float(log_loss(y, pure, labels=[0, 1, 2])),
        "logloss_final": float(log_loss(y, final, labels=[0, 1, 2])),
        "brier_final": float(np.mean(np.sum((final - onehot) ** 2, axis=1))),
        "mean_model_weight": float(np.mean([x[0] for x in learned])),
        "mean_temperature": float(np.mean([x[1] for x in learned])),
        "blocks": len(learned),
    }

    if len(opening) >= min_required:
        out["opening_n"] = len(opening)
        out["logloss_opening"] = float(log_loss(np.asarray(opening_y), np.asarray(opening), labels=[0, 1, 2]))
        out["accuracy_opening"] = float(accuracy_score(np.asarray(opening_y), np.asarray(opening).argmax(axis=1)))
    if len(closing) >= min_required:
        out["closing_n"] = len(closing)
        out["logloss_closing"] = float(log_loss(np.asarray(closing_y), np.asarray(closing), labels=[0, 1, 2]))
        out["accuracy_closing"] = float(accuracy_score(np.asarray(closing_y), np.asarray(closing).argmax(axis=1)))
    if move_abs_pp:
        out["mean_market_move_abs_pp"] = float(np.mean(move_abs_pp))

    return out


def main():
    mode = "CI_FAST" if FAST_MODE else "FULL"
    print(f"BACKTEST_MODE={mode} fast_rows={FAST_TEST_ROWS if FAST_MODE else 'n/a'}")

    out = {name: evaluate(path, fast=FAST_MODE) for name, path in FILES.items()}
    for name, m in out.items():
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

    total = sum(v["n"] for v in out.values())
    weights = [v["n"] for v in out.values()]
    pure_ll = float(np.average([v["logloss_pure"] for v in out.values()], weights=weights))
    final_ll = float(np.average([v["logloss_final"] for v in out.values()], weights=weights))
    pure_acc = float(np.average([v["accuracy_pure"] for v in out.values()], weights=weights))
    final_acc = float(np.average([v["accuracy_final"] for v in out.values()], weights=weights))

    assert total >= (300 if FAST_MODE else 1000), total
    assert np.isfinite(final_ll) and final_ll < 1.099
    # En CI la muestra es más pequeña: permitimos una tolerancia ligeramente mayor
    # sin eliminar la prueba de que incorporar mercado no degrade severamente el modelo.
    tolerance = 0.02 if FAST_MODE else 0.008
    assert final_ll <= pure_ll + tolerance, (pure_ll, final_ll, tolerance)

    summary = {
        "mode": mode,
        "n": total,
        "pure_logloss": round(pure_ll, 4),
        "final_logloss": round(final_ll, 4),
        "pure_accuracy": round(pure_acc, 4),
        "final_accuracy": round(final_acc, 4),
    }

    open_rows = [v for v in out.values() if "logloss_opening" in v]
    if open_rows:
        ow = [v["opening_n"] for v in open_rows]
        summary["opening_logloss"] = round(float(np.average([v["logloss_opening"] for v in open_rows], weights=ow)), 4)
    close_rows = [v for v in out.values() if "logloss_closing" in v]
    if close_rows:
        cw = [v["closing_n"] for v in close_rows]
        summary["closing_logloss"] = round(float(np.average([v["logloss_closing"] for v in close_rows], weights=cw)), 4)

    print("WALK_FORWARD_OK", summary)


if __name__ == "__main__":
    main()
