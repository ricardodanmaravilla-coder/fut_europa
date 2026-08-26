"""Holdout temporal estricto para las cinco ligas.

Entrena con el 80% cronológicamente anterior y evalúa el 20% posterior.
Compara modelo puro, mercado no-vig y probabilidad final (blend aprendido sólo
con datos de entrenamiento). Las cuotas usadas son prepartido.
"""
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


def rating(tabla, team):
    try:
        return float(tabla.loc[tabla["Equipo"] == team, "ELO_Rating"].iloc[0])
    except Exception:
        return 1500.0


def market_vec(r):
    try:
        odds = np.array([float(r["Cuota_2"]), float(r["Cuota_X"]), float(r["Cuota_1"])])
        if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0): return None
        q = 1.0 / odds
        return q / q.sum()
    except Exception:
        return None


def pred_vec(ml, loc, vis, el, ev, odds=None):
    p = ml.predecir_mercados_completos(loc, vis, elo_local=el, elo_visita=ev, cuotas_1x2=odds)["Resultado_1X2"]
    v = np.array([p["Gana Visita"], p["Empate"], p["Gana Local"]], dtype=float) / 100.0
    return v / v.sum()


def evaluate(path):
    df = pd.read_csv(path)
    df["_fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", format="%Y-%m-%d")
    df = df.sort_values("_fecha", kind="stable").reset_index(drop=True)
    cut = max(240, int(len(df) * 0.80))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    ml = PredictorMLEuropa(); assert ml.entrenar(train), path
    tabla = SistemaEloEuropa().actualizar_ratings(train)

    y, pure, final, market = [], [], [], []
    market_y = []
    for _, r in test.iterrows():
        loc, vis = str(r["Local"]), str(r["Visitante"])
        el, ev = rating(tabla, loc), rating(tabla, vis)
        g_l, g_v = float(r["Goles_Local"]), float(r["Goles_Visita"])
        target = 2 if g_l > g_v else (1 if g_l == g_v else 0)
        pv = pred_vec(ml, loc, vis, el, ev, odds=None)
        odds = {"1": r.get("Cuota_1"), "X": r.get("Cuota_X"), "2": r.get("Cuota_2")}
        fv = pred_vec(ml, loc, vis, el, ev, odds=odds)
        if not np.all(np.isfinite(pv)) or not np.all(np.isfinite(fv)): continue
        y.append(target); pure.append(pv); final.append(fv)
        mv = market_vec(r)
        if mv is not None:
            market.append(mv); market_y.append(target)

    y = np.asarray(y, dtype=int)
    pure, final = np.asarray(pure), np.asarray(final)
    assert len(y) >= 50
    onehot = np.eye(3)[y]
    result = {
        "n": int(len(y)),
        "accuracy_pure": float(accuracy_score(y, pure.argmax(axis=1))),
        "accuracy_final": float(accuracy_score(y, final.argmax(axis=1))),
        "logloss_pure": float(log_loss(y, pure, labels=[0,1,2])),
        "logloss_final": float(log_loss(y, final, labels=[0,1,2])),
        "brier_final": float(np.mean(np.sum((final - onehot) ** 2, axis=1))),
        "model_weight": float(ml.market_model_weight),
        "temperature": float(ml.temp_1x2),
    }
    if len(market) >= 40:
        result["market_n"] = len(market)
        result["logloss_market"] = float(log_loss(np.asarray(market_y), np.asarray(market), labels=[0,1,2]))
    return result


def main():
    out = {name: evaluate(path) for name, path in FILES.items()}
    for name, m in out.items():
        print(name, {k: round(v,4) if isinstance(v,float) else v for k,v in m.items()})
    total = sum(v["n"] for v in out.values())
    weights = [v["n"] for v in out.values()]
    pure_ll = float(np.average([v["logloss_pure"] for v in out.values()], weights=weights))
    final_ll = float(np.average([v["logloss_final"] for v in out.values()], weights=weights))
    pure_acc = float(np.average([v["accuracy_pure"] for v in out.values()], weights=weights))
    final_acc = float(np.average([v["accuracy_final"] for v in out.values()], weights=weights))
    assert total >= 500
    assert np.isfinite(final_ll) and final_ll < 1.099
    # El calibrador/mercado no puede degradar materialmente al modelo puro agregado.
    assert final_ll <= pure_ll + 0.008, (pure_ll, final_ll)
    print("TEMPORAL_HOLDOUT_OK", {
        "n": total, "pure_logloss": round(pure_ll,4), "final_logloss": round(final_ll,4),
        "pure_accuracy": round(pure_acc,4), "final_accuracy": round(final_acc,4),
    })


if __name__ == "__main__":
    main()
