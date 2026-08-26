"""Diagnóstico temporal reproducible para las cinco ligas.

No optimiza parámetros ni usa cuotas. Entrena con el 80% cronológicamente
anterior de cada liga y evalúa el 20% posterior, evitando mezclas aleatorias.
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


def evaluate(path):
    df = pd.read_csv(path)
    df["_fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    df = df.sort_values("_fecha", kind="stable").reset_index(drop=True)
    cut = max(120, int(len(df) * 0.80))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    ml = PredictorMLEuropa()
    assert ml.entrenar(train), path
    tabla = SistemaEloEuropa().actualizar_ratings(train)

    y, probs = [], []
    for _, r in test.iterrows():
        loc, vis = str(r["Local"]), str(r["Visitante"])
        p = ml.predecir_mercados_completos(
            loc, vis, elo_local=rating(tabla, loc), elo_visita=rating(tabla, vis)
        )["Resultado_1X2"]
        vec = np.array([p["Gana Visita"], p["Empate"], p["Gana Local"]], dtype=float) / 100.0
        s = vec.sum()
        if not np.isfinite(s) or s <= 0:
            continue
        vec /= s
        g_l, g_v = float(r["Goles_Local"]), float(r["Goles_Visita"])
        y.append(2 if g_l > g_v else (1 if g_l == g_v else 0))
        probs.append(vec)

    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    assert len(y) >= 50, (path, len(y))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    pred = probs.argmax(axis=1)
    onehot = np.eye(3)[y]
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "logloss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier_multiclass": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
    }


def main():
    out = {name: evaluate(path) for name, path in FILES.items()}
    for name, m in out.items():
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    total = sum(v["n"] for v in out.values())
    mean_ll = float(np.average([v["logloss"] for v in out.values()], weights=[v["n"] for v in out.values()]))
    assert total >= 500
    assert np.isfinite(mean_ll)
    print("TEMPORAL_HOLDOUT_OK", {"n": total, "weighted_logloss": round(mean_ll, 4)})


if __name__ == "__main__":
    main()
