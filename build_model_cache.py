"""Train FUT Europa ML/Elo bundles at image build time.

This moves expensive fitting out of user requests. The resulting joblib files
are copied into the Cloud Run image and loaded on demand in milliseconds.
"""
from __future__ import annotations

from pathlib import Path
import re

import joblib

from modules.data_store import cargar_historico
from modules.elo_europa import SistemaEloEuropa
from modules.ml_europa import PredictorMLEuropa

ARCHIVOS = {
    "Premier League": "data/historico_premier.csv",
    "La Liga": "data/historico_laliga.csv",
    "Serie A": "data/historico_seriea.csv",
    "Bundesliga": "data/historico_bundesliga.csv",
    "Ligue 1": "data/historico_ligue1.csv",
}
CACHE_DIR = Path("model_cache")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def data_version(df) -> str:
    try:
        return f"{len(df)}:{str(df.iloc[-1].get('Fecha', ''))}"
    except Exception:
        return str(len(df))


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for liga, path in ARCHIVOS.items():
        print(f"[model-cache] training {liga} from {path}", flush=True)
        df = cargar_historico(path)
        if df is None or df.empty:
            failures.append(f"{liga}: histórico vacío")
            continue
        df = df.copy()
        df["Local"] = df["Local"].astype(str).str.strip()
        df["Visitante"] = df["Visitante"].astype(str).str.strip()
        tabla = SistemaEloEuropa().actualizar_ratings(df)
        ml = PredictorMLEuropa()
        ml_ok = bool(ml.entrenar(df))
        if not ml_ok:
            failures.append(f"{liga}: ML no entrenó")
            continue
        bundle = {
            "schema": 1,
            "league": liga,
            "data_version": data_version(df),
            "tabla": tabla,
            "ml": ml,
            "ml_ok": True,
        }
        out = CACHE_DIR / f"{slug(liga)}.joblib"
        joblib.dump(bundle, out, compress=3)
        print(f"[model-cache] ready {liga}: {out} rows={len(df)} train={getattr(ml, 'n_train', 0)}", flush=True)
    if failures:
        raise SystemExit("Model cache build failed: " + " | ".join(failures))


if __name__ == "__main__":
    main()
