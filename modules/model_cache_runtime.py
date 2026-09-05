"""Runtime loader for prebuilt FUT Europa model bundles."""
from __future__ import annotations

from pathlib import Path
from threading import RLock
import re

import joblib

CACHE_DIR = Path(__file__).resolve().parent.parent / "model_cache"
_LOCK = RLock()
_MEM = {}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _version(df) -> str:
    try:
        return f"{len(df)}:{str(df.iloc[-1].get('Fecha', ''))}"
    except Exception:
        return str(len(df))


def install_prebuilt_model_cache(core):
    """Patch core.obtener_motores to prefer build-time joblib bundles.

    Falls back to the original cached trainer if a bundle is unavailable or does
    not match the current historical data version.
    """
    original = core.obtener_motores

    def obtener_motores_prebuilt(nombre_liga, df):
        version = _version(df)
        mem_key = (nombre_liga, version)
        with _LOCK:
            if mem_key in _MEM:
                return _MEM[mem_key]

        path = CACHE_DIR / f"{_slug(nombre_liga)}.joblib"
        if path.exists():
            try:
                bundle = joblib.load(path)
                if bundle.get("league") == nombre_liga and bundle.get("data_version") == version and bundle.get("ml_ok"):
                    result = (bundle["tabla"], bundle["ml"], True)
                    with _LOCK:
                        _MEM[mem_key] = result
                    return result
            except Exception:
                pass

        # Safety fallback: preserve service availability if the image cache is stale.
        result = original(nombre_liga, df)
        with _LOCK:
            _MEM[mem_key] = result
        return result

    core.obtener_motores = obtener_motores_prebuilt
    core.MODEL_CACHE_PREBUILT = True
    core.MODEL_CACHE_DIR = str(CACHE_DIR)
    return obtener_motores_prebuilt
