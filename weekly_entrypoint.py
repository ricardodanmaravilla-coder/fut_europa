"""Cloud Run entrypoint enforcing the current-week fixture window.

The production core historically asks API-Sports for the next 15 fixtures per
league.  This wrapper keeps that retrieval as a resilience buffer but exposes
only fixtures from today through the nearest Sunday (inclusive) to every
consumer of ``web_app.obtener_proximos_partidos_europa``.  Because web_app_v2
imports the same cached web_app module, the patch applies consistently to the
individual fixture selector and the global scanner.
"""
from __future__ import annotations

from datetime import date, timedelta

import web_app as core

_original_get_fixtures = core.obtener_proximos_partidos_europa


def _week_window(today: date | None = None) -> tuple[date, date]:
    start = today or date.today()
    # Python weekday: Monday=0 ... Sunday=6. If today is Sunday, end=today.
    end = start + timedelta(days=(6 - start.weekday()))
    return start, end


def obtener_partidos_semana(league_id: int):
    fixtures = _original_get_fixtures(league_id)
    start, end = _week_window()
    filtered = {}
    for key, fx in fixtures.items():
        raw = str(fx.get("fecha", ""))[:10]
        try:
            match_date = date.fromisoformat(raw)
        except (TypeError, ValueError):
            # Never leak an undated fixture into the scanner.
            continue
        if start <= match_date <= end:
            filtered[key] = fx
    return filtered


core.obtener_proximos_partidos_europa = obtener_partidos_semana

# web_app_v2 imports ``web_app as core``; Python reuses the already patched
# module object above, so all endpoints and the streaming scanner share the
# exact same weekly window.
from web_app_v2 import app  # noqa: E402,F401
