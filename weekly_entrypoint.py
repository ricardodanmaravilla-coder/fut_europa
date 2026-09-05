"""Cloud Run entrypoint enforcing the current-week fixture window.

Every consumer sees only fixtures from the current Mexico City date through the
nearest Sunday (inclusive). API-Football is queried by explicit date range so
we do not depend on the historical ``next=15`` cap. The legacy getter remains
only as a fallback and its output is filtered to the same weekly window.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import time

import requests
import web_app as core

_original_get_fixtures = core.obtener_proximos_partidos_europa
_MEXICO_TZ = ZoneInfo("America/Mexico_City")
_fixture_cache: dict[tuple[int, date, date], tuple[float, dict]] = {}
_FIXTURE_CACHE_TTL_SECONDS = 300.0


def _week_window(today: date | None = None) -> tuple[date, date]:
    start = today or datetime.now(_MEXICO_TZ).date()
    end = start + timedelta(days=(6 - start.weekday()))
    return start, end


def _filter_week(fixtures: dict, start: date, end: date) -> dict:
    filtered = {}
    for key, fx in fixtures.items():
        raw = str(fx.get("fecha", ""))[:10]
        try:
            match_date = date.fromisoformat(raw)
        except (TypeError, ValueError):
            continue
        if start <= match_date <= end:
            filtered[key] = fx
    return filtered


def _api_sports_week(league_id: int, start: date, end: date) -> dict:
    if not core.API_KEY:
        return {}
    try:
        response = requests.get(
            f"{core.BASE_URL}/fixtures",
            headers=core.HEADERS,
            params={
                "league": league_id,
                "season": 2026,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "timezone": "America/Mexico_City",
            },
            timeout=10,
        )
        if response.status_code != 200:
            return {}
        partidos = {}
        for item in response.json().get("response", []):
            local = item.get("teams", {}).get("home", {}).get("name")
            visita = item.get("teams", {}).get("away", {}).get("name")
            fecha = str(item.get("fixture", {}).get("date", ""))[:10]
            fixture_id = item.get("fixture", {}).get("id")
            if local and visita and fecha:
                key = str(fixture_id or f"{fecha}-{local}-{visita}")
                partidos[key] = {
                    "local": local,
                    "visita": visita,
                    "fixture_id": fixture_id,
                    "fecha": fecha,
                }
        return _filter_week(partidos, start, end)
    except Exception:
        return {}


def obtener_partidos_semana(league_id: int):
    """Return a stable weekly fixture snapshot.

    The browser first loads /api/fixtures and later sends each fixture key to
    /api/scan-one. Re-querying API-Football for every match can hit rate limits,
    change the snapshot mid-scan, or temporarily return an empty response. Keep
    one exact per-league snapshot for five minutes and reuse a stale snapshot if
    the upstream API briefly fails, so fixture keys remain resolvable throughout
    a scan.
    """
    start, end = _week_window()
    cache_key = (league_id, start, end)
    now = time.monotonic()
    cached = _fixture_cache.get(cache_key)
    if cached and now - cached[0] < _FIXTURE_CACHE_TTL_SECONDS:
        return cached[1]

    partidos = _api_sports_week(league_id, start, end)
    if partidos:
        _fixture_cache[cache_key] = (now, partidos)
        return partidos

    fallback = _filter_week(_original_get_fixtures(league_id), start, end)
    if fallback:
        _fixture_cache[cache_key] = (now, fallback)
        return fallback

    # If upstream temporarily fails during a running scan, preserve the last
    # known snapshot instead of turning every queued fixture into a 404.
    if cached:
        return cached[1]
    return {}


core.obtener_proximos_partidos_europa = obtener_partidos_semana

# web_app_v2 imports the same cached web_app module object, so the individual
# selector and global scanner both use this patched weekly retrieval.
from web_app_v2 import app  # noqa: E402,F401
