"""Cloud Run entrypoint enforcing the current-week fixture window.

Every consumer sees only fixtures from the current Mexico City date through the
nearest Sunday (inclusive). API-Football is queried by explicit date range so
we do not depend on the historical ``next=15`` cap. The legacy getter remains
only as a fallback and its output is filtered to the same weekly window.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import web_app as core

_original_get_fixtures = core.obtener_proximos_partidos_europa
_MEXICO_TZ = ZoneInfo("America/Mexico_City")
_fixture_cache: dict[tuple[int, date, date], tuple[float, dict]] = {}


def _week_window(today: date | None = None) -> tuple[date, date]:
    start = today or datetime.now(_MEXICO_TZ).date()
    # Python weekday: Monday=0 ... Sunday=6. If today is Sunday, end=today.
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
    start, end = _week_window()

    # Prefer the explicit date-range query: it returns every fixture in the
    # weekly window, even if a league has more than 15 upcoming matches.
    partidos = _api_sports_week(league_id, start, end)
    if partidos:
        return partidos

    # Resilience fallback for API errors / missing key. Never expose fixtures
    # outside the same Mexico City weekly window.
    return _filter_week(_original_get_fixtures(league_id), start, end)


core.obtener_proximos_partidos_europa = obtener_partidos_semana

# web_app_v2 imports the same cached web_app module object, so the individual
# selector and global scanner both use this patched weekly retrieval.
from web_app_v2 import app  # noqa: E402,F401
