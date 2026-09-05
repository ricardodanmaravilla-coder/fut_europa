"""Cloud Run entrypoint for the short-horizon FUT Europa scanner.

The scanner now exposes only matches from today and tomorrow in Mexico City,
and only while they have not started. This keeps finished/live matches out of
ML/MC work and materially reduces scanner load. API-Football is the source of
truth for kickoff/status; the legacy getter is only a tomorrow-only fallback
because it does not reliably preserve kickoff time/status.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import time

import requests
import web_app as core
from modules.model_cache_runtime import install_prebuilt_model_cache

_original_get_fixtures = core.obtener_proximos_partidos_europa
_MEXICO_TZ = ZoneInfo("America/Mexico_City")
_fixture_cache: dict[tuple[int, date, date], tuple[float, dict]] = {}
_FIXTURE_CACHE_TTL_SECONDS = 300.0
_ALLOWED_PREMATCH_STATUSES = {"NS", "TBD"}


def _week_window(today: date | None = None) -> tuple[date, date]:
    """Backward-compatible helper retained for existing regression tests."""
    start = today or datetime.now(_MEXICO_TZ).date()
    end = start + timedelta(days=(6 - start.weekday()))
    return start, end


def _scan_window(today: date | None = None) -> tuple[date, date]:
    start = today or datetime.now(_MEXICO_TZ).date()
    return start, start + timedelta(days=1)


def _parse_kickoff(raw) -> datetime | None:
    try:
        text = str(raw or "").strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_MEXICO_TZ)
        return dt.astimezone(_MEXICO_TZ)
    except Exception:
        return None


def _still_upcoming(fx: dict, now_local: datetime | None = None) -> bool:
    """True only for a scheduled fixture whose kickoff is still in the future."""
    now_local = now_local or datetime.now(_MEXICO_TZ)
    status = str(fx.get("status_short", "")).upper().strip()
    if status and status not in _ALLOWED_PREMATCH_STATUSES:
        return False
    kickoff = _parse_kickoff(fx.get("kickoff"))
    if kickoff is None:
        return False
    return kickoff > now_local


def _filter_scan_window(fixtures: dict, start: date, end: date, now_local: datetime | None = None) -> dict:
    now_local = now_local or datetime.now(_MEXICO_TZ)
    filtered = {}
    for key, fx in (fixtures or {}).items():
        raw = str(fx.get("fecha", ""))[:10]
        try:
            match_date = date.fromisoformat(raw)
        except (TypeError, ValueError):
            continue
        if start <= match_date <= end and _still_upcoming(fx, now_local):
            filtered[key] = fx
    return filtered


def _api_sports_week(league_id: int, start: date, end: date) -> dict:
    """Fetch the requested date range and discard live/finished/started games."""
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
            fixture = item.get("fixture", {}) or {}
            local = item.get("teams", {}).get("home", {}).get("name")
            visita = item.get("teams", {}).get("away", {}).get("name")
            kickoff_raw = fixture.get("date")
            kickoff = _parse_kickoff(kickoff_raw)
            fecha = kickoff.date().isoformat() if kickoff else str(kickoff_raw or "")[:10]
            fixture_id = fixture.get("id")
            status_short = str((fixture.get("status", {}) or {}).get("short", "")).upper().strip()
            if local and visita and fecha:
                key = str(fixture_id or f"{fecha}-{local}-{visita}")
                partidos[key] = {
                    "local": local,
                    "visita": visita,
                    "fixture_id": fixture_id,
                    "fecha": fecha,
                    "kickoff": str(kickoff_raw or ""),
                    "status_short": status_short,
                }
        return _filter_scan_window(partidos, start, end)
    except Exception:
        return {}


def _fallback_tomorrow_only(league_id: int, tomorrow: date) -> dict:
    """Legacy fallback: only tomorrow is safe because kickoff/status may be absent."""
    legacy = _original_get_fixtures(league_id)
    safe = {}
    for key, fx in (legacy or {}).items():
        try:
            if date.fromisoformat(str(fx.get("fecha", ""))[:10]) != tomorrow:
                continue
        except Exception:
            continue
        copied = dict(fx)
        copied["kickoff"] = f"{tomorrow.isoformat()}T23:59:59-06:00"
        copied["status_short"] = "NS"
        safe[key] = copied
    return safe


def obtener_partidos_semana(league_id: int):
    """Return a stable snapshot of only today/tomorrow games that have not started."""
    start, end = _scan_window()
    cache_key = (league_id, start, end)
    now_mono = time.monotonic()
    now_local = datetime.now(_MEXICO_TZ)
    cached = _fixture_cache.get(cache_key)

    if cached and now_mono - cached[0] < _FIXTURE_CACHE_TTL_SECONDS:
        fresh_cached = _filter_scan_window(cached[1], start, end, now_local)
        _fixture_cache[cache_key] = (cached[0], fresh_cached)
        return fresh_cached

    partidos = _api_sports_week(league_id, start, end)
    if partidos:
        _fixture_cache[cache_key] = (now_mono, partidos)
        return partidos

    fallback = _fallback_tomorrow_only(league_id, end)
    if fallback:
        _fixture_cache[cache_key] = (now_mono, fallback)
        return fallback

    if cached:
        return _filter_scan_window(cached[1], start, end, now_local)
    return {}


core.obtener_proximos_partidos_europa = obtener_partidos_semana
install_prebuilt_model_cache(core)

# web_app_v2 imports the same cached web_app module object, so the individual
# selector and global scanner use both the filtered fixtures and prebuilt models.
from web_app_v2 import app  # noqa: E402,F401
