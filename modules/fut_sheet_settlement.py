import os
import re
from datetime import datetime, timezone
from urllib.parse import quote

import requests

from modules.fut_sheet_ledger import HEADERS, SPREADSHEET_ID, WORKSHEET, _session

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
API_HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}
FINAL_STATUSES = {"FT", "AET", "PEN"}


def _values_url(range_a1):
    encoded = quote(range_a1, safe="!:$")
    return f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{encoded}"


def _num(value, default=0.0):
    try:
        return float(value)
    except Exception:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
        return float(match.group(0)) if match else default


def _fixture_result(fixture_id):
    if not API_KEY:
        raise RuntimeError("API_SPORTS_KEY missing")
    response = requests.get(
        f"{BASE_URL}/fixtures",
        headers=API_HEADERS,
        params={"id": str(fixture_id)},
        timeout=12,
    )
    response.raise_for_status()
    items = response.json().get("response", [])
    if not items:
        return None
    item = items[0]
    status = str(item.get("fixture", {}).get("status", {}).get("short", "")).upper()
    if status not in FINAL_STATUSES:
        return {"final": False, "status": status}

    score = item.get("score", {}) or {}
    fulltime = score.get("fulltime", {}) or {}
    goals = item.get("goals", {}) or {}
    home_goals = fulltime.get("home")
    away_goals = fulltime.get("away")
    if home_goals is None:
        home_goals = goals.get("home")
    if away_goals is None:
        away_goals = goals.get("away")
    if home_goals is None or away_goals is None:
        raise RuntimeError(f"Fixture {fixture_id} final without full-time score")
    return {
        "final": True,
        "status": status,
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
    }


def _fixture_statistics(fixture_id):
    response = requests.get(
        f"{BASE_URL}/fixtures/statistics",
        headers=API_HEADERS,
        params={"fixture": str(fixture_id)},
        timeout=12,
    )
    response.raise_for_status()
    teams = response.json().get("response", [])
    totals = {"corners": 0.0, "cards": 0.0}
    found = {"corners": False, "cards": False}
    for team in teams:
        for stat in team.get("statistics", []) or []:
            kind = str(stat.get("type", "")).strip().lower()
            value = stat.get("value")
            if value is None:
                continue
            if kind == "corner kicks":
                totals["corners"] += _num(value)
                found["corners"] = True
            elif kind in {"yellow cards", "red cards"}:
                totals["cards"] += _num(value)
                found["cards"] = True
    totals["corners"] = totals["corners"] if found["corners"] else None
    totals["cards"] = totals["cards"] if found["cards"] else None
    return totals


def _over_under(selection, observed):
    text = str(selection).lower()
    match = re.search(r"(over|under)\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        raise ValueError(f"Unsupported total selection: {selection}")
    side = match.group(1)
    line = float(match.group(2))
    if observed == line:
        return "push"
    if side == "over":
        return "win" if observed > line else "loss"
    return "win" if observed < line else "loss"


def _grade(selection, market, result, stats):
    text = str(selection).strip().lower()
    market_text = str(market).strip().lower()
    hg, ag = result["home_goals"], result["away_goals"]

    if "corner" in text or "corner" in market_text:
        if stats.get("corners") is None:
            raise RuntimeError("Corner statistics unavailable")
        observed = float(stats["corners"])
        return _over_under(selection, observed), f"corners={observed:g}"

    if "tarjeta" in text or "card" in text or "tarjeta" in market_text:
        if stats.get("cards") is None:
            raise RuntimeError("Card statistics unavailable")
        observed = float(stats["cards"])
        return _over_under(selection, observed), f"cards={observed:g}"

    if "over" in text or "under" in text or "goles" in text or market_text in {"totales", "totals"}:
        observed = float(hg + ag)
        return _over_under(selection, observed), f"score={hg}-{ag}; goals={observed:g}"

    if text in {"gana local", "local", "1"}:
        outcome = "win" if hg > ag else "loss"
    elif text in {"empate", "draw", "x"}:
        outcome = "win" if hg == ag else "loss"
    elif text in {"gana visita", "visita", "2"}:
        outcome = "win" if ag > hg else "loss"
    else:
        raise ValueError(f"Unsupported selection: {selection}")
    return outcome, f"score={hg}-{ag}"


def _profit(outcome, odds, stake):
    if outcome == "win":
        units = max(0.0, odds - 1.0)
    elif outcome == "loss":
        units = -1.0
    else:
        units = 0.0
    return round(units, 4), round(stake * units, 2)


def _update_rows(session, updates):
    if not updates:
        return 0
    payload = {"valueInputOption": "RAW", "data": []}
    for row_number, values in updates:
        payload["data"].append({
            "range": f"{WORKSHEET}!U{row_number}:Y{row_number}",
            "majorDimension": "ROWS",
            "values": [values],
        })
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values:batchUpdate"
    response = session.post(url, json=payload, timeout=20)
    if not response.ok:
        raise RuntimeError(f"Sheets batch update {response.status_code}: {response.text[:700]}")
    return len(updates)


def settle_pending_sheet():
    """Settle pending FUT Europa picks directly from Google Sheets.

    Uses API-Football final scores/statistics. Only FT/AET/PEN fixtures are graded.
    The operation is idempotent because only rows with result_status=pending are touched.
    """
    out = {
        "ok": False,
        "configured": bool(SPREADSHEET_ID and API_KEY),
        "checked": 0,
        "pending_seen": 0,
        "settled": 0,
        "updated": 0,
        "errors": [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not SPREADSHEET_ID:
        out["errors"].append("GOOGLE_SHEETS_ID missing")
        return out
    if not API_KEY:
        out["errors"].append("API_SPORTS_KEY missing")
        return out

    try:
        session = _session()
        response = session.get(_values_url(f"{WORKSHEET}!A1:Y2000"), timeout=20)
        if not response.ok:
            raise RuntimeError(f"Sheets read {response.status_code}: {response.text[:700]}")
        rows = response.json().get("values", [])
        if not rows:
            out["ok"] = True
            return out
        headers = rows[0]
        if headers != HEADERS:
            raise RuntimeError("FUT_Europa_Picks header mismatch")

        idx = {name: i for i, name in enumerate(headers)}
        fixture_cache = {}
        stats_cache = {}
        updates = []

        for row_number, raw in enumerate(rows[1:], start=2):
            row = list(raw) + [""] * (len(headers) - len(raw))
            if str(row[idx["result_status"]]).strip().lower() != "pending":
                continue
            out["pending_seen"] += 1
            fixture_id = str(row[idx["fixture_id"]]).strip()
            if not fixture_id or fixture_id == "999999":
                out["errors"].append(f"row {row_number}: fixture_id not settleable ({fixture_id or 'blank'})")
                continue
            out["checked"] += 1
            try:
                if fixture_id not in fixture_cache:
                    fixture_cache[fixture_id] = _fixture_result(fixture_id)
                result = fixture_cache[fixture_id]
                if not result or not result.get("final"):
                    continue

                selection = row[idx["selection"]]
                market = row[idx["market"]]
                need_stats = any(x in str(selection).lower() for x in ("corner", "tarjeta", "card"))
                stats = {}
                if need_stats:
                    if fixture_id not in stats_cache:
                        stats_cache[fixture_id] = _fixture_statistics(fixture_id)
                    stats = stats_cache[fixture_id]

                outcome, result_value = _grade(selection, market, result, stats)
                odds = _num(row[idx["odds"]])
                stake = _num(row[idx["stake_mxn"]])
                units, profit_mxn = _profit(outcome, odds, stake)
                settled_utc = datetime.now(timezone.utc).isoformat()
                updates.append((row_number, [outcome, result_value, units, profit_mxn, settled_utc]))
                out["settled"] += 1
            except Exception as exc:
                out["errors"].append(f"row {row_number} fixture {fixture_id}: {type(exc).__name__}: {exc}")

        out["updated"] = _update_rows(session, updates)
        out["ok"] = True
        return out
    except Exception as exc:
        out["errors"].append(f"{type(exc).__name__}: {exc}")
        return out
