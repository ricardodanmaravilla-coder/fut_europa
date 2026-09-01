import os
import re
from datetime import datetime, timezone
from urllib.parse import quote

import google.auth
from google.auth.transport.requests import AuthorizedSession

SPREADSHEET_ID = os.environ.get("GOOGLE_SHEETS_ID", "1VsB21QUsQL5EyXu7Sek5WVeNVznECiTuoIMMB4JXno4")
WORKSHEET = os.environ.get("FUT_SHEETS_WORKSHEET", "FUT_Europa_Picks")
BANKROLL_MXN = float(os.environ.get("BANKROLL_MXN", "5000"))
MODEL_VERSION = os.environ.get("FUT_MODEL_VERSION", "fut-europa-v2.2-sheets")

HEADERS = [
    "record_key", "snapshot_utc", "game_date", "fixture_id", "league", "away", "home",
    "market", "selection", "odds", "prob_ml", "prob_mc", "prob_combined", "disagreement_pp",
    "ev_pct", "kelly_pct", "bankroll_mxn", "stake_mxn", "verdict", "model_version",
    "result_status", "result_value", "profit_units", "profit_mxn", "settled_utc",
]


def _number(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else default


def _market_group(selection):
    text = str(selection).lower()
    if "corner" in text:
        return "Corners"
    if "tarjeta" in text:
        return "Tarjetas"
    if "goles" in text or "over" in text or "under" in text:
        return "Totales"
    return "1X2"


def _session():
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/spreadsheets"])
    return AuthorizedSession(credentials)


def _values_url(range_a1):
    encoded = quote(range_a1, safe="!:$")
    return f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{encoded}"


def _existing_keys(session):
    response = session.get(_values_url(f"{WORKSHEET}!A2:A2000"), timeout=15)
    response.raise_for_status()
    values = response.json().get("values", [])
    return {str(row[0]) for row in values if row}


def _build_row(liga, fixture, bet):
    selection = str(bet.get("Mercado", "")).strip()
    fixture_id = fixture.get("fixture_id", "")
    game_date = str(fixture.get("fecha", ""))[:10]
    home = str(fixture.get("local", ""))
    away = str(fixture.get("visita", ""))
    odds = _number(bet.get("Cuota real"))
    prob_ml = _number(bet.get("Prob. ML"))
    prob_mc = _number(bet.get("Prob. MC"))
    prob_combined = _number(bet.get("Prob. usada"))
    disagreement = _number(bet.get("Desacuerdo pp"))
    ev_pct = _number(bet.get("EV (Valor)"))
    kelly_pct = _number(bet.get("Stake Recomendado"))
    stake_mxn = round(BANKROLL_MXN * kelly_pct / 100.0, 2)
    record_key = f"{game_date}|{fixture_id}|{liga}|{away}|{home}|{selection}|{MODEL_VERSION}"
    now = datetime.now(timezone.utc).isoformat()
    return [
        record_key, now, game_date, fixture_id, liga, away, home,
        _market_group(selection), selection, odds, prob_ml, prob_mc, prob_combined, disagreement,
        ev_pct, kelly_pct, BANKROLL_MXN, stake_mxn, str(bet.get("Veredicto", "")), MODEL_VERSION,
        "pending", "", "", "", "",
    ]


def persist_recommendations(liga, fixture, bets):
    """Append scanner recommendations to FUT_Europa_Picks, idempotently.

    Uses Cloud Run Application Default Credentials and fails soft so a Sheets problem never breaks the scanner.
    """
    if not SPREADSHEET_ID:
        return {"ok": False, "configured": False, "written": 0, "skipped": 0, "error": "GOOGLE_SHEETS_ID missing"}
    if not bets:
        return {"ok": True, "configured": True, "written": 0, "skipped": 0}
    try:
        session = _session()
        existing = _existing_keys(session)
        rows = []
        skipped = 0
        for bet in bets:
            row = _build_row(liga, fixture, bet)
            if row[0] in existing:
                skipped += 1
                continue
            rows.append(row)
            existing.add(row[0])
        if not rows:
            return {"ok": True, "configured": True, "written": 0, "skipped": skipped}
        url = _values_url(f"{WORKSHEET}!A:Y") + ":append"
        response = session.post(
            url,
            params={"valueInputOption": "RAW", "insertDataOption": "INSERT_ROWS"},
            json={"majorDimension": "ROWS", "values": rows},
            timeout=20,
        )
        response.raise_for_status()
        return {"ok": True, "configured": True, "written": len(rows), "skipped": skipped}
    except Exception as exc:
        return {
            "ok": False,
            "configured": True,
            "written": 0,
            "skipped": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
