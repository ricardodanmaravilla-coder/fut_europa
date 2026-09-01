import os
import re
from datetime import datetime, timezone
from urllib.parse import quote

import google.auth
import requests
from google.auth.transport.requests import AuthorizedSession

SPREADSHEET_ID = os.environ.get("GOOGLE_SHEETS_ID", "1VsB21QUsQL5EyXu7Sek5WVeNVznECiTuoIMMB4JXno4")
WORKSHEET = os.environ.get("FUT_SHEETS_WORKSHEET", "FUT_Europa_Picks")
BANKROLL_MXN = float(os.environ.get("BANKROLL_MXN", "5000"))
MODEL_VERSION = os.environ.get("FUT_MODEL_VERSION", "fut-europa-v2.3-sheets")

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
    if not response.ok:
        raise RuntimeError(f"Sheets read {response.status_code}: {response.text[:500]}")
    values = response.json().get("values", [])
    return {str(row[0]) for row in values if row}


def _runtime_service_account():
    try:
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"}, timeout=2,
        )
        if response.ok:
            return response.text.strip()
    except Exception:
        pass
    return "unknown"


def sheets_diagnostic():
    out = {
        "configured": bool(SPREADSHEET_ID),
        "spreadsheet_id": SPREADSHEET_ID,
        "worksheet": WORKSHEET,
        "service_account": _runtime_service_account(),
        "model_version": MODEL_VERSION,
    }
    try:
        session = _session()
        keys = _existing_keys(session)
        header_response = session.get(_values_url(f"{WORKSHEET}!A1:Y1"), timeout=15)
        if not header_response.ok:
            raise RuntimeError(f"Sheets header read {header_response.status_code}: {header_response.text[:500]}")
        headers = (header_response.json().get("values") or [[]])[0]
        out.update({
            "ok": True,
            "read_access": True,
            "existing_rows": len(keys),
            "header_count": len(headers),
            "schema_ok": headers == HEADERS,
        })
    except Exception as exc:
        out.update({"ok": False, "read_access": False, "error": f"{type(exc).__name__}: {exc}"})
    return out


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
    """Append scanner recommendations to FUT_Europa_Picks, idempotently."""
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
        if not response.ok:
            return {
                "ok": False,
                "configured": True,
                "written": 0,
                "skipped": skipped,
                "error": f"Sheets append {response.status_code}: {response.text[:700]}",
            }
        return {"ok": True, "configured": True, "written": len(rows), "skipped": skipped}
    except Exception as exc:
        return {
            "ok": False,
            "configured": True,
            "written": 0,
            "skipped": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
