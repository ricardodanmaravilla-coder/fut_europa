import os
import re
import unicodedata

import pandas as pd
import requests

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}

PRIMARY_BOOKMAKER = "bet365"
CALIBRATION_VERSION = "strict-v10-bet365-canonical-ids-price-main-line-lock"
CANONICAL_MARKET_IDS = {
    "1x2": 1,
    "goles": 5,
    "corners": 45,
    "tarjetas": 80,
}
SUPPORTED_HALF_LINES = {
    "goles": {1.5, 2.5, 3.5, 4.5},
    "corners": {7.5, 8.5, 9.5, 10.5, 11.5, 12.5},
    "tarjetas": {2.5, 3.5, 4.5, 5.5, 6.5, 7.5},
}
MAX_DISAGREEMENT_PP = 8.0
MIN_MODEL_PROB_PCT = 60.0
STRONG_EV_PCT = 10.0
MODERATE_EV_PCT = 5.0
STRONG_KELLY_PCT = 1.0
MODERATE_KELLY_PCT = 0.75
KELLY_FRACTION = 0.10


def american_to_decimal(american):
    try:
        american = float(american)
    except Exception:
        return 0.0
    if american == 0:
        return 0.0
    if american > 0:
        return round((american / 100.0) + 1.0, 3)
    return round((100.0 / abs(american)) + 1.0, 3)


def normalizar_nombre(nombre):
    return (
        unicodedata.normalize("NFKD", str(nombre))
        .encode("ASCII", "ignore")
        .decode("utf-8")
        .lower()
        .strip()
    )


def _extract_line(value):
    match = re.search(r"(?:Over|Under)\s*([0-9]+(?:\.[0-9]+)?)", str(value), flags=re.I)
    return float(match.group(1)) if match else None


def _market_type(mercado):
    """Map only canonical API-Football pre-match market IDs.

    This intentionally does not infer cards/corners/goals by market name. That
    prevents team, player, half, handicap and other derivative props from being
    mixed into the full-match line ladder.
    """
    try:
        mid = int(mercado.get("id"))
    except Exception:
        return None
    for tipo, canonical_id in CANONICAL_MARKET_IDS.items():
        if mid == canonical_id:
            return tipo
    return None


def _supported_line(tipo, line):
    try:
        return round(float(line), 2) in SUPPORTED_HALF_LINES.get(tipo, set())
    except Exception:
        return False


def _market_price_map(mercado, tipo):
    """Return complete Over/Under price ladder for one canonical market."""
    price_map = {}
    if _market_type(mercado) != tipo or tipo == "1x2":
        return price_map
    for valor in mercado.get("values", []) or []:
        key = str(valor.get("value", "")).strip()
        low = key.lower()
        if not low.startswith(("over", "under")):
            continue
        line = _extract_line(key)
        if line is None:
            continue
        try:
            odd = float(valor.get("odd"))
        except Exception:
            continue
        if odd <= 1.01:
            continue
        side = "Over" if low.startswith("over") else "Under"
        price_map.setdefault(float(line), {})[side] = odd
    return price_map


def _market_has_usable_price(mercado, tipo):
    if _market_type(mercado) != tipo:
        return False
    if tipo == "1x2":
        found = set()
        for valor in mercado.get("values", []) or []:
            key = str(valor.get("value", "")).strip()
            try:
                odd = float(valor.get("odd"))
            except Exception:
                continue
            if odd > 1.01 and key in {"Home", "Draw", "Away"}:
                found.add(key)
        return len(found) == 3

    price_map = _market_price_map(mercado, tipo)
    for line, sides in price_map.items():
        try:
            over = float(sides.get("Over", 0))
            under = float(sides.get("Under", 0))
        except Exception:
            continue
        if over > 1.01 and under > 1.01 and _supported_line(tipo, line):
            return True
    return False


def _bookmaker_has_market(bookmaker, tipo):
    if not bookmaker:
        return False
    return any(_market_has_usable_price(m, tipo) for m in bookmaker.get("bets", []) or [])


def _select_bookmaker_for_market(bookmakers, tipo):
    """Bet365 first for every market; fallback only when that market is unusable."""
    normalized = [(normalizar_nombre(b.get("name", "")), b) for b in (bookmakers or [])]
    primary = next(
        (b for name, b in normalized if PRIMARY_BOOKMAKER in name and _bookmaker_has_market(b, tipo)),
        None,
    )
    if primary:
        return primary, "bet365", True

    fallback = next(
        (
            b
            for name, b in normalized
            if PRIMARY_BOOKMAKER not in name and _bookmaker_has_market(b, tipo)
        ),
        None,
    )
    if fallback:
        name = fallback.get("name") or "fallback"
        return fallback, f"fallback:{name}", True
    return None, "unavailable", False


def _main_supported_line(tipo, price_map):
    """Choose the sportsbook main line from the canonical full-match ladder.

    Among complete supported Over/Under pairs, the main line is the one whose
    no-vig Over probability is closest to 50%. Overround and line value are only
    deterministic tie breakers. This avoids selecting an alternate ladder line
    merely because it appears first or sits in the middle of the returned list.
    """
    candidates = []
    for line, sides in (price_map or {}).items():
        try:
            line = float(line)
            over = float(sides.get("Over", 0))
            under = float(sides.get("Under", 0))
        except Exception:
            continue
        if over <= 1.01 or under <= 1.01 or not _supported_line(tipo, line):
            continue
        implied_over = 1.0 / over
        implied_under = 1.0 / under
        total = implied_over + implied_under
        if total <= 0:
            continue
        p_over_no_vig = implied_over / total
        balance = abs(p_over_no_vig - 0.5)
        overround = abs(total - 1.0)
        candidates.append((round(balance, 10), round(overround, 10), line))

    if not candidates:
        return None, "missing_supported"
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    chosen = candidates[0]
    return chosen[2], f"price_main_no_vig:{len(candidates)}"


# Compatibility alias for older tests/imports; logic is now the canonical v10 selector.
def _balanced_supported_line(tipo, price_map):
    return _main_supported_line(tipo, price_map)


def _parse_market_from_bookmaker(bookmaker, tipo, cuotas):
    line_prices = {}
    canonical_id = CANONICAL_MARKET_IDS[tipo]
    for mercado in bookmaker.get("bets", []) or []:
        if _market_type(mercado) != tipo:
            continue
        try:
            mid = int(mercado.get("id"))
        except Exception:
            continue
        if mid != canonical_id:
            continue

        if tipo == "1x2":
            for valor in mercado.get("values", []) or []:
                key = str(valor.get("value", "")).strip()
                try:
                    odd = float(valor.get("odd"))
                except Exception:
                    continue
                if odd <= 1.01:
                    continue
                if key == "Home":
                    cuotas["1"] = odd
                elif key == "Draw":
                    cuotas["X"] = odd
                elif key == "Away":
                    cuotas["2"] = odd
            continue

        market_prices = _market_price_map(mercado, tipo)
        suffix = "Goles" if tipo == "goles" else "Corners" if tipo == "corners" else "Tarjetas"
        for line, sides in market_prices.items():
            for side in ("Over", "Under"):
                try:
                    odd = float(sides.get(side, 0))
                except Exception:
                    continue
                if odd <= 1.01:
                    continue
                cuotas[f"{side} {line:g} {suffix}"] = odd
                line_prices.setdefault(float(line), {})[side] = odd
    return line_prices


def obtener_cuotas_europa(fixture_id, nombre_liga=None, local=None, visita=None):
    cuotas = {
        "1": 0.0,
        "X": 0.0,
        "2": 0.0,
        "_lineas": {},
        "_bookmaker": None,
        "_pricing_mode": "unavailable",
        "_persist_allowed": False,
        "_bookmakers": {},
        "_pricing_modes": {},
        "_market_persist_allowed": {},
        "_line_status": {},
        "_calibration": CALIBRATION_VERSION,
    }
    if not API_KEY or fixture_id in (None, 999999):
        return cuotas

    try:
        response = requests.get(
            f"{BASE_URL}/odds",
            headers=HEADERS,
            params={"fixture": str(fixture_id)},
            timeout=8,
        )
        if response.status_code != 200:
            return cuotas
        data = response.json().get("response", [])
        if not data:
            return cuotas
        bookmakers = data[0].get("bookmakers", []) or []

        for tipo in ("1x2", "goles", "corners", "tarjetas"):
            bm, pricing_mode, persist_allowed = _select_bookmaker_for_market(bookmakers, tipo)
            cuotas["_bookmakers"][tipo] = bm.get("name") if bm else None
            cuotas["_pricing_modes"][tipo] = pricing_mode
            cuotas["_market_persist_allowed"][tipo] = bool(persist_allowed)
            if not bm:
                if tipo != "1x2":
                    cuotas["_line_status"][tipo] = "missing"
                continue

            line_prices = _parse_market_from_bookmaker(bm, tipo, cuotas)
            if tipo == "1x2":
                continue

            chosen, status = _main_supported_line(tipo, line_prices)
            if chosen is not None:
                cuotas["_lineas"][tipo] = chosen
                cuotas["_line_status"][tipo] = f"canonical_id:{CANONICAL_MARKET_IDS[tipo]}:{status}"
            else:
                cuotas["_line_status"][tipo] = "missing_supported_complete_line"
                cuotas["_market_persist_allowed"][tipo] = False

        used_bookmakers = [v for v in cuotas["_bookmakers"].values() if v]
        cuotas["_bookmaker"] = used_bookmakers[0] if used_bookmakers else None
        modes = [v for v in cuotas["_pricing_modes"].values() if v != "unavailable"]
        cuotas["_pricing_mode"] = (
            "mixed-per-market" if len(set(modes)) > 1 else (modes[0] if modes else "unavailable")
        )
        cuotas["_persist_allowed"] = any(cuotas["_market_persist_allowed"].values())
    except Exception:
        return cuotas
    return cuotas


def calcular_kelly_fraccional(prob_modelo_decimal, cuota_decimal, fraccion=KELLY_FRACTION):
    if cuota_decimal <= 1.0 or not 0 < prob_modelo_decimal < 1:
        return 0.0
    b = cuota_decimal - 1.0
    kelly = ((b * prob_modelo_decimal) - (1.0 - prob_modelo_decimal)) / b
    return round(max(0.0, kelly) * fraccion * 100.0, 2)


def _mc_market_probability(resultados_mc, tipo, side, line):
    label = "Goles" if tipo == "goles" else "Corners" if tipo == "corners" else "Tarjetas"
    try:
        return float(
            resultados_mc.get("Lineas_Casino", {})
            .get(tipo, {})
            .get(f"{side} {line:g} {label}", 0.0)
        )
    except Exception:
        return 0.0


def _line_locked(cuotas, preds_ml, resultados_mc, tipo, line, side):
    try:
        sportsbook = float(cuotas.get("_lineas", {}).get(tipo))
        ml = float(preds_ml.get("Meta", {}).get("lineas_modeladas", {}).get(tipo))
        if abs(sportsbook - line) > 1e-9 or abs(ml - line) > 1e-9:
            return False
        label = "Goles" if tipo == "goles" else "Corners" if tipo == "corners" else "Tarjetas"
        key = f"{side} {line:g} {label}"
        return key in resultados_mc.get("Lineas_Casino", {}).get(tipo, {})
    except Exception:
        return False


def analizar_apuestas_europa(
    resultados_mc,
    preds_ml,
    fixture_id,
    cuotas_personalizadas=None,
    nombre_liga=None,
    local=None,
    visita=None,
):
    cuotas = (
        cuotas_personalizadas
        if cuotas_personalizadas is not None
        else obtener_cuotas_europa(fixture_id, nombre_liga, local, visita)
    )
    if not cuotas:
        return pd.DataFrame()

    def get_prob(data, cat, key):
        try:
            return float(data.get(cat, {}).get(key, 0.0))
        except Exception:
            return 0.0

    mercados = [
        {"nombre": "Gana Local", "cat": "Resultado_1X2", "key": "Gana Local", "odd_key": "1", "tipo": "1x2"},
        {"nombre": "Empate", "cat": "Resultado_1X2", "key": "Empate", "odd_key": "X", "tipo": "1x2"},
        {"nombre": "Gana Visita", "cat": "Resultado_1X2", "key": "Gana Visita", "odd_key": "2", "tipo": "1x2"},
    ]
    lineas = cuotas.get("_lineas", {}) if isinstance(cuotas.get("_lineas", {}), dict) else {}
    for tipo in ("goles", "corners", "tarjetas"):
        try:
            line = float(lineas[tipo])
        except Exception:
            continue
        label = "Goles" if tipo == "goles" else "Corners" if tipo == "corners" else "Tarjetas"
        for side in ("Over", "Under"):
            odd_key = f"{side} {line:g} {label}"
            mercados.append({"nombre": odd_key, "odd_key": odd_key, "tipo": tipo, "line": line, "side": side})

    rows = []
    per_market_books = cuotas.get("_bookmakers", {}) or {}
    per_market_modes = cuotas.get("_pricing_modes", {}) or {}
    per_market_allowed = cuotas.get("_market_persist_allowed", {}) or {}

    for market in mercados:
        tipo = market["tipo"]
        try:
            cuota = float(cuotas.get(market["odd_key"], 0.0))
        except Exception:
            cuota = 0.0
        if cuota <= 1.01:
            continue

        if tipo == "1x2":
            prob_mc = get_prob(resultados_mc, market["cat"], market["key"])
            prob_ml = get_prob(preds_ml, market["cat"], market["key"])
        else:
            line = float(market["line"])
            if not _line_locked(cuotas, preds_ml, resultados_mc, tipo, line, market["side"]):
                continue
            prob_mc = _mc_market_probability(resultados_mc, tipo, market["side"], line)
            if tipo == "goles":
                cat, key = "Goles_Over_Under", f"{market['side']} {line:g}"
            elif tipo == "corners":
                cat, key = "Corners_Totales", f"{market['side']} {line:g} Corners"
            else:
                cat, key = "Tarjetas_Totales", f"{market['side']} {line:g} Tarjetas"
            prob_ml = get_prob(preds_ml, cat, key)

        if prob_mc <= 0 or prob_ml <= 0:
            continue

        disagreement = abs(prob_mc - prob_ml)
        prob_modelo_pct = 0.65 * prob_ml + 0.35 * prob_mc
        p = prob_modelo_pct / 100.0
        ev_pct = ((p * cuota) - 1.0) * 100.0
        kelly = calcular_kelly_fraccional(p, cuota)

        if disagreement > MAX_DISAGREEMENT_PP:
            verdict = "❌ NO BET — modelos en desacuerdo"
            kelly = 0.0
        elif min(prob_mc, prob_ml) < MIN_MODEL_PROB_PCT:
            verdict = "❌ NO BET — confianza insuficiente"
            kelly = 0.0
        elif ev_pct >= STRONG_EV_PCT and kelly >= STRONG_KELLY_PCT:
            verdict = "🔥 Value Fuerte"
        elif ev_pct >= MODERATE_EV_PCT and kelly >= MODERATE_KELLY_PCT:
            verdict = "✅ Value Moderado"
        elif ev_pct > 0:
            verdict = "⚠️ EV Positivo Marginal"
            kelly = 0.0
        else:
            verdict = "❌ EV Negativo"
            kelly = 0.0

        official = bool(per_market_allowed.get(tipo, False))
        if not official and ("🔥" in verdict or "✅" in verdict):
            verdict = "🧪 BOOKMAKER NO OFICIAL"
            kelly = 0.0

        bookmaker = per_market_books.get(tipo) or "sin bookmaker"
        pricing_mode = per_market_modes.get(tipo, "unavailable")
        rows.append(
            {
                "Mercado": market["nombre"],
                "Bookmaker": bookmaker,
                "Modo cuota": pricing_mode,
                "Calibración": CALIBRATION_VERSION,
                "Fuente prob.": "ML line-aware + MC line-locked",
                "Prob. MC": f"{prob_mc:.1f}%",
                "Prob. ML": f"{prob_ml:.1f}%",
                "Prob. usada": f"{prob_modelo_pct:.1f}%",
                "Desacuerdo pp": round(disagreement, 1),
                "Cuota real": round(cuota, 3),
                "EV (Valor)": f"{ev_pct:.2f}%",
                "Stake Recomendado": f"{kelly:.2f}% Bank",
                "Veredicto": verdict,
            }
        )
    return pd.DataFrame(rows)
