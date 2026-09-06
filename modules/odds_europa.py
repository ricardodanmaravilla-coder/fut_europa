import os
import re
import unicodedata

import pandas as pd
import requests

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}

PRIMARY_BOOKMAKER = "bet365"
CALIBRATION_VERSION = "strict-v7-bet365-primary-per-market-fallback-line-lock"
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
    mid = mercado.get("id")
    name = normalizar_nombre(mercado.get("name", ""))
    if mid == 1:
        return "1x2"
    if mid == 5:
        return "goles"
    if mid == 45:
        return "corners"
    if "card" in name or "booking" in name or "tarjeta" in name:
        return "tarjetas"
    return None


def _market_has_usable_price(mercado, tipo):
    if _market_type(mercado) != tipo:
        return False
    for valor in mercado.get("values", []) or []:
        try:
            odd = float(valor.get("odd"))
        except Exception:
            continue
        if odd > 1.01:
            return True
    return False


def _bookmaker_has_market(bookmaker, tipo):
    if not bookmaker:
        return False
    return any(_market_has_usable_price(m, tipo) for m in bookmaker.get("bets", []) or [])


def _select_bookmaker_for_market(bookmakers, tipo):
    normalized = [(normalizar_nombre(b.get("name", "")), b) for b in (bookmakers or [])]

    # Bet365 is ALWAYS the first choice for every individual market.
    primary = next(
        (b for name, b in normalized if PRIMARY_BOOKMAKER in name and _bookmaker_has_market(b, tipo)),
        None,
    )
    if primary:
        return primary, "bet365", True

    # Only if Bet365 lacks this specific market, use another real bookmaker
    # that actually exposes that same market. Never replace Bet365 just for price.
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


def _supported_line(tipo, line):
    try:
        return round(float(line), 2) in SUPPORTED_HALF_LINES.get(tipo, set())
    except Exception:
        return False


def _balanced_supported_line(tipo, price_map):
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
        symmetry = abs(over - under)
        candidates.append((round(balance, 8), round(symmetry, 8), line))

    if not candidates:
        return None, "missing_supported"
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    best = candidates[0]
    tied = [
        c
        for c in candidates
        if abs(c[0] - best[0]) < 1e-8 and abs(c[1] - best[1]) < 1e-8
    ]
    if len(tied) > 1:
        return None, f"balanced_tie:{len(tied)}"
    return best[2], f"balanced_supported:{len(candidates)}"


def _parse_market_from_bookmaker(bookmaker, tipo, cuotas):
    line_prices = {}
    for mercado in bookmaker.get("bets", []) or []:
        if _market_type(mercado) != tipo:
            continue
        mid = mercado.get("id")
        for valor in mercado.get("values", []) or []:
            key = str(valor.get("value", "")).strip()
            try:
                odd = float(valor.get("odd"))
            except Exception:
                continue
            if odd <= 1.01:
                continue

            if tipo == "1x2" and mid == 1:
                if key == "Home":
                    cuotas["1"] = odd
                elif key == "Draw":
                    cuotas["X"] = odd
                elif key == "Away":
                    cuotas["2"] = odd
                continue

            if tipo in {"goles", "corners", "tarjetas"} and key.lower().startswith(("over", "under")):
                line = _extract_line(key)
                if line is None:
                    continue
                side = "Over" if key.lower().startswith("over") else "Under"
                suffix = "Goles" if tipo == "goles" else "Corners" if tipo == "corners" else "Tarjetas"
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

            chosen, status = _balanced_supported_line(tipo, line_prices)
            if chosen is not None:
                cuotas["_lineas"][tipo] = chosen
                prefix = "fullmatch_market:" if tipo == "goles" else ""
                cuotas["_line_status"][tipo] = f"{prefix}{status}"
            else:
                complete = []
                for line, sides in line_prices.items():
                    try:
                        if float(sides.get("Over", 0)) > 1.01 and float(sides.get("Under", 0)) > 1.01:
                            complete.append(float(line))
                    except Exception:
                        pass
                if status.startswith("balanced_tie"):
                    cuotas["_line_status"][tipo] = status
                elif complete:
                    cuotas["_line_status"][tipo] = "unsupported_or_no_supported_complete"
                else:
                    cuotas["_line_status"][tipo] = "missing"

        # Backward-compatible summary metadata for existing UI/diagnostics.
        used_bookmakers = [v for v in cuotas["_bookmakers"].values() if v]
        cuotas["_bookmaker"] = used_bookmakers[0] if used_bookmakers else None
        modes = [v for v in cuotas["_pricing_modes"].values() if v != "unavailable"]
        cuotas["_pricing_mode"] = "mixed-per-market" if len(set(modes)) > 1 else (modes[0] if modes else "unavailable")
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
