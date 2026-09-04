import os
import re
import requests
import pandas as pd
import unicodedata

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}


def american_to_decimal(american):
    try:
        american = float(american)
    except Exception:
        return 0.0
    if american == 0:
        return 0.0
    return round((american / 100.0) + 1.0, 3) if american > 0 else round((100.0 / abs(american)) + 1.0, 3)


def normalizar_nombre(nombre):
    return unicodedata.normalize("NFKD", str(nombre)).encode("ASCII", "ignore").decode("utf-8").lower().strip()


def _extract_line(value):
    m = re.search(r"(?:Over|Under)\s*([0-9]+(?:\.[0-9]+)?)", str(value), flags=re.I)
    return float(m.group(1)) if m else None


def _preferred_line(lines, canonical):
    if not lines:
        return None
    uniq = sorted({float(x) for x in lines})
    if float(canonical) in uniq:
        return float(canonical)
    return min(uniq, key=lambda x: abs(x - float(canonical)))


def extraer_cuotas_espn(nombre_liga, local, visita):
    """Obtiene únicamente cuotas explícitamente publicadas por ESPN."""
    espn_leagues = {
        "Premier League": "eng.1",
        "La Liga": "esp.1",
        "Serie A": "ita.1",
        "Bundesliga": "ger.1",
        "Ligue 1": "fra.1",
    }
    codigo = espn_leagues.get(nombre_liga)
    if not codigo:
        return {}

    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{codigo}/scoreboard"
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
    except Exception:
        return {}

    loc_norm = normalizar_nombre(local)
    vis_norm = normalizar_nombre(visita)
    for event in res.json().get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((normalizar_nombre(x.get("team", {}).get("displayName", "")) for x in competitors if x.get("homeAway") == "home"), "")
        away = next((normalizar_nombre(x.get("team", {}).get("displayName", "")) for x in competitors if x.get("homeAway") == "away"), "")
        if not ((loc_norm in home or home in loc_norm) and (vis_norm in away or away in vis_norm)):
            continue

        odds_list = comp.get("odds", [])
        if not odds_list:
            return {}
        od = odds_list[0]
        out = {}
        if isinstance(od.get("homeTeamOdds"), dict):
            out["1"] = american_to_decimal(od["homeTeamOdds"].get("moneyLine", 0))
        if isinstance(od.get("drawOdds"), dict):
            out["X"] = american_to_decimal(od["drawOdds"].get("moneyLine", 0))
        if isinstance(od.get("awayTeamOdds"), dict):
            out["2"] = american_to_decimal(od["awayTeamOdds"].get("moneyLine", 0))
        return {k: v for k, v in out.items() if v > 1.01}
    return {}


def obtener_cuotas_europa(fixture_id, nombre_liga=None, local=None, visita=None):
    """Devuelve precios reales y la línea real seleccionada del sportsbook."""
    cuotas = {"1": 0.0, "X": 0.0, "2": 0.0, "_lineas": {}, "_bookmaker": None}

    exito_api = False
    if API_KEY and fixture_id not in (None, 999999):
        try:
            response = requests.get(
                f"{BASE_URL}/odds", headers=HEADERS,
                params={"fixture": str(fixture_id)}, timeout=8
            )
            if response.status_code == 200:
                data = response.json().get("response", [])
                if data:
                    bookmakers = data[0].get("bookmakers", [])
                    if bookmakers:
                        bm = next((x for x in bookmakers if x.get("id") == 8), bookmakers[0])
                        cuotas["_bookmaker"] = bm.get("name")
                        found_lines = {"goles": [], "corners": [], "tarjetas": []}

                        for mercado in bm.get("bets", []):
                            mid = mercado.get("id")
                            bet_name = normalizar_nombre(mercado.get("name", ""))
                            if mid == 5 or ("goal" in bet_name and "over" in bet_name):
                                tipo = "goles"
                            elif mid == 45 or "corner" in bet_name:
                                tipo = "corners"
                            elif "card" in bet_name or "booking" in bet_name or "tarjeta" in bet_name:
                                tipo = "tarjetas"
                            else:
                                tipo = None

                            for valor in mercado.get("values", []):
                                key = str(valor.get("value", "")).strip()
                                try:
                                    odd = float(valor.get("odd"))
                                except Exception:
                                    continue

                                if mid == 1 and key == "Home":
                                    cuotas["1"] = odd
                                    exito_api = True
                                elif mid == 1 and key == "Draw":
                                    cuotas["X"] = odd
                                elif mid == 1 and key == "Away":
                                    cuotas["2"] = odd
                                elif tipo and key.lower().startswith(("over", "under")):
                                    line = _extract_line(key)
                                    if line is None:
                                        continue
                                    found_lines[tipo].append(line)
                                    suffix = "Goles" if tipo == "goles" else "Corners" if tipo == "corners" else "Tarjetas"
                                    side = "Over" if key.lower().startswith("over") else "Under"
                                    cuotas[f"{side} {line:g} {suffix}"] = odd
                                    exito_api = True

                        canonical = {"goles": 2.5, "corners": 9.5, "tarjetas": 4.5}
                        for tipo, lines in found_lines.items():
                            chosen = _preferred_line(lines, canonical[tipo])
                            if chosen is not None:
                                cuotas["_lineas"][tipo] = chosen
        except Exception:
            pass

    if not exito_api and nombre_liga and local and visita:
        for k, v in extraer_cuotas_espn(nombre_liga, local, visita).items():
            cuotas[k] = v
    return cuotas


def calcular_kelly_fraccional(prob_modelo_decimal, cuota_decimal, fraccion=0.20):
    if cuota_decimal <= 1.0 or not 0 < prob_modelo_decimal < 1:
        return 0.0
    b = cuota_decimal - 1.0
    kelly = ((b * prob_modelo_decimal) - (1.0 - prob_modelo_decimal)) / b
    return round(max(0.0, kelly) * fraccion * 100.0, 2)


def _mc_market_probability(resultados_mc, tipo, side, line):
    label = "Goles" if tipo == "goles" else "Corners" if tipo == "corners" else "Tarjetas"
    try:
        return float(resultados_mc.get("Lineas_Casino", {}).get(tipo, {}).get(f"{side} {line:g} {label}", 0.0))
    except Exception:
        return 0.0


def analizar_apuestas_europa(resultados_mc, preds_ml, fixture_id, cuotas_personalizadas=None,
                              nombre_liga=None, local=None, visita=None):
    cuotas = cuotas_personalizadas if cuotas_personalizadas is not None else obtener_cuotas_europa(fixture_id, nombre_liga, local, visita)
    if not cuotas:
        return pd.DataFrame()

    def get_prob(d, cat, key):
        try:
            return float(d.get(cat, {}).get(key, 0.0))
        except Exception:
            return 0.0

    mercados = [
        {"nombre": "Gana Local", "cat": "Resultado_1X2", "key": "Gana Local", "odd_key": "1", "tipo": "1x2"},
        {"nombre": "Empate", "cat": "Resultado_1X2", "key": "Empate", "odd_key": "X", "tipo": "1x2"},
        {"nombre": "Gana Visita", "cat": "Resultado_1X2", "key": "Gana Visita", "odd_key": "2", "tipo": "1x2"},
    ]

    lineas = cuotas.get("_lineas", {}) if isinstance(cuotas.get("_lineas", {}), dict) else {}
    for tipo in ("goles", "corners", "tarjetas"):
        line = lineas.get(tipo)
        if line is None:
            continue
        try:
            line = float(line)
        except Exception:
            continue
        label = "Goles" if tipo == "goles" else "Corners" if tipo == "corners" else "Tarjetas"
        for side in ("Over", "Under"):
            odd_key = f"{side} {line:g} {label}"
            mercados.append({"nombre": odd_key, "odd_key": odd_key, "tipo": tipo, "line": line, "side": side})

    rows = []
    for m in mercados:
        try:
            cuota = float(cuotas.get(m["odd_key"], 0.0))
        except Exception:
            cuota = 0.0
        if cuota <= 1.01:
            continue

        if m["tipo"] == "1x2":
            prob_mc = get_prob(resultados_mc, m["cat"], m["key"])
            prob_ml = get_prob(preds_ml, m["cat"], m["key"])
        else:
            line = float(m["line"])
            prob_mc = _mc_market_probability(resultados_mc, m["tipo"], m["side"], line)
            if m["tipo"] == "goles":
                cat, key = "Goles_Over_Under", f"{m['side']} {line:g}"
            elif m["tipo"] == "corners":
                cat, key = "Corners_Totales", f"{m['side']} {line:g} Corners"
            else:
                cat, key = "Tarjetas_Totales", f"{m['side']} {line:g} Tarjetas"
            prob_ml = get_prob(preds_ml, cat, key)

        if prob_mc <= 0 or prob_ml <= 0:
            continue

        disagreement = abs(prob_mc - prob_ml)
        prob_modelo_pct = 0.65 * prob_ml + 0.35 * prob_mc
        source = "ML line-aware + MC"
        p = prob_modelo_pct / 100.0
        ev_pct = ((p * cuota) - 1.0) * 100.0
        kelly = calcular_kelly_fraccional(p, cuota)

        if disagreement > 12.0:
            verdict = "❌ NO BET — modelos en desacuerdo"
            kelly = 0.0
        elif min(prob_mc, prob_ml) < 55.0:
            verdict = "❌ NO BET — confianza insuficiente"
            kelly = 0.0
        elif ev_pct >= 8.0 and kelly >= 1.0:
            verdict = "🔥 Value Fuerte"
        elif ev_pct >= 3.0 and kelly >= 0.5:
            verdict = "✅ Value Moderado"
        elif ev_pct > 0:
            verdict = "⚠️ EV Positivo Marginal"
            kelly = 0.0
        else:
            verdict = "❌ EV Negativo"
            kelly = 0.0

        rows.append({
            "Mercado": m["nombre"],
            "Fuente prob.": source,
            "Prob. MC": f"{prob_mc:.1f}%",
            "Prob. ML": f"{prob_ml:.1f}%",
            "Prob. usada": f"{prob_modelo_pct:.1f}%",
            "Desacuerdo pp": round(disagreement, 1),
            "Cuota real": round(cuota, 3),
            "EV (Valor)": f"{ev_pct:.2f}%",
            "Stake Recomendado": f"{kelly:.2f}% Bank",
            "Veredicto": verdict,
        })
    return pd.DataFrame(rows)
