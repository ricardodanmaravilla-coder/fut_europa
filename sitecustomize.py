"""Runtime safety patch for FUT Europa sportsbook line selection.

Loaded automatically by Python at startup. This patch pins the supported
pre-match market IDs to API-Football's canonical full-match markets:
1 Match Winner, 5 Goals Over/Under, 45 Corners Over/Under, 80 Cards Over/Under.
It prevents card derivatives (team/player/half/asian handicap, etc.) from being
mistaken for the full-match total-card market.
"""

try:
    import modules.odds_europa as oe

    def _market_type_strict(mercado):
        try:
            mid = int(mercado.get("id"))
        except Exception:
            return None

        if mid == 1:
            return "1x2"
        if mid == 5:
            return "goles"
        if mid == 45:
            return "corners"
        if mid == 80:
            return "tarjetas"
        return None

    def _central_supported_line(tipo, price_map):
        complete = []
        for line, sides in (price_map or {}).items():
            try:
                line = float(line)
                over = float(sides.get("Over", 0))
                under = float(sides.get("Under", 0))
            except Exception:
                continue
            if over <= 1.01 or under <= 1.01 or not oe._supported_line(tipo, line):
                continue

            implied_over = 1.0 / over
            implied_under = 1.0 / under
            total = implied_over + implied_under
            balance = abs((implied_over / total) - 0.5) if total > 0 else 999.0
            overround = abs(total - 1.0) if total > 0 else 999.0
            complete.append((line, round(balance, 8), round(overround, 8)))

        if not complete:
            return None, "missing_supported"

        complete.sort(key=lambda x: x[0])
        n = len(complete)
        if n == 1:
            return complete[0][0], "provider_main:1"
        if n % 2 == 1:
            return complete[n // 2][0], f"central_main:{n}"

        left = complete[(n // 2) - 1]
        right = complete[n // 2]
        chosen = min((left, right), key=lambda x: (x[1], x[2], x[0]))
        return chosen[0], f"central_main_even:{n}"

    oe._market_type = _market_type_strict
    oe._balanced_supported_line = _central_supported_line
    oe.CALIBRATION_VERSION = "strict-v9-bet365-canonical-market-ids-line-lock"
except Exception:
    pass
