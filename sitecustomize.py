"""Runtime safety patch for FUT Europa sportsbook line selection.

Loaded automatically by Python at startup. It tightens full-match card-market
classification and replaces the old 'most balanced price' heuristic with a
central-line selector for totals. This prevents alternative Bet365 lines such
as cards 2.5 or corners 8.5 from being mistaken for the sportsbook's main line
when the full ladder is 4.5/5.5/6.5 or 8.5/9.5/10.5.
"""

try:
    import modules.odds_europa as oe

    _orig_market_type = oe._market_type

    def _market_type_strict(mercado):
        mid = mercado.get("id")
        name = oe.normalizar_nombre(mercado.get("name", ""))

        if mid == 1:
            return "1x2"
        if mid == 5:
            return "goles"
        if mid == 45:
            return "corners"

        # Cards: accept only full-match aggregate totals. Explicitly reject
        # team/player/half/period/handicap derivatives that otherwise contain
        # the word 'card' and contaminate the line ladder with 1.5/2.5 markets.
        cardish = any(token in name for token in ("card", "booking", "tarjeta"))
        if not cardish:
            return None

        rejected = (
            "home", "away", "team", "player", "1st", "first", "2nd", "second",
            "half", "period", "handicap", "asian", "race", "time of", "minute",
        )
        if any(token in name for token in rejected):
            return None

        # Require an aggregate-total style market name whenever the provider
        # supplies descriptive naming. This keeps generic 'total cards' /
        # 'cards over under' while excluding most derivative card props.
        if any(token in name for token in ("total", "over", "under", "cards")):
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
            # Keep price-quality metadata only as a tie breaker between the two
            # central candidates when the ladder has an even number of lines.
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
        if n % 2 == 1:
            chosen = complete[n // 2]
            return chosen[0], f"central_main:{n}"

        left = complete[(n // 2) - 1]
        right = complete[n // 2]
        chosen = min((left, right), key=lambda x: (x[1], x[2], x[0]))
        return chosen[0], f"central_main_even:{n}"

    oe._market_type = _market_type_strict
    oe._balanced_supported_line = _central_supported_line
    oe.CALIBRATION_VERSION = "strict-v8-bet365-main-line-fullmatch-market-lock"
except Exception:
    # Never block application startup because of an optional runtime patch.
    pass
