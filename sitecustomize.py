"""Runtime safety patch for FUT Europa sportsbook line selection.

Loaded automatically by Python at startup. This patch pins the supported
PRE-MATCH API-Football market IDs to their canonical full-match markets and
selects the sportsbook main total line from the price ladder rather than from
its numeric position.

Canonical pre-match ids used here:
  1  Match Winner
  5  Goals Over/Under
  45 Corners Over Under
  80 Cards Over/Under

Important: API-Football pre-match odds do not expose a reliable ``main`` flag
for these ladders. The best available proxy for the bookmaker's main line is
the complete Over/Under pair whose no-vig probability is closest to 50/50.
This must only be done AFTER isolating the exact full-match market id; otherwise
team/half/player card derivatives can contaminate the ladder.
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

    def _price_main_line(tipo, price_map):
        candidates = []
        for line, sides in (price_map or {}).items():
            try:
                line = float(line)
                over = float(sides.get("Over", 0))
                under = float(sides.get("Under", 0))
            except Exception:
                continue
            if over <= 1.01 or under <= 1.01 or not oe._supported_line(tipo, line):
                continue

            io = 1.0 / over
            iu = 1.0 / under
            total = io + iu
            if total <= 0:
                continue
            p_over_no_vig = io / total
            balance = abs(p_over_no_vig - 0.5)
            # Tie-breakers prefer the tighter two-way book, then the smaller
            # absolute price asymmetry. Numeric line itself is only last.
            overround = abs(total - 1.0)
            symmetry = abs(over - under)
            candidates.append((
                round(balance, 10),
                round(overround, 10),
                round(symmetry, 10),
                line,
            ))

        if not candidates:
            return None, "missing_supported"
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        best = candidates[0]
        return best[3], f"price_main_no_vig:{len(candidates)}"

    oe._market_type = _market_type_strict
    oe._balanced_supported_line = _price_main_line
    oe.CALIBRATION_VERSION = "strict-v10-bet365-canonical-ids-price-main-line-lock"
except Exception:
    pass
