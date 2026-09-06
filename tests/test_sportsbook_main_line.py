import importlib

import modules.odds_europa as oe

# Ensure runtime safety overrides are installed in test environments too.
import sitecustomize  # noqa: F401
importlib.invalidate_caches()


def test_canonical_prematch_market_ids():
    assert oe._market_type({"id": 1, "name": "Match Winner"}) == "1x2"
    assert oe._market_type({"id": 5, "name": "Goals Over/Under"}) == "goles"
    assert oe._market_type({"id": 45, "name": "Corners Over Under"}) == "corners"
    assert oe._market_type({"id": 80, "name": "Cards Over/Under"}) == "tarjetas"
    assert oe._market_type({"id": 82, "name": "Home Team Total Cards"}) is None
    assert oe._market_type({"id": 83, "name": "Away Team Total Cards"}) is None


def test_cards_main_line_uses_price_not_lowest_line():
    ladder = {
        2.5: {"Over": 1.18, "Under": 4.60},
        4.5: {"Over": 1.48, "Under": 2.50},
        5.5: {"Over": 1.91, "Under": 1.91},
        6.5: {"Over": 2.60, "Under": 1.45},
    }
    line, status = oe._balanced_supported_line("tarjetas", ladder)
    assert line == 5.5
    assert status.startswith("price_main_no_vig")


def test_corners_main_line_uses_price_not_lowest_line():
    ladder = {
        8.5: {"Over": 1.50, "Under": 2.45},
        9.5: {"Over": 1.90, "Under": 1.90},
        10.5: {"Over": 2.35, "Under": 1.55},
    }
    line, status = oe._balanced_supported_line("corners", ladder)
    assert line == 9.5
    assert status.startswith("price_main_no_vig")
