"""Glicko-1 rating mechanics (leakage is covered by tests/test_leakage.py, which checks
every profile feature, including glicko_rating / glicko_rd)."""
import pandas as pd
import pytest
from conftest import make_appearances

from src.ratings import INITIAL_RATING, INITIAL_RD, MIN_RD, compute_glicko_ratings


def _by(g, fight, fighter):
    row = g[(g["fight_id"] == fight) & (g["fighter_id"] == fighter)].iloc[0]
    return row["glicko_rating"], row["glicko_rd"]


def test_debutants_start_at_defaults_and_winner_rises():
    app = make_appearances([
        ("f1", "x", "2020-01-01", "W", "DEC", 30), ("f1", "y", "2020-01-01", "L", "DEC", 10),
        ("f2", "x", "2020-02-01", "W", "DEC", 30), ("f2", "z", "2020-02-01", "L", "DEC", 10),
    ])
    g = compute_glicko_ratings(app)
    assert _by(g, "f1", "x") == (INITIAL_RATING, INITIAL_RD)  # pre-fight value of a debut
    r_after, rd_after = _by(g, "f2", "x")
    assert r_after > INITIAL_RATING  # beat y
    assert MIN_RD <= rd_after < INITIAL_RD  # more certain after a fight (small layoff)


def test_rd_grows_with_layoff():
    rows = [("f1", "x", "2020-01-01", "W", "DEC", 30), ("f1", "y", "2020-01-01", "L", "DEC", 10),
            ("f2", "x", "2020-03-01", "W", "DEC", 30), ("f2", "a", "2020-03-01", "L", "DEC", 10),
            ("f3", "y", "2020-03-01", "W", "DEC", 30), ("f3", "b", "2020-03-01", "L", "DEC", 10),
            ("f4", "y", "2024-03-01", "W", "DEC", 30), ("f4", "c", "2024-03-01", "L", "DEC", 10)]
    g = compute_glicko_ratings(make_appearances(rows))
    _, rd_short = _by(g, "f2", "x")   # 2 months after first fight
    _, rd_long = _by(g, "f4", "y")    # 4 years after last fight
    assert rd_long > rd_short


def test_draws_split_and_no_contests_are_ignored():
    rows = [("f1", "x", "2020-01-01", "D", "DEC", 30), ("f1", "y", "2020-01-01", "D", "DEC", 30),
            ("f2", "x", "2020-02-01", "NC", "OTHER", 30), ("f2", "y", "2020-02-01", "NC", "OTHER", 30),
            ("f3", "x", "2020-03-01", "W", "DEC", 30), ("f3", "y", "2020-03-01", "L", "DEC", 30)]
    g = compute_glicko_ratings(make_appearances(rows))
    rx, _ = _by(g, "f2", "x")
    ry, _ = _by(g, "f2", "y")
    assert rx == pytest.approx(INITIAL_RATING) and ry == pytest.approx(INITIAL_RATING)  # equal debuts drew
    # The no contest changed nothing: f3's pre-fight ratings equal f2's.
    assert _by(g, "f3", "x")[0] == pytest.approx(rx) and _by(g, "f3", "y")[0] == pytest.approx(ry)


def test_same_night_bouts_share_the_pre_card_rating():
    rows = [("f1", "x", "2020-01-01", "W", "DEC", 30), ("f1", "y", "2020-01-01", "L", "DEC", 10),
            ("f2", "x", "2020-01-01", "W", "DEC", 30), ("f2", "z", "2020-01-01", "L", "DEC", 10)]
    g = compute_glicko_ratings(make_appearances(rows))
    assert _by(g, "f1", "x") == _by(g, "f2", "x")  # tournament night: second bout can't see the first


def test_output_is_row_aligned_with_input():
    app = make_appearances([("f1", "x", "2020-05-01", "W", "DEC", 30), ("f1", "y", "2020-05-01", "L", "DEC", 10),
                            ("f0", "x", "2020-01-01", "L", "DEC", 30), ("f0", "y", "2020-01-01", "W", "DEC", 10)])
    g = compute_glicko_ratings(app)
    assert list(g["fight_id"]) == list(app["fight_id"])
    assert list(g["fighter_id"]) == list(app["fighter_id"])
    assert _by(g, "f1", "y")[0] > _by(g, "f1", "x")[0]  # y won the earlier fight (input not date-sorted)
