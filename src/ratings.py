"""Leakage-safe Glicko-1 opponent-strength ratings.

Why: the rolling features describe what a fighter did, never *who they did it against*.
A Glicko rating pools information across every fight in the UFC network, so beating
strong opposition counts for more than beating weak opposition. Glicko's rating
deviation (RD) is an explicit uncertainty that grows during layoffs and starts wide
for debutants -- exactly the thin / irregular histories MMA has.

How (Glickman, "The Glicko system", glicko.net/glicko/glicko.pdf):
- A rating period is one event date across the whole dataset. Every fighter on that
  date is snapshotted *before* any of that date's results are applied, then all of
  that date's results are applied as one batch update. So same-night tournament bouts
  never see each other, matching the rule used by every other feature.
- The value attached to an appearance is the snapshot *entering* that fight: it only
  reflects fights on strictly earlier dates (plus RD growth for the layoff).
- Wins score 1, losses 0, draws 0.5. No contests carry no skill information and are
  not applied (the fighter's RD keeps growing as if inactive).
- Rows with any other result (e.g. the 'NA' placeholder used for predictions) are
  snapshotted but never applied, which is how inference shares this exact code.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

INITIAL_RATING = 1500.0
INITIAL_RD = 350.0    # a brand-new fighter's uncertainty (Glicko's conventional maximum)
MIN_RD = 30.0         # floor so very active fighters never become "certain"
INACTIVITY_DAYS_TO_RESET = 1825
# RD growth per day of inactivity, chosen so that ~5 years out of the cage takes even a
# fully established rating (RD = MIN_RD) back to a debutant's uncertainty:
#   INITIAL_RD^2 = MIN_RD^2 + C^2 * INACTIVITY_DAYS_TO_RESET
# Tuned by logistic-regression walk-forward CV on the TRAINING period only (2017-21
# folds; validation and test untouched): no Glicko 0.6612, 2y reset 0.6570, 5y 0.6545,
# 10y 0.6545, no layoff growth 0.6563. A 2-year reset kept almost every pre-fight RD
# near the 350 maximum, because one fight is little information against a long layoff.
C = math.sqrt((INITIAL_RD ** 2 - MIN_RD ** 2) / INACTIVITY_DAYS_TO_RESET)
Q = math.log(10) / 400
SCORES = {"W": 1.0, "L": 0.0, "D": 0.5}


def _g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / (math.pi * math.pi))


def _opponents(app: pd.DataFrame) -> pd.Series:
    """Opponent fighter_id per row: the opponent_id column if present, else the other
    row of the same two-row fight (lets small synthetic test tables work too)."""
    if "opponent_id" in app.columns:
        return app["opponent_id"]
    pair = app.groupby("fight_id")["fighter_id"].transform("count") == 2
    two = app.loc[pair, ["fight_id", "fighter_id"]].reset_index()
    m = two.merge(two, on="fight_id", suffixes=("", "_opp"))
    m = m[m["fighter_id"] != m["fighter_id_opp"]].set_index("index")["fighter_id_opp"]
    return m.reindex(app.index)


def compute_glicko_ratings(appearances: pd.DataFrame) -> pd.DataFrame:
    """Pre-fight (glicko_rating, glicko_rd) for every row of `appearances`, same row order."""
    app = appearances.reset_index(drop=True)
    n = len(app)
    days = app["event_date"].values.astype("datetime64[D]").astype(np.int64)
    fids = app["fighter_id"].to_numpy(dtype=object)
    opps = _opponents(app).to_numpy(dtype=object)
    scores = app["result"].map(SCORES).to_numpy(dtype=float)  # NaN = not applied

    rating: dict[str, float] = {}
    rd: dict[str, float] = {}
    last_day: dict[str, int] = {}

    def snapshot(fid: str, day: int) -> tuple[float, float]:
        if fid not in rating:
            return INITIAL_RATING, INITIAL_RD
        grown = math.sqrt(rd[fid] ** 2 + C * C * max(day - last_day[fid], 0))
        return rating[fid], min(INITIAL_RD, grown)

    pre_r = np.empty(n)
    pre_rd = np.empty(n)
    order = np.argsort(days, kind="stable")
    i = 0
    while i < n:
        day = days[order[i]]
        j = i
        while j < n and days[order[j]] == day:
            j += 1
        idx = order[i:j]

        # 1. snapshot everyone involved today, before applying any of today's results
        snap: dict[str, tuple[float, float]] = {}
        for k in idx:
            for fid in (fids[k], opps[k]):
                if isinstance(fid, str) and fid not in snap:
                    snap[fid] = snapshot(fid, day)
        for k in idx:
            pre_r[k], pre_rd[k] = snap[fids[k]]

        # 2. one batch update per fighter from today's results (Glicko-1 period update)
        acc: dict[str, list[float]] = {}
        for k in idx:
            s, opp = scores[k], opps[k]
            if np.isnan(s) or not isinstance(opp, str):
                continue
            r, _ = snap[fids[k]]
            r_opp, rd_opp = snap[opp]
            g = _g(rd_opp)
            e = 1.0 / (1.0 + 10.0 ** (-g * (r - r_opp) / 400.0))
            a = acc.setdefault(fids[k], [0.0, 0.0])
            a[0] += g * g * e * (1.0 - e)
            a[1] += g * (s - e)
        for fid, (info, surprise) in acc.items():
            r, rd_pre = snap[fid]
            denom = 1.0 / (rd_pre * rd_pre) + Q * Q * info
            rating[fid] = r + (Q / denom) * surprise
            rd[fid] = max(MIN_RD, math.sqrt(1.0 / denom))
            last_day[fid] = day
        i = j

    return pd.DataFrame({"fight_id": app["fight_id"].values, "fighter_id": fids,
                         "glicko_rating": pre_r, "glicko_rd": pre_rd})
