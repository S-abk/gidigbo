"""Leakage-safe pre-fight fighter features and matchup (difference) features.

Core guarantee
--------------
`compute_prefight_features` returns, for every fighter appearance, features built
ONLY from that fighter's appearances on strictly earlier event dates. Fights on the
same date are excluded too: early UFC tournaments had several bouts per fighter in
one night and the bout order within a night is not recorded, so we cannot know
which one came first.

Training and inference share this exact code: to profile a fighter "as of today",
`fighter_profiles_as_of` appends a placeholder appearance dated after every real
fight and runs the same function (see predict.py).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.ratings import compute_glicko_ratings

# Shrinkage priors: long-run UFC averages computed once from 2001-2021 appearances
# (training period only), rounded. A fighter's rate is pulled toward these values
# in proportion to how little data they have, so debutants get league-average
# rates instead of NaN or extreme values.
PRIOR_SLPM = 3.3            # significant strikes landed per minute
PRIOR_SIG_ACC = 0.44        # significant strike accuracy
PRIOR_TD15 = 1.5            # takedowns landed per 15 min
PRIOR_TD_ACC = 0.38         # takedown accuracy
PRIOR_SUB15 = 0.55          # submission attempts per 15 min
PRIOR_KD15 = 0.3            # knockdowns per 15 min
PRIOR_CTRL = 0.21           # fraction of fight time in control
PRIOR_MINUTES = 15.0        # pseudo-minutes of league-average exposure
PRIOR_STRIKE_ATT = 40.0     # pseudo-attempts for accuracy / defence
PRIOR_TD_ATT = 6.0          # pseudo-attempts for takedown accuracy / defence
PRIOR_FIGHTS = 2.0          # pseudo-fights for win-rate style proportions

DEFAULT_AGE = 30.0          # used only when DOB is unknown (no missingness flag: see README)
REACH_MINUS_HEIGHT = 2.0    # median reach - height (inches); an integer, so imputed reach looks like real reach

WINDOWS = (3, 5)

# Per-appearance quantities that are summed over a history window.
_COUNT_COLS = ["n", "w", "l", "ko_w", "sub_w", "dec_w", "fin_l", "minutes"]
_STAT_COLS = ["s_min", "sig_l", "sig_a", "osig_l", "osig_a", "td_l", "td_a", "otd_l", "otd_a",
              "sub", "kd", "okd", "ctrl", "ctrl_min"]
_SUM_COLS = _COUNT_COLS + _STAT_COLS

# Features that are differenced (A - B) into matchup features. Grouped for explanations.
FEATURE_GROUPS: dict[str, list[str]] = {
    "opponent-strength rating": ["glicko_rating", "glicko_rd"],
    "UFC experience": ["n_fights", "win_pct"],
    "recent form": ["wins_last3", "wins_last5", "win_pct_last5", "streak"],
    "age": ["age"],
    "height": ["height_in"],
    "reach": ["reach_in"],
    "stance": ["southpaw", "switch"],
    "activity (days since last fight)": ["days_since_last"],
    "striking output": ["slpm", "slpm_last3", "slpm_last5", "kd15"],
    "striking defence / damage absorbed": ["sapm", "sapm_last3", "sapm_last5", "sig_def"],
    "striking differential": ["sig_diff_pm", "sig_diff_pm_last3", "sig_diff_pm_last5"],
    "striking accuracy": ["sig_acc"],
    "wrestling offence": ["td15", "td15_last5", "td_acc", "ctrl_pct", "ctrl_pct_last5"],
    "takedown defence": ["td_def"],
    "submission threat": ["sub15", "sub_win_share"],
    "finishing / durability": ["finish_rate", "ko_win_share", "dec_win_share", "fin_loss_rate", "avg_fight_min"],
}
PROFILE_FEATURES: list[str] = [f for fs in FEATURE_GROUPS.values() for f in fs]
DIFF_FEATURES: list[str] = [f"{f}_diff" for f in PROFILE_FEATURES]
# Symmetric context features (identical when A and B are swapped).
CONTEXT_FEATURES: list[str] = ["min_prior_fights"]
MODEL_FEATURES: list[str] = DIFF_FEATURES + CONTEXT_FEATURES


# --------------------------------------------------------------------------- per-appearance
def _appearance_quantities(app: pd.DataFrame) -> pd.DataFrame:
    """Numeric per-appearance quantities. Stats count only when the fight has round stats."""
    q = pd.DataFrame(index=app.index)
    w, l = app["result"].eq("W"), app["result"].eq("L")
    finish = app["method"].isin(["KO/TKO", "SUB"])
    q["n"] = 1.0
    q["w"], q["l"] = w.astype(float), l.astype(float)
    q["ko_w"] = (w & app["method"].eq("KO/TKO")).astype(float)
    q["sub_w"] = (w & app["method"].eq("SUB")).astype(float)
    q["dec_w"] = (w & app["method"].eq("DEC")).astype(float)
    q["fin_l"] = (l & finish).astype(float)
    q["minutes"] = (app["duration_s"] / 60).fillna(0.0)

    hs = app["has_stats"].fillna(False).astype(bool)
    def s(col):  # stat only counts when the fight has full stats
        return app[col].where(hs, 0.0).fillna(0.0)
    q["s_min"] = (app["duration_s"] / 60).where(hs, 0.0).fillna(0.0)
    q["sig_l"], q["sig_a"] = s("sig_landed"), s("sig_att")
    q["osig_l"], q["osig_a"] = s("opp_sig_landed"), s("opp_sig_att")
    q["td_l"], q["td_a"] = s("td_landed"), s("td_att")
    q["otd_l"], q["otd_a"] = s("opp_td_landed"), s("opp_td_att")
    q["sub"], q["kd"], q["okd"] = s("sub_att"), s("kd"), s("opp_kd")
    has_ctrl = hs & app["ctrl_s"].notna()
    q["ctrl"] = app["ctrl_s"].where(has_ctrl, 0.0).fillna(0.0)
    q["ctrl_min"] = (app["duration_s"] / 60).where(has_ctrl, 0.0).fillna(0.0)
    return q


def _streak_after_each_day(app: pd.DataFrame) -> pd.Series:
    """Signed streak (+wins / -losses) after each (fighter, date). Draw/NC resets to 0."""
    out = {}
    for fid, g in app.sort_values(["fighter_id", "event_date", "fight_id"]).groupby("fighter_id", sort=False):
        streak = 0
        for date, res in zip(g["event_date"], g["result"]):
            if res == "W":
                streak = streak + 1 if streak > 0 else 1
            elif res == "L":
                streak = streak - 1 if streak < 0 else -1
            elif res in ("D", "NC"):
                streak = 0
            out[(fid, date)] = streak  # last write per day wins
    return pd.Series(out, name="streak_after")


def _shrunk(num, den, prior, k):
    return (num + prior * k) / (den + k)


def _ratios(s: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Rate features from summed history columns `s` (one window)."""
    f = pd.DataFrame(index=s.index)
    m = s["s_min"]
    slpm = _shrunk(s["sig_l"], m, PRIOR_SLPM, PRIOR_MINUTES)
    sapm = _shrunk(s["osig_l"], m, PRIOR_SLPM, PRIOR_MINUTES)
    f[f"slpm{suffix}"] = slpm
    f[f"sapm{suffix}"] = sapm
    f[f"sig_diff_pm{suffix}"] = slpm - sapm
    f[f"td15{suffix}"] = _shrunk(s["td_l"], m, PRIOR_TD15 / 15, PRIOR_MINUTES) * 15
    f[f"ctrl_pct{suffix}"] = _shrunk(s["ctrl"] / 60, s["ctrl_min"], PRIOR_CTRL, PRIOR_MINUTES)
    f[f"win_pct{suffix}"] = _shrunk(s["w"], s["n"], 0.5, PRIOR_FIGHTS)
    return f


# --------------------------------------------------------------------------- main entry
def compute_prefight_features(appearances: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """Pre-fight profile for every appearance, keyed by (fight_id, fighter_id).

    `appearances` needs: fight_id, fighter_id, event_date, result, method,
    duration_s, has_stats and the raw stat columns from build_dataset.
    """
    app = appearances.reset_index(drop=True)
    q = _appearance_quantities(app)
    q["fighter_id"], q["event_date"] = app["fighter_id"].values, app["event_date"].values

    # One row per (fighter, event date): tournament nights collapse together.
    day = q.groupby(["fighter_id", "event_date"], sort=True)[_SUM_COLS].sum().reset_index()
    g = day.groupby("fighter_id", sort=False)

    # Exclusive cumulative sums = totals over strictly earlier dates.
    cum_incl = g[_SUM_COLS].cumsum()
    career = cum_incl - day[_SUM_COLS]
    prior_hist = {"career": career}
    for k in WINDOWS:
        # sum over the previous k fight-days = excl_cum(t) - excl_cum(t-k)
        lagged = career.groupby(day["fighter_id"], sort=False).shift(k).fillna(0.0)
        prior_hist[f"last{k}"] = career - lagged

    feat = pd.DataFrame({"fighter_id": day["fighter_id"], "event_date": day["event_date"]})
    c = career
    feat["n_fights"] = c["n"]
    feat["wins"], feat["losses"] = c["w"], c["l"]
    feat["draws_nc"] = c["n"] - c["w"] - c["l"]
    feat = feat.join(_ratios(c))
    feat["wins_last3"] = prior_hist["last3"]["w"]
    feat["wins_last5"] = prior_hist["last5"]["w"]
    feat["win_pct_last5"] = _shrunk(prior_hist["last5"]["w"], prior_hist["last5"]["n"], 0.5, PRIOR_FIGHTS)
    for k in WINDOWS:
        r = _ratios(prior_hist[f"last{k}"], f"_last{k}")
        feat = feat.join(r[[col for col in r.columns if not col.startswith("win_pct")]])

    feat["sig_acc"] = _shrunk(c["sig_l"], c["sig_a"], PRIOR_SIG_ACC, PRIOR_STRIKE_ATT)
    feat["sig_def"] = 1 - _shrunk(c["osig_l"], c["osig_a"], PRIOR_SIG_ACC, PRIOR_STRIKE_ATT)
    feat["td_acc"] = _shrunk(c["td_l"], c["td_a"], PRIOR_TD_ACC, PRIOR_TD_ATT)
    feat["td_def"] = 1 - _shrunk(c["otd_l"], c["otd_a"], PRIOR_TD_ACC, PRIOR_TD_ATT)
    feat["sub15"] = _shrunk(c["sub"], c["s_min"], PRIOR_SUB15 / 15, PRIOR_MINUTES) * 15
    feat["kd15"] = _shrunk(c["kd"], c["s_min"], PRIOR_KD15 / 15, PRIOR_MINUTES) * 15

    # Finish profile (proportions shrunk toward neutral-ish shares).
    feat["finish_rate"] = _shrunk(c["ko_w"] + c["sub_w"], c["w"], 0.5, PRIOR_FIGHTS)
    feat["ko_win_share"] = _shrunk(c["ko_w"], c["w"], 0.33, PRIOR_FIGHTS)
    feat["sub_win_share"] = _shrunk(c["sub_w"], c["w"], 0.2, PRIOR_FIGHTS)
    feat["dec_win_share"] = _shrunk(c["dec_w"], c["w"], 0.45, PRIOR_FIGHTS)
    feat["fin_loss_rate"] = _shrunk(c["fin_l"], c["n"], 0.15, PRIOR_FIGHTS)
    feat["avg_fight_min"] = _shrunk(c["minutes"], c["n"], 10.5, 1.0)

    # Streak and layoff, from the previous fight-day only.
    streak_after = _streak_after_each_day(app)
    feat["streak"] = (
        pd.Series(streak_after.reindex(pd.MultiIndex.from_frame(day[["fighter_id", "event_date"]])).values)
        .groupby(day["fighter_id"].values).shift(1).fillna(0.0).values
    )
    prev_date = g["event_date"].shift(1)
    feat["days_since_last"] = (day["event_date"] - prev_date).dt.days.astype(float)  # NaN for debut

    # Static attributes (tott snapshot). Missingness is NOT exposed to the model: it
    # correlates with short careers (future information). See README "Leakage".
    fx = fighters.set_index("fighter_id")
    dob = feat["fighter_id"].map(fx["dob"])
    feat["age"] = ((feat["event_date"] - dob).dt.days / 365.25).fillna(DEFAULT_AGE)
    feat["height_in"] = feat["fighter_id"].map(fx["height_in"]).astype(float)
    reach = feat["fighter_id"].map(fx["reach_in"]).astype(float)
    feat["reach_in"] = reach.fillna(feat["height_in"] + REACH_MINUS_HEIGHT)
    stance = feat["fighter_id"].map(fx["stance"]).fillna("")
    feat["southpaw"] = stance.eq("Southpaw").astype(float)
    feat["switch"] = stance.eq("Switch").astype(float)
    feat["stance"] = stance.replace("", "Unknown")
    feat["dob_known"] = dob.notna()  # display only, never a model feature
    feat["reach_known"] = reach.notna()

    # Broadcast fighter-day rows back to individual appearances.
    out = app[["fight_id", "fighter_id", "event_date"]].merge(feat, on=["fighter_id", "event_date"], how="left")
    assert len(out) == len(app)

    # Opponent-strength rating (src/ratings.py): same strictly-earlier-dates rule,
    # computed row-aligned with `app` (a left merge keeps the left row order).
    ratings = compute_glicko_ratings(app)
    assert (ratings["fight_id"].values == out["fight_id"].values).all()
    out["glicko_rating"] = ratings["glicko_rating"].values
    out["glicko_rd"] = ratings["glicko_rd"].values
    return out


# --------------------------------------------------------------------------- matchups
def matchup_features(prof_a: pd.DataFrame, prof_b: pd.DataFrame) -> pd.DataFrame:
    """Model input from two aligned pre-fight profiles (same row order).

    Swapping A and B negates every *_diff column and leaves context columns unchanged.
    """
    a = prof_a.reset_index(drop=True)
    b = prof_b.reset_index(drop=True)
    X = pd.DataFrame({f"{f}_diff": a[f].astype(float) - b[f].astype(float) for f in PROFILE_FEATURES})
    X["days_since_last_diff"] = X["days_since_last_diff"].clip(-1500, 1500)
    # Remaining NaNs (debutant layoff, both heights unknown) become a neutral 0
    # difference. Using 0 rather than a flag keeps missingness invisible to the model.
    X = X.fillna(0.0)
    X["min_prior_fights"] = np.minimum(a["n_fights"].values, b["n_fights"].values).astype(float)
    return X[MODEL_FEATURES]


def orientation_flip(fight_ids: pd.Series) -> np.ndarray:
    """Deterministic pseudo-random coin per fight (stable across rebuilds)."""
    return np.array([int(hashlib.md5(f.encode()).hexdigest(), 16) % 2 == 1 for f in fight_ids])


def build_training_table(fights: pd.DataFrame, appearances: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """One row per binary-outcome fight with A/B randomly (but deterministically) assigned."""
    prof = compute_prefight_features(appearances, fighters).set_index(["fight_id", "fighter_id"])
    f = fights[fights["winner_side"].notna()].reset_index(drop=True)
    flip = orientation_flip(f["fight_id"])
    a_id = np.where(flip, f["fighter_2_id"], f["fighter_1_id"])
    b_id = np.where(flip, f["fighter_1_id"], f["fighter_2_id"])
    a_won = np.where(flip, f["winner_side"] == 2, f["winner_side"] == 1).astype(int)
    pa = prof.loc[list(zip(f["fight_id"], a_id))].reset_index()
    pb = prof.loc[list(zip(f["fight_id"], b_id))].reset_index()
    X = matchup_features(pa, pb)
    meta = pd.DataFrame({
        "fight_id": f["fight_id"], "event": f["event"], "event_date": f["event_date"],
        "weight_class": f["weight_class"], "fighter_a_id": a_id, "fighter_b_id": b_id,
        "fighter_a_name": np.where(flip, f["fighter_2_name"], f["fighter_1_name"]),
        "fighter_b_name": np.where(flip, f["fighter_1_name"], f["fighter_2_name"]),
        "a_prior_fights": pa["n_fights"].values, "b_prior_fights": pb["n_fights"].values,
        "label": a_won,
    })
    return pd.concat([meta, X], axis=1)


def mirror(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Append the B-vs-A copy of every row (diffs negated, label flipped)."""
    Xm = X.copy()
    Xm[DIFF_FEATURES] = -Xm[DIFF_FEATURES]
    return pd.concat([X, Xm], ignore_index=True), np.concatenate([y, 1 - y])
