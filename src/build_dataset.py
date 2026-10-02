"""Build the canonical fight-level and fighter-appearance datasets.

Outputs (data/processed/):
  fights.parquet       one row per fight (completed bouts with a known event date)
  appearances.parquet  two rows per fight: one per fighter, with own + opponent stats
  fighters.parquet     fighter master table (static attributes) incl. synthetic IDs

Run:  python -m src.build_dataset
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.data_loader import fight_duration_seconds, load_raw, normalize_name, parse_round_lengths

# Fight-file spelling -> tott spelling. Each pair was checked by hand (see README):
# same weight class, and either the tott spelling has no fights of its own or the
# two careers do not overlap in time (so this joins one career, never two people).
NAME_ALIASES = {
    "Waldo Cortes Acosta": "Waldo Cortes-Acosta",
    "YiSak Lee": "Yi Sak Lee",
    "Zach Reese": "Zachary Reese",
    "Cam Nelson": "Cameron Nelson",
    "Kai Kamaka III": "Kai Kamaka",
    "Sean King III": "Sean King",
    "Levi Rodrigues Jr.": "Levi Rodrigues",
    "Michael Aswell Jr.": "Michael Aswell",
    "Ilimbek Akylbek": "Ilimbek Akylbek Uulu",
    "Rafael Cerquiera": "Rafael Cerqueira",
    "Muhammad Saidov": "Muhammad Said",
    "Hector Santiago": "Hector de Sousa Santiago",
    "Ben Johnston": "Benjamin Donovan Johnston",
    "Adrian Luna Martinetti": "Juan Adrian Luna",
    "Patricio Pitbull": "Patricio Freire",
    "Mahammadali Osmanli": "Mehemmedeli Osmanli",
}

# Ordered: more specific patterns first.
WEIGHT_CLASSES = [
    ("Women's Strawweight", 115, True), ("Women's Flyweight", 125, True),
    ("Women's Bantamweight", 135, True), ("Women's Featherweight", 145, True),
    ("Light Heavyweight", 205, False), ("Super Heavyweight", np.nan, False),
    ("Heavyweight", 265, False), ("Strawweight", 115, False), ("Flyweight", 125, False),
    ("Bantamweight", 135, False), ("Featherweight", 145, False), ("Lightweight", 155, False),
    ("Welterweight", 170, False), ("Middleweight", 185, False),
    ("Catch Weight", np.nan, False), ("Open Weight", np.nan, False),
]

STAT_COLS = [
    "kd", "sub_att", "rev", "ctrl_s",
    "sig_landed", "sig_att", "tot_landed", "tot_att", "td_landed", "td_att",
    "head_landed", "head_att", "body_landed", "body_att", "leg_landed", "leg_att",
    "dist_landed", "dist_att", "clinch_landed", "clinch_att", "ground_landed", "ground_att",
]


def _log(msg: str) -> None:
    print(f"[build_dataset] {msg}")


def canonical_weight_class(raw: str) -> tuple[str, float, bool, bool]:
    raw = raw or ""
    is_title = bool(re.search(r"title|championship", raw, re.I))
    for name, lbs, female in WEIGHT_CLASSES:
        if name.lower() in raw.lower():
            return name, lbs, female or "women" in raw.lower(), is_title
    return "Other", np.nan, "women" in raw.lower(), is_title


def method_category(raw: str) -> str:
    raw = (raw or "").lower()
    if raw.startswith("ko/tko") or raw.startswith("tko"):
        return "KO/TKO"
    if raw.startswith("submission"):
        return "SUB"
    if raw.startswith("decision"):
        return "DEC"
    if raw == "dq":
        return "DQ"
    return "OTHER"  # Overturned, Could Not Continue, Other


# --------------------------------------------------------------------------- identity
def resolve_fighter_ids(fights: pd.DataFrame, fighters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach fighter_1_id / fighter_2_id. Fight files only carry names.

    1. Apply NAME_ALIASES, then match on a normalised name against tott.
    2. Name shared by several tott fighters: pick the one whose tott weight is
       closest to the bout's weight class, then require a plausible age on the fight
       date. Still ambiguous -> fight-scoped ID (never merge two careers).
    3. No tott match -> synthetic ID 'unk_<normalised name>' (no physical data).
    """
    by_norm = fighters.groupby("name_norm")
    candidates = {k: g for k, g in by_norm}
    report = {"exact_unique": 0, "alias": 0, "disambiguated": 0, "ambiguous_fight_scoped": 0, "unmatched_synthetic": 0}
    synthetic = {}

    def resolve(name: str, fight) -> str:
        key_name = NAME_ALIASES.get(name, name)
        if key_name != name:
            report["alias"] += 1
        cands = candidates.get(normalize_name(key_name))
        if cands is None:
            report["unmatched_synthetic"] += 1
            fid = "unk_" + normalize_name(name)
            synthetic[fid] = name
            return fid
        if len(cands) == 1:
            report["exact_unique"] += key_name == name
            return cands["fighter_id"].iloc[0]
        c = cands.copy()
        age = (fight.event_date - c["dob"]).dt.days / 365.25
        c = c[~(age < 18) & ~(age > 45)]  # NaN DOB is kept
        if not np.isnan(fight.weight_lbs) and c["tott_weight_lbs"].notna().any():
            d = (c["tott_weight_lbs"] - fight.weight_lbs).abs()
            c = c[d == d.min()]
        if len(c) == 1:
            report["disambiguated"] += 1
            return c["fighter_id"].iloc[0]
        report["ambiguous_fight_scoped"] += 1
        fid = f"amb_{normalize_name(name)}_{fight.fight_id}"
        synthetic[fid] = name
        return fid

    ids1, ids2 = [], []
    for f in fights.itertuples(index=False):
        ids1.append(resolve(f.fighter_1_name, f))
        ids2.append(resolve(f.fighter_2_name, f))
    fights = fights.assign(fighter_1_id=ids1, fighter_2_id=ids2)
    _log(f"fighter-name resolution (fighter slots): {report}")
    assert (fights["fighter_1_id"] != fights["fighter_2_id"]).all(), "fighter matched against themself"

    extra = pd.DataFrame({"fighter_id": list(synthetic), "name": list(synthetic.values())})
    extra["name_norm"] = extra["name"].map(normalize_name)
    fighters_all = pd.concat([fighters.assign(synthetic=False), extra.assign(synthetic=True)], ignore_index=True)
    return fights, fighters_all


# --------------------------------------------------------------------------- fights
def build_fights(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    res, ev = raw["results"], raw["events"]
    _log(f"fight_results rows: {len(res)}")

    # Some fights appear twice under a renamed event ("UFC Fight Night: X" vs "Noche UFC: X").
    # Keep the copy whose event name exists in event_details (the one with a date).
    res = res.assign(_has_event=res["event"].isin(ev["event"]))
    res = res.sort_values("_has_event", ascending=False).drop_duplicates("fight_id", keep="first")
    _log(f"after de-duplicating fight URLs: {len(res)}")

    res = res.merge(ev[["event", "event_date"]], on="event", how="left")
    undated = res["event_date"].isna()
    _log(f"dropping {undated.sum()} fights with no event date: {sorted(res.loc[undated, 'event'].unique())}")
    res = res[~undated].copy()

    wc = res["weightclass_raw"].map(canonical_weight_class)
    res["weight_class"] = wc.str[0]
    res["weight_lbs"] = wc.str[1].astype(float)
    res["is_female"] = wc.str[2].astype(bool)
    res["is_title"] = wc.str[3].astype(bool)
    res["method"] = res["method_raw"].map(method_category)
    res["scheduled_rounds"] = res["time_format"].map(lambda t: len(parse_round_lengths(t) or []) or np.nan)
    res["duration_s"] = [
        fight_duration_seconds(tf, r, t) for tf, r, t in zip(res["time_format"], res["end_round"], res["end_time_s"])
    ]
    res["winner_side"] = res["outcome"].map({"W/L": 1, "L/W": 2})  # NaN for draws / no contests
    res["result_type"] = res["outcome"].map({"W/L": "win", "L/W": "win", "D/D": "draw", "NC/NC": "no_contest"})
    _log(f"outcome counts: {res['result_type'].value_counts().to_dict()}")
    return res.drop(columns=["_has_event"]).sort_values(["event_date", "fight_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- stats
def aggregate_fight_stats(stats: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """Sum per-round stats to one row per (fight_id, side)."""
    _log(f"fight_stats rows (with a round): {len(stats)} (raw incl. empty: {stats.attrs.get('n_raw')})")
    keys = fights[["event", "bout", "fight_id", "fighter_1_name", "fighter_2_name"]]
    ambiguous = keys.duplicated(["event", "bout"], keep=False)
    if ambiguous.any():
        _log(f"ignoring stats for {ambiguous.sum()} fights sharing the same (event, bout) key: "
             f"{keys.loc[ambiguous, 'bout'].tolist()}")
    s = stats.merge(keys[~ambiguous], on=["event", "bout"], how="inner")
    _log(f"stats rows joined to kept fights: {len(s)}")
    s["side"] = np.where(s["fighter_name"] == s["fighter_1_name"], 1,
                         np.where(s["fighter_name"] == s["fighter_2_name"], 2, 0))
    assert (s["side"] > 0).all(), "stats row whose fighter is not in the bout"
    dup = s.duplicated(["fight_id", "side", "round"])
    assert not dup.any(), f"duplicate (fight, fighter, round) stats rows: {dup.sum()}"

    g = s.groupby(["fight_id", "side"])
    agg = g[STAT_COLS].sum(min_count=1)
    # Control time is "--" for some early events: unknown if any round is unknown.
    agg["ctrl_s"] = agg["ctrl_s"].where(g["ctrl_s"].apply(lambda x: x.notna().all()))
    return agg.reset_index()


def build_appearances(fights: pd.DataFrame, fight_stats: pd.DataFrame) -> pd.DataFrame:
    """Two rows per fight (one per fighter) with own and opponent stats."""
    base_cols = ["fight_id", "event", "event_date", "weight_class", "weight_lbs", "is_title",
                 "method", "duration_s", "end_round", "scheduled_rounds"]
    rows = []
    for side, opp in [(1, 2), (2, 1)]:
        r = fights[base_cols].copy()
        r["side"] = side
        r["fighter_id"] = fights[f"fighter_{side}_id"]
        r["fighter_name"] = fights[f"fighter_{side}_name"]
        r["opponent_id"] = fights[f"fighter_{opp}_id"]
        r["result"] = np.select(
            [fights["winner_side"] == side, fights["winner_side"] == opp, fights["result_type"] == "draw"],
            ["W", "L", "D"], default="NC",
        )
        rows.append(r)
    app = pd.concat(rows, ignore_index=True)

    own = fight_stats.rename(columns={c: f"{c}" for c in STAT_COLS})
    opp = fight_stats.assign(side=3 - fight_stats["side"]).rename(columns={c: f"opp_{c}" for c in STAT_COLS})
    app = app.merge(own, on=["fight_id", "side"], how="left").merge(opp, on=["fight_id", "side"], how="left")
    app["has_stats"] = app["sig_att"].notna() & app["opp_sig_att"].notna() & app["duration_s"].notna()
    _log(f"appearances: {len(app)} rows ({app['has_stats'].mean():.1%} with round stats)")
    return app.sort_values(["event_date", "fight_id", "side"]).reset_index(drop=True)


def build_all(save: bool = True) -> dict[str, pd.DataFrame]:
    raw = load_raw()
    fights = build_fights(raw)
    fights, fighters = resolve_fighter_ids(fights, raw["fighters"])
    stats = aggregate_fight_stats(raw["stats"], fights)
    app = build_appearances(fights, stats)
    _log(f"fights: {len(fights)} ({fights['event_date'].min().date()} .. {fights['event_date'].max().date()}), "
         f"binary-outcome fights: {fights['winner_side'].notna().sum()}, fighters: {len(fighters)}")
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        fights.to_parquet(PROCESSED_DIR / "fights.parquet", index=False)
        app.to_parquet(PROCESSED_DIR / "appearances.parquet", index=False)
        fighters.to_parquet(PROCESSED_DIR / "fighters.parquet", index=False)
        _log(f"saved to {PROCESSED_DIR}")
    return {"fights": fights, "appearances": app, "fighters": fighters}


if __name__ == "__main__":
    build_all()
