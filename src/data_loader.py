"""Load and normalise the raw UFCStats CSV files.

Source: https://github.com/Greco1899/scrape_ufc_stats (cloned into data/raw/scrape_ufc_stats).

Field provenance
----------------
ufc_event_details.csv   EVENT, URL (event), DATE, LOCATION
    -> event name and **event date** (the only date source for fights).
ufc_fight_details.csv   EVENT, BOUT, URL (fight)
    -> list of bouts per event. Redundant with fight_results (same URLs); not needed.
ufc_fight_results.csv   EVENT, BOUT, OUTCOME, WEIGHTCLASS, METHOD, ROUND, TIME,
                        TIME FORMAT, REFEREE, DETAILS, URL (fight)
    -> one row per fight: the two fighter *names* (BOUT = "A vs. B"), the result
       (OUTCOME = "W/L", "L/W", "D/D" or "NC/NC", aligned with BOUT order),
       method, finishing round/time and the scheduled round structure.
ufc_fight_stats.csv     EVENT, BOUT, ROUND, FIGHTER, KD, SIG.STR., SIG.STR. %, TOTAL STR.,
                        TD, TD %, SUB.ATT, REV., CTRL, HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND
    -> one row per fighter per round. No fight URL and no fighter URL: rows are
       keyed by (EVENT, BOUT) and the fighter's *name*.
ufc_fighter_details.csv FIRST, LAST, NICKNAME, URL (fighter)
    -> fighter list; tott contains the same URLs plus physical data, so only tott is used.
ufc_fighter_tott.csv    FIGHTER, HEIGHT, WEIGHT, REACH, STANCE, DOB, URL (fighter)
    -> "tale of the tape": a *current snapshot* of each fighter. HEIGHT, REACH,
       STANCE and DOB are treated as static attributes. WEIGHT is the fighter's
       current division and is deliberately NOT used as a feature.

Missing values: the scrape uses "--" / "---" for unknown values. They become NaN,
never 0, except where 0 genuinely means zero (e.g. "0 of 0" strikes landed).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from src.config import RAW_DIR

MISSING_TOKENS = ["--", "---", ""]


# --------------------------------------------------------------------------- parsers
def normalize_name(name: str) -> str:
    """Lower-case ASCII letters only; used for tolerant fighter name matching."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def parse_height_inches(s: pd.Series) -> pd.Series:
    """'5\\' 11"' -> 71.0; '--' -> NaN."""
    m = s.astype("string").str.extract(r"(\d+)'\s*(\d+)")
    return m[0].astype(float) * 12 + m[1].astype(float)


def parse_inches(s: pd.Series) -> pd.Series:
    """'72"' -> 72.0; '--' -> NaN."""
    return s.astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)


def parse_mmss_seconds(s: pd.Series) -> pd.Series:
    """'3:44' -> 224.0; '--' -> NaN."""
    m = s.astype("string").str.extract(r"^\s*(\d+):(\d+)\s*$")
    return m[0].astype(float) * 60 + m[1].astype(float)


def parse_landed_of(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'29 of 73' -> (29.0, 73.0)."""
    m = s.astype("string").str.extract(r"(\d+)\s+of\s+(\d+)")
    return m[0].astype(float), m[1].astype(float)


def parse_round_lengths(time_format: str) -> list[float] | None:
    """Round lengths in minutes from TIME FORMAT.

    '3 Rnd (5-5-5)' -> [5, 5, 5]; '1 Rnd + OT (12-3)' -> [12, 3];
    '1 Rnd (20)' -> [20]; 'No Time Limit' -> None (unknown schedule).
    """
    if not isinstance(time_format, str):
        return None
    m = re.search(r"\(([\d\-]+)\)", time_format)
    if not m:
        return None
    return [float(x) for x in m.group(1).split("-") if x]


def fight_duration_seconds(time_format: str, end_round: int, end_time_s: float) -> float:
    """Elapsed fight time = full length of completed rounds + time in the final round."""
    if pd.isna(end_round) or pd.isna(end_time_s):
        return np.nan
    lengths = parse_round_lengths(time_format)
    end_round = int(end_round)
    if lengths is None:  # "No Time Limit": single continuous period
        return end_time_s if end_round == 1 else np.nan
    if end_round > len(lengths):
        return np.nan
    return sum(lengths[: end_round - 1]) * 60 + end_time_s


# --------------------------------------------------------------------------- loaders
def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / name, na_values=MISSING_TOKENS, keep_default_na=True)
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
    return df


def load_events() -> pd.DataFrame:
    ev = _read("ufc_event_details.csv")
    ev["event_date"] = pd.to_datetime(ev["DATE"], format="%B %d, %Y")
    return ev.rename(columns={"EVENT": "event", "URL": "event_url", "LOCATION": "location"})[
        ["event", "event_url", "event_date", "location"]
    ]


def load_fight_results() -> pd.DataFrame:
    fr = _read("ufc_fight_results.csv")
    sides = fr["BOUT"].str.split(" vs. ", n=1, expand=True)
    out = pd.DataFrame(
        {
            "event": fr["EVENT"],
            "bout": fr["BOUT"],
            "fighter_1_name": sides[0].str.strip(),
            "fighter_2_name": sides[1].str.strip(),
            "outcome": fr["OUTCOME"],
            "weightclass_raw": fr["WEIGHTCLASS"],
            "method_raw": fr["METHOD"],
            "end_round": pd.to_numeric(fr["ROUND"], errors="coerce"),
            "end_time_s": parse_mmss_seconds(fr["TIME"]),
            "time_format": fr["TIME FORMAT"],
            "details": fr["DETAILS"],
            "fight_url": fr["URL"],
        }
    )
    out["fight_id"] = out["fight_url"].str.rsplit("/", n=1).str[-1]
    return out


def load_fight_stats() -> pd.DataFrame:
    """Per fighter per round stats with numeric columns.

    Rows with no ROUND (fights with no recorded stats) are dropped here.
    """
    fs = _read("ufc_fight_stats.csv")
    n_raw = len(fs)
    fs = fs.dropna(subset=["ROUND", "FIGHTER"]).copy()
    out = pd.DataFrame(
        {
            "event": fs["EVENT"],
            "bout": fs["BOUT"],
            "round": fs["ROUND"].str.extract(r"(\d+)")[0].astype(int),
            "fighter_name": fs["FIGHTER"],
            "kd": pd.to_numeric(fs["KD"], errors="coerce"),
            "sub_att": pd.to_numeric(fs["SUB.ATT"], errors="coerce"),
            "rev": pd.to_numeric(fs["REV."], errors="coerce"),
            "ctrl_s": parse_mmss_seconds(fs["CTRL"]),  # NaN for "--" (not recorded, early events)
        }
    )
    for col, name in [
        ("SIG.STR.", "sig"), ("TOTAL STR.", "tot"), ("TD", "td"),
        ("HEAD", "head"), ("BODY", "body"), ("LEG", "leg"),
        ("DISTANCE", "dist"), ("CLINCH", "clinch"), ("GROUND", "ground"),
    ]:
        out[f"{name}_landed"], out[f"{name}_att"] = parse_landed_of(fs[col])
    out.attrs["n_raw"] = n_raw
    return out


def load_fighters() -> pd.DataFrame:
    """Fighter master table from tott (a current snapshot; static attributes only)."""
    t = _read("ufc_fighter_tott.csv")
    out = pd.DataFrame(
        {
            "fighter_id": t["URL"].str.rsplit("/", n=1).str[-1],
            "name": t["FIGHTER"],
            "height_in": parse_height_inches(t["HEIGHT"]),
            "reach_in": parse_inches(t["REACH"]),
            "tott_weight_lbs": parse_inches(t["WEIGHT"]),  # identity resolution only, NOT a feature
            "stance": t["STANCE"],
            "dob": pd.to_datetime(t["DOB"], format="%b %d, %Y", errors="coerce"),
            "fighter_url": t["URL"],
        }
    )
    out["name_norm"] = out["name"].map(normalize_name)
    return out


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "events": load_events(),
        "results": load_fight_results(),
        "stats": load_fight_stats(),
        "fighters": load_fighters(),
    }
