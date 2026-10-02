"""Predict hypothetical fights between two fighters.

Usage:
    from src.predict import predict_fight
    result = predict_fight("Islam Makhachev", "Charles Oliveira")

Profiles are rebuilt by the same `compute_prefight_features` used in training: each
fighter gets a placeholder appearance dated after every recorded fight, so their
profile includes every completed fight and nothing else.
"""
from __future__ import annotations

from functools import lru_cache

import json

import joblib
import numpy as np
import pandas as pd

from src.config import LIMITED_HISTORY_THRESHOLD, MODELS_DIR, PROCESSED_DIR
from src.features import FEATURE_GROUPS, MODEL_FEATURES, compute_prefight_features, matchup_features
from src.build_dataset import canonical_weight_class
from src.method_model import METHODS, method_given_winner
from src.modeling import FightModel

NAMED_DIVISIONS = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
                   "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]

PLACEHOLDER_FIGHT_ID = "__prediction__"


def fighter_profiles_as_of(appearances: pd.DataFrame, fighters: pd.DataFrame,
                           fighter_ids: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
    """Pre-fight profiles for `fighter_ids` for a fight on `as_of` (shared training code)."""
    known = appearances[appearances["fighter_id"].isin(fighter_ids)]
    if len(known) and as_of <= known["event_date"].max():
        raise ValueError("as_of must be after every recorded fight of these fighters")
    placeholder = pd.DataFrame({
        "fight_id": PLACEHOLDER_FIGHT_ID, "fighter_id": fighter_ids, "event_date": as_of,
        "result": "NA", "method": "NA", "duration_s": np.nan, "has_stats": False,
    })
    combined = pd.concat([appearances, placeholder], ignore_index=True)
    prof = compute_prefight_features(combined, fighters)
    prof = prof[prof["fight_id"] == PLACEHOLDER_FIGHT_ID].set_index("fighter_id")
    return prof.loc[fighter_ids]


def build_roster(appearances: pd.DataFrame, fighters: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Fighters with at least one UFC fight, with a GUARANTEED-unique display name.

    Two different fighters can share a real name (e.g. two "Bruno Silva"s), so `name`
    alone is not a safe UI or table key. `display_name` disambiguates by appending
    division + last-fought year; if that still collides (same name, same division,
    same year -- not observed in the current data, but not impossible as it grows) the
    fighter_id is appended as a final, always-unique tie-breaker (the full ID: synthetic
    IDs such as "amb_<name>_<fight>" share prefixes, so a fragment could still collide).
    Code must key on `display_name`, never on the raw `name`, wherever two fighters'
    data could appear in the same table or dict (column headers, dict keys, ...).
    """
    a = appearances.sort_values("event_date")
    last = a.groupby("fighter_id").agg(last_fight=("event_date", "max"), last_weight_class=("weight_class", "last"))
    r = fighters.set_index("fighter_id")[["name"]].join(last, how="inner")
    r["n_fights"] = profiles.loc[r.index, "n_fights"].astype(int)
    dup = r["name"].duplicated(keep=False)
    r["display_name"] = np.where(dup, r["name"] + " (" + r["last_weight_class"].fillna("?") + ", last "
                                 + r["last_fight"].dt.year.astype(str) + ")", r["name"])
    still_dup = r["display_name"].duplicated(keep=False)
    if still_dup.any():
        ids = r.index.to_series()
        r.loc[still_dup, "display_name"] = r.loc[still_dup, "display_name"] + " [" + ids[still_dup] + "]"
    assert r["display_name"].is_unique
    return r.reset_index().sort_values(["display_name"]).reset_index(drop=True)


class Predictor:
    def __init__(self, model_path=MODELS_DIR / "model.joblib", data_dir=PROCESSED_DIR):
        self.model: FightModel = joblib.load(model_path)
        method_path = MODELS_DIR / "method_model.joblib"
        self.method_model = joblib.load(method_path) if method_path.exists() else None
        split_path = MODELS_DIR / "round_split.json"
        self.round_split = json.loads(split_path.read_text()) if split_path.exists() else None
        self.fights = pd.read_parquet(data_dir / "fights.parquet")
        self.appearances = pd.read_parquet(data_dir / "appearances.parquet")
        self.fighters = pd.read_parquet(data_dir / "fighters.parquet")
        self.as_of = self.appearances["event_date"].max() + pd.Timedelta(days=1)
        self._extra_profiles: dict[str, pd.DataFrame] = {}
        ids = self.fighters["fighter_id"].tolist()
        self.profiles = fighter_profiles_as_of(self.appearances, self.fighters, ids, self.as_of)
        self.roster = self._build_roster()

    # ------------------------------------------------------------------ lookup
    def _build_roster(self) -> pd.DataFrame:
        """Fighters with at least one UFC fight, with a unique display name."""
        return build_roster(self.appearances, self.fighters, self.profiles)

    def resolve(self, fighter: str) -> str:
        """Accept a fighter_id, display name or name (unique) and return the fighter_id."""
        r = self.roster
        for col in ("fighter_id", "display_name", "name"):
            hit = r[r[col] == fighter]
            if len(hit) == 1:
                return hit["fighter_id"].iloc[0]
            if len(hit) > 1:
                raise ValueError(f"'{fighter}' is ambiguous; use one of: {hit['display_name'].tolist()}")
        hit = r[r["name"].str.lower() == str(fighter).lower()]
        if len(hit) == 1:
            return hit["fighter_id"].iloc[0]
        raise KeyError(f"Unknown fighter: {fighter!r}")

    def profile(self, fighter: str) -> dict:
        fid = self.resolve(fighter)
        p = self.profiles.loc[fid]
        meta = self.fighters.set_index("fighter_id").loc[fid]
        return {
            "fighter_id": fid, "name": meta["name"],
            "display_name": self.roster.set_index("fighter_id").loc[fid, "display_name"],
            "dob": None if pd.isna(meta.get("dob")) else str(pd.Timestamp(meta["dob"]).date()),
            "age": float(p["age"]) if bool(p["dob_known"]) else None,
            "height_in": None if pd.isna(meta.get("height_in")) else float(meta["height_in"]),
            "reach_in": None if pd.isna(meta.get("reach_in")) else float(meta["reach_in"]),
            "stance": p["stance"],
            "record": {"wins": int(p["wins"]), "losses": int(p["losses"]), "draws_nc": int(p["draws_nc"])},
            "n_fights": int(p["n_fights"]),
            "days_since_last": None if pd.isna(p["days_since_last"]) else int(p["days_since_last"]),
            "features": {f: float(p[f]) for f in FEATURE_GROUPS_FLAT},
        }

    def recent_fights(self, fighter: str, n: int = 5) -> pd.DataFrame:
        fid = self.resolve(fighter)
        a = self.appearances[self.appearances["fighter_id"] == fid].sort_values("event_date", ascending=False).head(n)
        names = self.fighters.set_index("fighter_id")["name"]
        return pd.DataFrame({
            "date": a["event_date"].dt.date, "event": a["event"],
            "opponent": a["opponent_id"].map(names), "result": a["result"], "method": a["method"],
            "round": a["end_round"], "sig_str": a["sig_landed"].astype("Int64").astype(str) + " / " +
            a["opp_sig_landed"].astype("Int64").astype(str), "takedowns": a["td_landed"].astype("Int64"),
        }).reset_index(drop=True)

    # ------------------------------------------------------------------ predict
    def profile_row(self, fid: str) -> pd.DataFrame:
        """One-row profile. IDs unknown to the data (e.g. 'new_<name>' debutants) get a
        zero-history profile from the same feature code."""
        if fid in self.profiles.index:
            return self.profiles.loc[[fid]]
        if fid not in self._extra_profiles:
            self._extra_profiles[fid] = fighter_profiles_as_of(self.appearances, self.fighters, [fid], self.as_of)
        return self._extra_profiles[fid]

    def matchup(self, fid_a: str, fid_b: str) -> pd.DataFrame:
        return matchup_features(self.profile_row(fid_a), self.profile_row(fid_b))

    def predict_ids(self, fid_a: str, fid_b: str, weight_class: str | None = None,
                    scheduled_rounds: int = 3, is_title: bool = False) -> dict:
        """Lightweight prediction by fighter ID (used for whole fight cards)."""
        if fid_a == fid_b:
            raise ValueError("Fighter A and Fighter B must be different fighters.")
        p_a = float(self.model.predict_proba_symmetric(self.matchup(fid_a, fid_b))[0])
        return {"prob_a": p_a, "prob_b": 1 - p_a,
                "a_prior_fights": int(self.profile_row(fid_a)["n_fights"].iloc[0]),
                "b_prior_fights": int(self.profile_row(fid_b)["n_fights"].iloc[0]),
                "method": self.predict_method(fid_a, fid_b, p_a,
                                              self.bout_context(fid_a, fid_b, weight_class, scheduled_rounds, is_title))}

    # ------------------------------------------------------------------ method of victory
    def latest_division(self, fid: str) -> str | None:
        """Most recent named weight class (e.g. "Women's Flyweight"), ignoring catch/open weight."""
        a = self.appearances[self.appearances["fighter_id"] == fid].sort_values("event_date")
        named = a[a["weight_class"].str.removeprefix("Women's ").isin(NAMED_DIVISIONS)]
        return named["weight_class"].iloc[-1] if len(named) else None

    def default_weight_class(self, fid_a: str, fid_b: str) -> str:
        """Bout weight class for a hypothetical fight: shared division, else the heavier one."""
        divs = [d for d in (self.latest_division(fid_a), self.latest_division(fid_b)) if d]
        if not divs:
            return "Catch Weight"
        return max(divs, key=lambda d: canonical_weight_class(d)[1])

    def bout_context(self, fid_a: str, fid_b: str, weight_class: str | None = None,
                     scheduled_rounds: int = 3, is_title: bool = False) -> dict:
        weight_class = weight_class or self.default_weight_class(fid_a, fid_b)
        name, lbs, female, _ = canonical_weight_class(weight_class)
        return {"weight_class": weight_class, "weight_lbs": lbs, "is_female": female,
                "scheduled_rounds": int(scheduled_rounds), "is_title": bool(is_title)}

    def predict_method(self, fid_a: str, fid_b: str, p_a: float, context: dict) -> dict | None:
        """Six-way outcome: P(X wins by m) = P(X wins) * P(m | X beats the other)."""
        if self.method_model is None:
            return None
        ctx = pd.DataFrame([context])
        pa, pb = self.profile_row(fid_a), self.profile_row(fid_b)
        given_a = method_given_winner(self.method_model, pa, pb, ctx)[0]
        given_b = method_given_winner(self.method_model, pb, pa, ctx)[0]
        a = dict(zip(METHODS, (p_a * given_a).tolist()))
        b = dict(zip(METHODS, ((1 - p_a) * given_b).tolist()))
        outcomes = [("A", m, a[m]) for m in METHODS] + [("B", m, b[m]) for m in METHODS]
        winner, method, prob = max(outcomes, key=lambda t: t[2])
        return {
            "A": a, "B": b,
            "given_A_wins": dict(zip(METHODS, given_a.tolist())),
            "given_B_wins": dict(zip(METHODS, given_b.tolist())),
            "overall": {m: a[m] + b[m] for m in METHODS},
            "most_likely": {"winner": winner, "method": method, "prob": prob},
            "rounds": self.round_odds(a["DEC"] + b["DEC"], context["scheduled_rounds"]),
            "context": context,
        }

    def round_odds(self, p_decision: float, scheduled_rounds: int) -> dict | None:
        """P(goes the distance) and P(ends in round r) = (1 - P(decision)) x historical split."""
        split = (self.round_split or {}).get(str(int(scheduled_rounds)))
        if split is None:
            return None
        return {"distance": p_decision, "by_round": [(1 - p_decision) * s for s in split]}

    def predict_fight(self, fighter_a: str, fighter_b: str, explain: bool = True, weight_class: str | None = None,
                      scheduled_rounds: int = 3, is_title: bool = False) -> dict:
        fid_a, fid_b = self.resolve(fighter_a), self.resolve(fighter_b)
        if fid_a == fid_b:
            raise ValueError("Fighter A and Fighter B must be different fighters.")
        X = self.matchup(fid_a, fid_b)
        p_a = float(self.model.predict_proba_symmetric(X)[0])
        pa, pb = self.profile(fid_a), self.profile(fid_b)
        warnings = []
        for p in (pa, pb):
            if p["n_fights"] < LIMITED_HISTORY_THRESHOLD:
                # Stated as a fact, not a reliability claim: on 2017-23 out-of-sample fights,
                # predictions involving such fighters were NOT less accurate or worse calibrated.
                warnings.append(f"{p['name']} has little UFC history ({p['n_fights']} prior fight"
                                f"{'' if p['n_fights'] == 1 else 's'}), so their stats lean on league averages.")
            if p["age"] is None or p["reach_in"] is None:
                warnings.append(f"{p['name']} is missing some physical data (age/reach); neutral values were used.")
        return {
            "fighter_a": pa, "fighter_b": pb,
            "prob_a": p_a, "prob_b": 1 - p_a,
            "oriented_prob_a": float(self.model.oriented_proba(X)[0]),
            "features": X.iloc[0].to_dict(),
            "factors": self.explain(X) if explain else [],
            "warnings": warnings,
            "method": self.predict_method(fid_a, fid_b, p_a,
                                          self.bout_context(fid_a, fid_b, weight_class, scheduled_rounds, is_title)),
            "model_type": self.model.kind,
            "as_of": str(self.as_of.date()),
        }

    def explain(self, X: pd.DataFrame) -> list[dict]:
        """Group occlusion: change in P(A) when a feature group's differences are set to 0.

        Positive impact favours Fighter A, negative favours Fighter B. These are model
        factors (what moved this model's estimate), not causal effects.
        """
        base = float(self.model.predict_proba_symmetric(X)[0])
        out = []
        for group, feats in FEATURE_GROUPS.items():
            Xo = X.copy()
            Xo[[f"{f}_diff" for f in feats]] = 0.0
            impact = base - float(self.model.predict_proba_symmetric(Xo)[0])
            out.append({"group": group, "impact": impact,
                        "favors": "A" if impact > 0 else "B",
                        "features": {f"{f}_diff": float(X[f"{f}_diff"].iloc[0]) for f in feats}})
        return sorted(out, key=lambda d: -abs(d["impact"]))


FEATURE_GROUPS_FLAT = [f for fs in FEATURE_GROUPS.values() for f in fs]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


def predict_fight(fighter_a: str, fighter_b: str) -> dict:
    return get_predictor().predict_fight(fighter_a, fighter_b)


if __name__ == "__main__":
    import json
    import sys
    a, b = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("Islam Makhachev", "Charles Oliveira")
    r = predict_fight(a, b)
    print(f"{r['fighter_a']['name']}: {r['prob_a']:.1%}   {r['fighter_b']['name']}: {r['prob_b']:.1%}")
    for f in r["factors"][:6]:
        print(f"  {f['group']:38s} favours {'A' if f['impact'] > 0 else 'B'}  {f['impact']:+.3f}")
    print(json.dumps(r["warnings"]))
