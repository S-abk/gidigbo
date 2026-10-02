# UFC Fight Predictor (local MVP)

[![Build](https://github.com/S-abk/gidigbo/actions/workflows/build.yml/badge.svg)](https://github.com/S-abk/gidigbo/actions/workflows/build.yml)
[![Tests](https://github.com/S-abk/gidigbo/actions/workflows/tests.yml/badge.svg)](https://github.com/S-abk/gidigbo/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Estimates **the probability that Fighter A beats Fighter B** for any two UFC fighters. It uses
historical UFCStats data, leakage-safe pre-fight features, chronologically validated models and a
local Streamlit UI.

> These probabilities are statistical model estimates, not guarantees. This project gives no
> betting advice.

---

## Quick start

Requires Python 3.10+ (developed on 3.12). Run all commands from the project root (`ufc_predictor/`).

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
```

| OS | Activate |
|----|----------|
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |

(On macOS/Linux you may need `python3` instead of `python`.)

**2. Install dependencies and get the data**

```bash
pip install -r requirements.txt
git clone https://github.com/Greco1899/scrape_ufc_stats data/raw/scrape_ufc_stats
```

**3. Train, launch and test**

```bash
python -m src.train        # builds the dataset, trains, evaluates, saves models/ (~30 s)
streamlit run app.py       # opens the UI in your browser
pytest -q                  # leakage, reversal, schema and prediction tests
```

To predict from the command line: `python -m src.predict "Islam Makhachev" "Charles Oliveira"`.

---

## Project layout

```text
ufc_predictor/
├── app.py                    Streamlit UI (Predict / Upcoming Card / Fighter Comparison / Model Performance / Model Insights)
├── requirements.txt
├── data/
│   ├── raw/scrape_ufc_stats/ cloned source repo (CSV files, not modified)
│   └── processed/            fights / appearances / fighters parquet (generated)
├── models/
│   ├── model.joblib          production model (base estimator + calibrator)
│   ├── method_model.joblib   method-of-victory model
│   ├── model_metadata.json   periods, features, hyperparameters, metrics, data commit
│   ├── evaluation.json       all models' test results, diagnostics, importance
│   └── candidate_*.joblib    the other evaluated models (generated)
├── src/
│   ├── config.py             paths, seed, split dates
│   ├── data_loader.py        raw CSV loading and parsing (documents field provenance)
│   ├── build_dataset.py      canonical fights + per-fighter appearances, fighter ID resolution
│   ├── features.py           pre-fight profiles, matchup differences, A/B orientation, mirroring
│   ├── modeling.py           model definitions, calibration, symmetric prediction rule
│   ├── method_model.py       method of victory (KO/TKO, SUB, DEC) given the winner
│   ├── train.py              split, walk-forward CV, calibration, selection, test evaluation
│   ├── evaluate.py           metrics and diagnostics
│   ├── predict.py            Predictor / predict_fight(), explanations
│   └── upcoming.py           scheduled events + fight cards from Wikipedia (on demand)
└── tests/
    ├── test_features.py
    ├── test_leakage.py
    ├── test_prediction.py
    └── test_upcoming.py
```

---

## Data

Source: [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats), a daily-refreshed
scrape of [ufcstats.com](http://ufcstats.com). The source commit used for training is stored in
`models/model_metadata.json`. The current build covers fights from **1994-03-11 to 2026-09-26**.

| File | Used for |
|------|----------|
| `ufc_event_details.csv` | event name → **event date** (the only date source) |
| `ufc_fight_results.csv` | one row per fight: `BOUT` ("A vs. B"), `OUTCOME` (W/L, L/W, D/D, NC/NC aligned with BOUT order), weight class, method, finishing round/time, `TIME FORMAT`, fight URL |
| `ufc_fight_stats.csv` | per fighter, per round: strikes, takedowns, submissions, control, knockdowns |
| `ufc_fighter_tott.csv` | fighter URL (→ fighter ID), height, reach, stance, DOB |
| `ufc_fight_details.csv`, `ufc_fighter_details.csv` | redundant with the files above; not needed |

### Join keys and data issues found

- **Fights** are keyed by the fight URL ID.
- **Stats have no fight URL and no fighter URL.** They join to fights on `(EVENT, BOUT)` and to a
  side by fighter name. Every stats row's fighter matches one side of its bout (checked by an
  assertion).
- **Fighters appear by name only** in fight files, so names are resolved to tott fighter IDs:
  - Exact or normalised name match covers the large majority.
  - 16 hand-checked aliases cover renames such as `Zach Reese → Zachary Reese` and
    `Waldo Cortes Acosta → Waldo Cortes-Acosta`. An alias is accepted only when the weight class
    agrees and the two spellings' careers do not overlap.
  - 9 names are shared by two fighters (52 fights). Each fight is resolved separately by a
    plausible age on the fight date and the closest tott weight to the bout's weight class.
  - Names with no tott match get a synthetic ID (`unk_<name>`) and no physical data. This affects 5
    fighters and 8 fighter slots.
- **25 fights appear twice** under renamed events ("UFC Fight Night: X" vs "Noche UFC: X"). They are
  de-duplicated by fight URL, keeping the copy whose event has a date.
- **2 fights (UFC – Road to UFC 4.6)** have no event in the event file and are dropped.
- **Sakuraba vs. Silveira (UFC – Ultimate Japan)** occurs twice with the same `(EVENT, BOUT)`, so its
  stats can't be attributed to either fight. Those stats are nulled; the results are kept.

Row counts: 8,936 result rows → 8,911 after URL de-duplication → **8,909 dated fights**. Of these,
8,754 have a binary winner, 90 are no contests and 65 are draws. That gives **17,818 fighter
appearances**, 99.7% of them with round-level stats.

Unknown values (`--`) become NaN, never 0, except where 0 genuinely means zero (for example
"0 of 0" strikes).

---

## Prediction target

`label = 1` if Fighter A won. **Draws and no contests are excluded as training targets.** UFCStats
records overturned results as NC, so those are excluded too. All of these still count as *prior
fights* in each fighter's history: they happened, and their stats are real. Method of victory is
predicted by a second model (below); round and duration are not.

---

## Features

All features live in `src/features.py`. The same function, `compute_prefight_features`, is used for
training and for the app.

- **History is strictly earlier event dates only.** Bouts on the same date are excluded as well.
  Early tournaments had several bouts per fighter in one night, and the bout order within a night is
  unknown.
- **Windows** are career-to-date plus the last 3 and last 5 fight-nights.
- **Rates are ratios of sums**, for example career significant strikes landed ÷ career minutes, not
  averages of per-fight rates. Minutes come from the finishing round and time plus the round lengths
  parsed from `TIME FORMAT`, which handles 5-round, overtime and "No Time Limit" formats. Only fights
  with round stats feed the stat numerators and denominators.
- **Shrinkage.** Each rate is pulled toward a league average in proportion to how little data the
  fighter has: 15 pseudo-minutes, 40 pseudo-strike attempts, 6 pseudo-takedown attempts, or 2
  pseudo-fights. Debutants therefore get sensible values, not NaN or extremes. The priors were
  computed once from **2001–2021 appearances only** (the training period).

| Group | Features |
|-------|----------|
| Experience | prior UFC fights, shrunk win % |
| Recent form | wins in last 3 and last 5, last-5 win %, current streak (draw/NC resets it) |
| Physical | age on fight date, height, reach, southpaw/switch flags |
| Activity | days since previous fight |
| Striking | sig. strikes landed and absorbed per minute, differential (career, last 3, last 5), accuracy, defence, knockdowns per 15 min |
| Grappling | takedowns per 15 min (career, last 5), takedown accuracy, takedown defence, control-time share, submission attempts per 15 min |
| Fight history | finish rate, KO/sub/decision share of wins, rate of being finished, average fight length |

**Matchup representation.** For each profile feature *f*, the model sees `f_diff = A − B`, so it
learns *differences*. The one extra symmetric context feature is `min_prior_fights`, the smaller of
the two fighters' experience. It tells the model how much history the matchup has. Remaining NaN
differences become 0 (no advantage).

---

## How leakage is prevented

For every historical fight the question is: *what did we know about these two fighters immediately
before this fight?*

1. **Chronological history.** Features use exclusive cumulative sums over (fighter, event date)
   groups, so the current fight and anything later are never included.
2. **Survivorship leak through missing data (found and fixed).** The tott file is a *current
   snapshot*, and fighters who were cut early are far more likely to have missing data. In training
   fights where exactly one side was missing:

   | Missing field | Missing side's win rate |
   |---------------|-------------------------|
   | reach | 11.6% |
   | DOB | 21% |
   | stance | 36% |

   A missingness flag would therefore leak future career length. Missing values are neutrally
   imputed per fighter before differencing, and no missingness indicator is created:
   - missing reach → height + 2 in (the median offset; an integer, so imputed reach is
     indistinguishable in form from real reach)
   - missing DOB → age 30
   - unknown stance → treated like orthodox
3. **Static attributes only.** Height, reach, stance and DOB are treated as static. tott `WEIGHT`
   (current division) is used only to tell same-named fighters apart, never as a feature.
4. **No post-fight information.** There are no rankings, final records or career averages: every
   number is "as of the day before".
5. **Tests** (`tests/test_leakage.py`):
   - *Perturbation test:* randomise every outcome and stat on or after a date *d*, rebuild the
     features, and assert that features for fights on *d* are unchanged.
   - *Truncation test:* features built from data before *d* must equal those built from the full
     dataset.
   - Same-night tournament bouts must not see each other.
   - Prior-fight counts and wins equal an independent count of strictly earlier fights, for every
     appearance and every training row.
   - Changing a fight's own result or stats does not change that fight's features.
   - *Inference = training:* a fighter's app-style profile built "as of" a real fight date equals the
     training features for that fight, which proves both paths share the same code.

---

## Fighter A / Fighter B bias

In the raw data the winner is listed first **64% of the time** (5,617 W/L vs 3,161 L/W).

- **Orientation.** A/B is reassigned per fight by a deterministic coin: an md5 hash of the fight ID,
  so it is stable across rebuilds. Label mean ≈ 0.49.
- **Mirrored training.** Training rows are mirrored (A↔B, differences negated, label flipped) *within
  the training split only*, so no fight crosses a split boundary.
- **Symmetric prediction rule.** Every prediction, in evaluation and in the app, is
  `P(A) = ½·[p(A,B) + 1 − p(B,A)]` after calibration. That makes `P(A beats B) = 1 − P(B beats A)`
  exactly.
- **Measured gap.** Before symmetrisation, the mean |p(A,B) − (1 − p(B,A))| on the test set is ~0
  for logistic regression (it is exactly antisymmetric), 0.012 for random forest and 0.012 for
  XGBoost.

---

## Chronological validation

A random split would let the model learn from fights *after* the ones it is tested on, and would
mix the same fighters' futures into training. Splits are therefore by date:

| Period | Dates | Fights |
|--------|-------|--------|
| Train | 2001-02-23 → 2021-12-18 | 6,046 |
| Validation | 2022-01-15 → 2023-12-16 | 1,010 |
| Test | 2024-01-13 → 2026-09-26 | 1,437 |

The 261 fights before 2001 are used only as fighter history.

1. **Walk-forward CV** inside the training period: test on each of 2017–2021, training on all prior
   years. This picks hyperparameters from small grids.
2. **Refit.** Each model family is refit on the full training period.
3. **Calibration** (none / Platt-sigmoid / isotonic) is chosen per model on the validation period by
   2-fold chronological cross-fitting, then fit on all of validation. The test set is never used for
   calibration.
4. **Production model** = lowest cross-fitted validation log loss. If another model beats logistic
   regression by less than 0.002, the interpretable logistic regression wins.
5. **The test set is scored once,** after selection.

Selected: **logistic regression** (C = 0.01, no extra calibration: cross-fitting found the raw
logistic regression already as well calibrated as Platt scaling). Its cross-fitted validation log loss
was 0.6560, versus 0.6565 for XGBoost and 0.6626 for random forest. The shipped `model.joblib` is
exactly the evaluated artifact: trained on the training period and calibrated on validation. At
prediction time it uses fighter histories through the latest fight in the data.

## Results (test set, 1,437 fights, 2024-01 → 2026-09)

| Model | Accuracy | Log loss | Brier | ROC-AUC |
|-------|---------:|---------:|------:|--------:|
| Always 50% | 0.475* | 0.6931 | 0.2500 | 0.500 |
| Win-% difference (logistic) | 0.6075 | 0.6686 | 0.2379 | 0.640 |
| Experience + record (logistic) | 0.6089 | 0.6702 | 0.2387 | 0.636 |
| **Logistic regression (selected)** | **0.6312** | **0.6365** | **0.2228** | **0.692** |
| Random forest | 0.6221 | 0.6488 | 0.2286 | 0.670 |
| XGBoost | 0.6186 | 0.6440 | 0.2265 | 0.675 |

\* A constant 0.5 counts as "A wins", so its accuracy is just the test label rate.

**Interpretation.**
- The full model improves log loss by about 0.032 over the best naive baseline and gains about 2.2
  points of accuracy.
- That is modest but real: MMA is noisy, and ~63% accuracy with ~0.64 log loss is plausible for
  public-stats models.
- Tree models did not beat the linear model out of sample.
- In validation permutation importance, the most influential feature group was **age**, followed by
  striking defence / damage absorbed, wrestling offence, finishing / durability and striking
  differential.

Calibration curves, confusion matrices and probability histograms are in `models/evaluation.json`
and on the **Model Performance** tab.

---

## Method of victory

A second model, `src/method_model.py`, estimates **how a fight ends given who wins**:
P(KO/TKO, submission, decision | winner beat loser). It is a multinomial logistic regression on the
same leakage-safe pre-fight profiles:
- the winner's finishing profile (KO/sub/decision win shares, knockdowns, submission attempts,
  output)
- the loser's durability (rate of being finished, strikes absorbed, defence)
- a few winner-minus-loser differences
- bout context: weight class, women's bout, 5 rounds, title fight

Combining it with the win model gives a six-way outcome that sums to 100%:
`P(A by KO) = P(A wins) × P(KO | A beats B)`. Swapping the fighters mirrors it exactly.

- **Training window.** It trains on 2010–2021 fights (5,043), because the decision share
  rose from ~37% in 2001–10 to ~49% after 2010. C is chosen on 2022–23, and it is tested once on
  2024+ (1,434 fights).
- **Excluded outcomes.** DQs and other rare outcomes (<1%) are not modelled.
- **Bout context.** In the app you can set weight class, 3 or 5 rounds and title fight. These change
  only the method split, never the win probability. The defaults are the fighters' shared division
  (or the heavier one), 3 rounds and non-title. On the Upcoming Card, the main event and title fights
  are scored as 5 rounds.

| Test (2024+) | Log loss | Top-method accuracy |
|---|---:|---:|
| **Method model** | **0.926** | **55.5%** |
| Weight-class method rates | 1.005 | 51.4% |
| Overall method rates | 1.017 | 49.7% |

On the test set, average predicted shares were close to the actual ones:

| | KO/TKO | Submission | Decision |
|---|---:|---:|---:|
| Predicted | 34.3% | 15.8% | 50.0% |
| Actual | 33.0% | 17.3% | 49.7% |

Full six-way outcome test log loss:

| Approach | Log loss |
|---|---:|
| **Win model × method model** | **1.562** |
| Win model × average method rates | 1.653 |
| Uniform guess | 1.792 |

## Upcoming cards

The **Upcoming Card** tab scores every bout on a scheduled event. UFCStats has no future fights
in its data, and its website now blocks scripted access with a JavaScript browser check, so
`src/upcoming.py` reads scheduled events and fight cards from **Wikipedia's public API**:
"List of UFC events", then each event page's `{{MMAevent bout}}` templates.

- **On demand only.** Nothing is fetched until you click *Load upcoming events*. Results are cached
  for an hour, and *Refresh* refetches them.
- **Name matching.** Names are matched to UFCStats fighters after stripping accents, applying the
  alias table and removing Jr./III suffixes. Same-name fighters are resolved by division or weight.
- **Debutants.** Fighters not in the data are scored as debutants with neutral, league-average
  profiles and are flagged in the table.
- **Full breakdown.** *Open in Predict tab* sends a bout to the Predict tab, with factors and comparison.
- **Accuracy.** Cards change late, and Wikipedia can lag or be edited, so check official sources.

## Explanations

- **Model factors.** The app shows *model factors*, not causes. For each feature group it measures how
  much P(A) changes when that group's differences are set to 0 (group occlusion through the same
  symmetric prediction rule).
- **Global importance.**
  - Standardised logistic-regression coefficients
  - Tree importances for candidate models
  - Group permutation importance on validation

---

## Limitations

- **UFC-only history.** Pre-UFC and regional records are not in UFCStats, so UFC debutants look like
  blank slates (shrunk to league averages). The app warns when a fighter has fewer than 3 prior UFC
  fights.
- **Snapshot attributes.** Height, reach and DOB come from a current snapshot. DOB is missing for some
  fighters (a neutral age of 30 is used), and reach is missing for many older fighters.
- **Name-based identity.** The fight files have no fighter IDs. Aliases and duplicate-name
  resolution are heuristic, and a few fighters only have synthetic IDs without physical data.
- **Sparse early stats.** Early events (1990s) lack control time and some stats. Those fights are
  history only.
- **Missing context.** There are no opponent-strength adjustments, so beating weak opposition counts
  like beating strong opposition. There is no injury, camp, weight-cut or short-notice information.
- **Hypothetical matchups ignore weight class, title status and 5-round scheduling.** The model does
  not use these.
- **Model age.** The shipped model's weights end in 2021 (train) and 2023 (calibration). Fighter
  profiles are current, but model weights do not include 2024+ fights.

## Retraining / refreshing data

```bash
git -C data/raw/scrape_ufc_stats pull     # get the latest scraped fights
python -m src.train                       # rebuild dataset, retrain, re-evaluate, overwrite models/
```

Split dates live in `src/config.py` (`VALIDATION_START`, `TEST_START`, `WALK_FORWARD_YEARS`). Move
them forward as data accumulates.

## Recommendations for the next version

- **Opponent strength:** Elo/Glicko ratings computed sequentially, and opponent-adjusted stats.
- **Production refit:** after evaluation, refit the chosen configuration on all data (with
  time-respecting calibration) for the production model.
- **Bout context in the UI:** weight class, 5-round and title flags as inputs, plus interactions
  learned from training.
- **Pre-UFC records** from another source, to reduce debutant uncertainty.
- **Betting odds** as an external *benchmark* for calibration only, not as a feature or product.
- **SHAP** local explanations for tree models, and per-feature uncertainty or prediction intervals.
- **Round prediction**, and a round-aware method model.

## License

This project's code is MIT licensed (see `LICENSE`). The data source,
[Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats), is separately
licensed under GPL-3.0; it is cloned by the user into `data/raw/` (see Quick start) and is not
redistributed as part of this repository.
