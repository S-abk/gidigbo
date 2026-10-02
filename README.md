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

Requires Python 3.12, the version CI tests and the pinned dependencies were resolved for. Run all
commands from the repository root.

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

**2. Install dependencies and launch**

```bash
pip install -r requirements-dev.txt   # runtime deps (requirements.txt) + pytest
streamlit run app.py                  # opens the UI in your browser
```

The trained models are in the repository, so the app runs without training. On its first start it
downloads the four source CSVs it needs (~10 MB) and builds the dataset, which takes about 30 s. The
download is pinned to the exact upstream commit the shipped model was trained on.

**3. Retrain and test (optional)**

```bash
python -m src.fetch_data   # the source CSVs, pinned (add --latest for the newest data)
python -m src.train        # builds the dataset, evaluates, refits for production, saves models/ (~40 s)
pytest -q                  # leakage, reversal, schema, prediction and model-quality tests
```

You can also `git clone https://github.com/Greco1899/scrape_ufc_stats data/raw/scrape_ufc_stats`
yourself. A git checkout there is used as-is and never overwritten.

To predict from the command line: `python -m src.predict "Islam Makhachev" "Charles Oliveira"`.

---

## Project layout

```text
ufc_predictor/
├── app.py                    Streamlit UI (Predict / Upcoming Card / Fighter Comparison / Model Performance / Model Insights)
├── requirements.txt          runtime dependencies, pinned
├── requirements-dev.txt      + pytest
├── .github/workflows/        build, tests, weekly data/model refresh
├── data/
│   ├── raw/scrape_ufc_stats/ cloned source repo (CSV files, not modified)
│   └── processed/            fights / appearances / fighters parquet (generated)
├── models/
│   ├── model.joblib          deployed model: the chosen configuration refit on all fights
│   ├── method_model.joblib   method-of-victory model (deployed, refit on all data)
│   ├── evaluated_*.joblib    the exact artifacts the held-out metrics describe
│   ├── model_metadata.json   periods, features, hyperparameters, metrics, data commit
│   ├── evaluation.json       all models' test results, diagnostics, importance
│   └── candidate_*.joblib    the other evaluated models (generated)
├── src/
│   ├── config.py             paths, seed, split dates
│   ├── data_loader.py        raw CSV loading and parsing (documents field provenance)
│   ├── fetch_data.py         download the source CSVs (pinned commit) and build the dataset if missing
│   ├── build_dataset.py      canonical fights + per-fighter appearances, fighter ID resolution
│   ├── features.py           pre-fight profiles, matchup differences, A/B orientation, mirroring
│   ├── modeling.py           model definitions, calibration, symmetric prediction rule
│   ├── method_model.py       method of victory (KO/TKO, SUB, DEC) given the winner
│   ├── train.py              split, walk-forward CV, calibration, selection, test evaluation
│   ├── evaluate.py           metrics and diagnostics
│   ├── predict.py            Predictor / predict_fight(), explanations
│   ├── quality_gate.py       model-quality gate used by CI and the refresh workflow
│   ├── ratings.py            leakage-safe Glicko-1 opponent-strength ratings
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
| Opponent-strength rating | Glicko-1 rating and rating deviation (RD), see below |
| Experience | prior UFC fights, shrunk win % |
| Recent form | wins in last 3 and last 5, last-5 win %, current streak (draw/NC resets it) |
| Physical | age on fight date, height, reach, southpaw/switch flags |
| Activity | days since previous fight |
| Striking | sig. strikes landed and absorbed per minute, differential (career, last 3, last 5), accuracy, defence, knockdowns per 15 min |
| Grappling | takedowns per 15 min (career, last 5), takedown accuracy, takedown defence, control-time share, submission attempts per 15 min |
| Fight history | finish rate, KO/sub/decision share of wins, rate of being finished, average fight length |

**Opponent-strength rating (`src/ratings.py`).** The other features describe what a fighter did,
never *who they did it against*. A Glicko-1 rating pools results across the whole UFC network, so
beating strong opposition counts for more. Its rating deviation (RD) is an explicit uncertainty: it
starts at 350 for debutants and grows during layoffs. Each event date is one rating period:
everyone on the card is snapshotted *before* that card's results are applied, so the same leakage
rules hold (strictly earlier dates only; same-night bouts can't see each other). Wins score 1,
losses 0, draws 0.5; no contests are not applied. The app shows the rating as *rating ± 95% range*
(± 1.96·RD). See [Glicko rating: backtest](#glicko-rating-backtest) for how it was tuned and
whether it earned its place.

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
5. **Tests** (`tests/test_leakage.py`). They check every profile feature, including the Glicko
   rating, and [CI](.github/workflows/tests.yml) runs them on every push and pull request:
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
  for logistic regression (it is exactly antisymmetric), 0.014 for random forest and 0.013 for
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
3. **Calibration** (none or Platt-sigmoid) is chosen per model on the validation period by 2-fold
   chronological cross-fitting, then fit on all of validation. The test set is never used for
   calibration. Isotonic calibration is deliberately not a candidate; see
   [Glicko rating: backtest](#glicko-rating-backtest).
4. **Production model** = lowest cross-fitted validation log loss. If another model beats logistic
   regression by less than 0.002, the interpretable logistic regression wins.
5. **The test set is scored once,** after selection.
6. **Production refit.** The deployed model is the chosen configuration (same model family,
   hyperparameters and calibration method) refit on *every* fight from 2001 through the latest
   event. Nothing is re-chosen at this step. If a calibrator is selected, it is fit on out-of-time
   predictions: each year from 2022 on is predicted by a model trained only on earlier fights.

Selected: **logistic regression** (C = 0.01, no extra calibration: cross-fitting found the raw
logistic regression already as well calibrated as Platt scaling). Its cross-fitted validation log loss
was 0.6473, versus 0.6495 for XGBoost and 0.6569 for random forest.

- **The metrics below come from the held-out run.** That run trained through 2021, chose on 2022–23
  and scored 2024+; it is saved as `models/evaluated_model.joblib`.
- **The deployed `model.joblib` is the refit.** It learned from 8,493 fights through the latest event
  and changes test-period predictions by 1.9 points on average.
- **No held-out score exists for the refit itself,** since it has seen every fight. This is the
  standard trade: estimate performance on held-out data, then deploy the same recipe trained on
  everything.

## Results (test set, 1,437 fights, 2024-01 → 2026-09)

These figures are a snapshot from October 2026. Each [data refresh](#automatic-data-refresh) adds
new fights to the test period and recomputes them; the app's **Model Performance** tab always shows
the current numbers.

| Model | Accuracy | Log loss | Brier | ROC-AUC |
|-------|---------:|---------:|------:|--------:|
| Always 50% | 0.475* | 0.6931 | 0.2500 | 0.500 |
| Win-% difference (logistic) | 0.6075 | 0.6686 | 0.2379 | 0.640 |
| Experience + record (logistic) | 0.6089 | 0.6702 | 0.2387 | 0.636 |
| **Logistic regression (selected)** | **0.6541** | **0.6282** | **0.2188** | **0.707** |
| Random forest | 0.6305 | 0.6390 | 0.2239 | 0.688 |
| XGBoost | 0.6333 | 0.6322 | 0.2209 | 0.697 |

\* A constant 0.5 counts as "A wins", so its accuracy is just the test label rate.

**Interpretation.**
- The full model improves log loss by about 0.040 over the best naive baseline and gains about 4.5
  points of accuracy.
- That is modest but real: MMA is noisy, and ~65% accuracy with ~0.63 log loss is plausible for
  public-stats models.
- Tree models did not beat the linear model out of sample.
- In validation permutation importance, the most influential feature group is the **opponent-strength
  rating**, then **age**, then striking defence / damage absorbed, wrestling offence and striking
  differential.

Calibration curves, confusion matrices and probability histograms are in `models/evaluation.json`
and on the **Model Performance** tab.

---

## Glicko rating: backtest

The opponent-strength rating came out of a robustness review. It was added only after it beat the
existing model on held-out data. **Decision rule, fixed before any run:** keep it only if the
production model's cross-fitted *validation* log loss improves on the pre-Glicko 0.6560 and
walk-forward CV doesn't get worse. Test results play no part.

**Tuning** used logistic-regression walk-forward CV on 2017–2021, the training period only. The
tuned constant is how long a layoff takes to reset a fighter's rating uncertainty to a debutant's:

| Setting | Walk-forward CV log loss |
|---|---:|
| No Glicko (baseline) | 0.6612 |
| 2-year reset | 0.6570 |
| **5-year reset (chosen)** | **0.6545** |
| 10-year reset | 0.6545 |
| No layoff growth | 0.6563 |

A 2-year reset kept almost every pre-fight RD near the 350 maximum, because one fight is little
information against a long layoff.

**Validation** (the decision): the production model went from 0.6560 to **0.6473**. Every family
improved: XGBoost 0.6565 → 0.6495, random forest 0.6626 → 0.6569. So it was kept.

**Test**, 1,437 fights, logistic regression:

| | Accuracy | Log loss | Brier | ROC-AUC |
|---|---:|---:|---:|---:|
| Before Glicko | 63.1% | 0.6365 | 0.2228 | 0.692 |
| **With Glicko** | **65.4%** | **0.6282** | **0.2188** | **0.707** |

**Disclosure: isotonic calibration removed after the test set was viewed.** In the first run with
Glicko, cross-fitting picked isotonic calibration over "none" by 0.0011, which is within noise. That
run scored a *worse* test log loss (0.6375). On investigation, isotonic regression is a step
function:
- It gave only 38 distinct probabilities across 1,437 test fights.
- It claimed over 95% confidence on fights it got right only ~82% of the time.
- Independent of any labels, it left ~54% of the app's occlusion-based model factors at exactly
  zero impact, because small changes don't move a plateau.

Isotonic is now excluded from the calibration candidates. The app's explanations need a smooth,
strictly increasing mapping, and isotonic is known to overfit small calibration sets (~500 fights
per cross-fit half here). `tests/test_prediction.py::test_explanations_are_not_degenerate` guards
against a step-function calibrator returning. Because this was noticed *after* viewing the test
set, treat the final test log loss as slightly optimistic. The validation figures above are the
clean estimate.

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
| **Win model × method model** | **1.553** |
| Win model × average method rates | 1.645 |
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
- **Fails loudly on format changes.** The parsers are regex-based, so a Wikipedia template change
  could otherwise look exactly like "no events" or "no bouts announced yet". If a page contains
  `{{dts|` or `{{MMAevent bout` templates but nothing parses, `src/upcoming.py` raises
  `ParseFailure` and the app shows an error saying the format likely changed. Genuinely empty
  results, such as every event being in the past, still return empty.
- **Accuracy.** Cards change late, and Wikipedia can lag or be edited, so check official sources.

## Explanations

- **Model factors.** The app shows *model factors*, not causes. For each feature group it measures how
  much P(A) changes when that group's differences are set to 0 (group occlusion through the same
  symmetric prediction rule). This only works with a smooth calibrator, which is why isotonic
  calibration is excluded.
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
  resolution are heuristic, and a few fighters only have synthetic IDs without physical data. Two
  different fighters can share a name (there are two "Bruno Silva"s), so the app keys everything on
  a `display_name` that is guaranteed unique, never on the raw name.
- **Sparse early stats.** Early events (1990s) lack control time and some stats. Those fights are
  history only.
- **Missing context.** Opponent strength enters only through the Glicko rating. The per-minute
  stats are not opponent-adjusted, so landing 5 strikes a minute on a weak opponent counts the same
  as on a strong one. There is no injury, camp, weight-cut or short-notice information.
- **The win model ignores weight class, title status and 5-round scheduling.** Only the
  method-of-victory model uses them.
- **Model age.** The deployed model learns from fights up to the last data refresh. A deployment
  that doesn't run the refresh workflow slowly goes stale. See [Model staleness](#model-staleness).

## Retraining / refreshing data

```bash
python -m src.fetch_data --latest   # newest scraped fights (or: git -C data/raw/scrape_ufc_stats pull)
python -m src.train                 # rebuild dataset, re-evaluate, refit for production, overwrite models/
pytest -q
```

Split dates live in `src/config.py` (`VALIDATION_START`, `TEST_START`, `WALK_FORWARD_YEARS`). Keeping
them fixed is deliberate: every new fight lands in the test period. Over time the test metrics
increasingly reflect fights that no design decision has seen.

### Model staleness

Model weights age even though fighter profiles advance with every rebuild. Concept drift is treated
as an ongoing decision, not something settled at training time. `src/config.py` sets two simple
guidelines, which are recorded in `models/model_metadata.json`:

- `RECALIBRATE_AFTER_DAYS = 180`
- `RETRAIN_AFTER_DAYS = 365`

The **Model Performance** tab shows the model's age. Age is measured from the last fight the
deployed weights were fit on (`production_fit.end`), *not* from when `train.py` last ran: retraining
on unchanged data makes nothing fresher. The age shows as a warning past the recalibration guideline
and as an error past the retrain guideline. With the production refit and the weekly
[data refresh](#automatic-data-refresh), the model stays within days of the latest event.

These are cheap stand-ins for a formal retain / recalibrate / refit rule, not an optimal
schedule.

## Deployment

The app is a single Streamlit script with its trained models committed, so any host that can run
`streamlit run app.py` works.

**Streamlit Community Cloud** (free, deploys from this GitHub repo):
1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Create an app from `S-abk/gidigbo`, branch `main`, main file `app.py`.
3. Under *Advanced settings*, choose **Python 3.12**. No secrets are needed.

On its first start the app downloads the source data and builds the dataset (~30 s). After that it
starts in a few seconds. Every push to `main` redeploys the app, including the weekly refresh. If the
new model was trained on a different upstream commit, the app re-downloads the matching data by
itself.

**Requirements for a host:**
- **Memory:** about 550 MB peak on a first start (download + build + model), about 450 MB after that,
  plus Streamlit's own overhead. A 512 MB tier is too small.
- **Network:** outbound HTTPS to `raw.githubusercontent.com` (first start) and `en.wikipedia.org`
  (Upcoming Card tab).
- **Python 3.12** with `requirements.txt`. The versions are pinned because the models are pickled
  scikit-learn objects; bump them only together with a retrain.

## Automatic data refresh

`.github/workflows/refresh.yml` runs every Monday at 09:00 UTC and on demand: *Actions → Refresh data
and model → Run workflow*, with an optional *force* flag.

1. **Skip if nothing changed.** If the upstream data repo has no new commit since the committed
   model, it stops.
2. **Retrain.** It downloads the latest CSVs and runs `python -m src.train`: held-out evaluation,
   then the production refit.
3. **Gate.** It runs the whole test suite, then `src/quality_gate.py` against the committed model.
   The gate fails if any of these is true:
   - the model no longer beats the naive baselines by 0.01 log loss
   - validation log loss exceeds 0.68
   - the dataset shrank or has under 8,000 fights, a sign of truncated upstream data
   - test log loss rose by more than 0.02
4. **Commit.** Only if everything passed does it commit `models/` as `github-actions[bot]`, which
   redeploys the app.

Commits made with the workflow's built-in token don't trigger the other CI workflows; the refresh
runs the same tests itself. In public repositories, GitHub can automatically disable scheduled
workflows after 60 days without repository activity. If that happens, re-enable it in the Actions
tab. The app's staleness warning is the backstop: it turns amber, then red, if refreshes stop. A failed run leaves the committed model untouched and shows red in the Actions tab. By default,
GitHub emails failures of scheduled workflows to whoever last edited the workflow's schedule.

## Recommendations for the next version

- **Clean re-evaluation:** once enough post-2026 fights exist, score the current pipeline on a new
  test period that no design decision has touched. The current test set has been viewed (see the
  Glicko backtest disclosure).
- **Opponent-adjusted stats**, such as strikes landed relative to what each opponent usually absorbs,
  and the Glicko rating as an input to the method-of-victory model.
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
