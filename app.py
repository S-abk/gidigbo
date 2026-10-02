"""Streamlit UI for the fight win-probability model.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import MODELS_DIR

RED, GOLD, STEEL, GRID = "#d7263d", "#f2b134", "#8a8f98", "#2a2d33"
DISCLAIMER = "These probabilities are statistical model estimates, not guarantees."
DEFAULT_A, DEFAULT_B = "Islam Makhachev", "Ilia Topuria"

st.set_page_config(page_title="Fight Predictor", page_icon="🥊", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&display=swap');
.stApp {{ background: radial-gradient(circle at 50% -20%, #2a1216 0%, #121316 55%); }}
h1, h2, h3, h4, .fp-name {{ font-family: 'Oswald', 'Arial Narrow', sans-serif !important;
    text-transform: uppercase; letter-spacing: .04em; }}
.fp-title {{ font-family: 'Oswald', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1;
    text-transform: uppercase; letter-spacing: .05em; margin: 0; }}
.fp-title span {{ color: {RED}; }}
.fp-sub {{ color: {STEEL}; margin: .3rem 0 1rem 0; }}
.fp-card {{ background: #1d1f24; border: 1px solid {GRID}; border-radius: 10px; padding: 1.1rem 1.3rem; }}
.fp-card.a {{ border-top: 4px solid {RED}; }}
.fp-card.b {{ border-top: 4px solid {GOLD}; }}
.fp-name {{ font-size: 1.6rem; font-weight: 700; margin: 0; }}
.fp-prob {{ font-family: 'Oswald', sans-serif; font-size: 3.4rem; font-weight: 700; line-height: 1.05; }}
.fp-label {{ color: {STEEL}; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }}
.fp-vs {{ font-family: 'Oswald', sans-serif; font-size: 2rem; color: {STEEL}; text-align: center; padding-top: 2.6rem; }}
.fp-factor {{ margin: .15rem 0; }}
.fp-disclaimer {{ color: {STEEL}; font-style: italic; font-size: .9rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Preparing fight data (first start downloads it, ~30 s)…")
def prepare_data(commit: str | None) -> str | None:
    """Download + build the dataset if missing or built from a different upstream commit
    than the shipped model (a fresh clone, e.g. a cloud deploy, has no data/ folder)."""
    from src.fetch_data import ensure_data
    return ensure_data(commit)


@st.cache_resource(show_spinner="Loading model and fighter histories…")
def load_predictor(data_commit: str | None, model_mtime: float):
    # Keyed on the data commit and model file so a redeploy with a new model reloads it.
    from src.predict import Predictor
    return Predictor()


@st.cache_data
def load_json(name: str, mtime: float) -> dict:
    return json.loads((MODELS_DIR / name).read_text())


def model_json(name: str) -> dict:
    return load_json(name, (MODELS_DIR / name).stat().st_mtime)


st.markdown('<p class="fp-title">Fight <span>Predictor</span></p>'
            '<p class="fp-sub">MMA win-probability model built on historical UFCStats fight data</p>',
            unsafe_allow_html=True)

if not (MODELS_DIR / "model.joblib").exists():
    st.error("No trained model found. Run `python -m src.train` first, then restart the app.")
    st.stop()

evaluation = model_json("evaluation.json")
metadata = model_json("model_metadata.json")
try:
    data_commit = prepare_data((metadata.get("data_source") or {}).get("commit"))
except Exception as exc:  # network down on first start, upstream moved, ...
    st.error(f"Could not prepare the fight data: {exc}. Check the network connection and reload, or "
             "run `python -m src.fetch_data` then `python -m src.build_dataset` locally.")
    st.stop()
MODEL_KEY = (data_commit, (MODELS_DIR / "model.joblib").stat().st_mtime)
predictor = load_predictor(*MODEL_KEY)
roster = predictor.roster


# --------------------------------------------------------------------------- formatting
def fmt_height(inches) -> str:
    if inches is None or pd.isna(inches):
        return "unknown"
    return f"{int(inches) // 12}'{int(inches) % 12}\""


def fmt_inches(x) -> str:
    return "unknown" if x is None or pd.isna(x) else f'{x:.0f}"'


def fmt_pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.0%}"


def fmt_num(x, digits=2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def fmt_glicko(feat: dict) -> str:
    """Glicko rating with its 95% range (rating ± 1.96 x RD); the range is wide for
    debutants and grows during layoffs."""
    return f"{feat['glicko_rating']:.0f} ± {1.96 * feat['glicko_rd']:.0f}"


def record(p: dict) -> str:
    r = p["record"]
    s = f"{r['wins']}-{r['losses']}"
    return s + (f"-{r['draws_nc']} (D/NC)" if r["draws_nc"] else "")


def layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecebe8"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# --------------------------------------------------------------------------- fighter index
DIVISIONS = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
             "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]
WOMEN_DIVISIONS = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight"]
ACTIVE_DAYS = 730  # "active" = fought within the last two years of data


@st.cache_resource(show_spinner=False)
def fighter_index(data_commit: str | None, model_mtime: float) -> pd.DataFrame:
    """Roster plus gender, divisions, record and activity, for filtering and labels.
    Same cache key as load_predictor, so a redeploy with new data rebuilds it too."""
    p = load_predictor(data_commit, model_mtime)
    app = p.appearances.sort_values("event_date")
    app = app.assign(female=app["weight_class"].str.startswith("Women's"),
                     division=app["weight_class"].str.removeprefix("Women's "))
    named = app[app["division"].isin(DIVISIONS)]  # skip catch / open weight bouts
    idx = roster.set_index("fighter_id").join(pd.DataFrame({
        "female": app.groupby("fighter_id")["female"].any(),
        "division": named.groupby("fighter_id")["division"].last(),
        "divisions": named.groupby("fighter_id")["division"].agg(frozenset),
    }))
    idx = idx.join(p.profiles[["wins", "losses", "draws_nc", "stance"]])
    idx["divisions"] = idx["divisions"].apply(lambda d: d if isinstance(d, frozenset) else frozenset())
    idx["active"] = (p.as_of - idx["last_fight"]).dt.days <= ACTIVE_DAYS
    return idx.reset_index()


def division_label(row) -> str:
    if pd.isna(row["division"]):
        return "Catch/Open weight"
    return ("Women's " if row["female"] else "") + row["division"]


def record_short(row) -> str:
    s = f"{int(row['wins'])}-{int(row['losses'])}"
    return s + (f"-{int(row['draws_nc'])}" if row["draws_nc"] else "")


fx = fighter_index(*MODEL_KEY)
fx_by_name = fx.set_index("display_name")
LABELS = {r.display_name: f"{r.display_name}  ·  {division_label(fx_by_name.loc[r.display_name])}  ·  "
                          f"{record_short(fx_by_name.loc[r.display_name])}  ·  last fought {r.last_fight:%b %Y}"
          for r in fx.itertuples()}


# --------------------------------------------------------------------------- filters
FILTER_DEFAULTS = {"flt_gender": "All", "flt_status": "Active (last 2 yrs)", "flt_min_fights": 1,
                   "flt_divisions": [], "flt_any_division": False}
for _k, _v in FILTER_DEFAULTS.items():  # defaults live in session state so code can reset them
    st.session_state.setdefault(_k, _v)

with st.expander("Filter fighters", expanded=True, icon=":material/filter_list:"):
    f1, f2, f3 = st.columns([2, 3, 3])
    with f1:
        gender = st.segmented_control("Gender", ["All", "Men", "Women"], key="flt_gender") or "All"
    with f2:
        status = st.segmented_control("Status", ["Active (last 2 yrs)", "All-time"], key="flt_status") or "All-time"
    with f3:
        min_fights = st.slider("Minimum UFC fights", 1, 15, key="flt_min_fights")
    div_options = WOMEN_DIVISIONS if gender == "Women" else DIVISIONS
    divisions = st.pills("Weight class (none selected = all)", div_options, selection_mode="multi",
                         key="flt_divisions")
    any_division = st.checkbox("Also include fighters who previously fought in these weight classes",
                               key="flt_any_division")

mask = fx["n_fights"] >= min_fights
if gender != "All":
    mask &= fx["female"] == (gender == "Women")
if status != "All-time":
    mask &= fx["active"]
if divisions:
    chosen = set(divisions)
    if any_division:
        mask &= fx["divisions"].apply(lambda d: bool(d & chosen))
    else:
        mask &= fx["division"].isin(chosen)
options = fx.loc[mask].sort_values("display_name")["display_name"].tolist()

st.caption(f"{len(options):,} fighters match the filters · type in a box to search by name")
if len(options) < 2:
    st.warning("Fewer than two fighters match these filters. Widen the filters to pick a matchup.")
    st.stop()


# --------------------------------------------------------------------------- fighter pickers
def _sync_selection() -> list[str]:
    """Keep both selections valid for the current filters; return Fighter B's options."""
    state = st.session_state
    if state.get("fighter_a") not in options:
        state["fighter_a"] = DEFAULT_A if DEFAULT_A in options else options[0]
    opts_b = [n for n in options if n != state["fighter_a"]]
    if state.get("fighter_b") not in opts_b:
        same_div = [n for n in opts_b
                    if fx_by_name.loc[n, "division"] == fx_by_name.loc[state["fighter_a"], "division"]]
        state["fighter_b"] = DEFAULT_B if DEFAULT_B in opts_b else (same_div or opts_b)[0]
    return opts_b


def _swap():
    st.session_state["fighter_a"], st.session_state["fighter_b"] = (
        st.session_state["fighter_b"], st.session_state["fighter_a"])


def _random_matchup(pool: list[str]):
    """Two random fighters from the filtered pool, from the same division when possible."""
    rng = pd.Series(pool).sample(frac=1)
    a = rng.iloc[0]
    same = [n for n in rng.iloc[1:] if fx_by_name.loc[n, "division"] == fx_by_name.loc[a, "division"]]
    st.session_state["fighter_a"] = a
    st.session_state["fighter_b"] = same[0] if same else rng.iloc[1]


def fighter_summary(name: str) -> str:
    r = fx_by_name.loc[name]
    stance = r["stance"] if r["stance"] != "Unknown" else "stance unknown"
    return (f"{division_label(r)} · record {record_short(r)} · {int(r['n_fights'])} UFC fights · "
            f"{stance} · last fought {r['last_fight']:%d %b %Y}")


options_b = _sync_selection()
pick_a, pick_swap, pick_b = st.columns([10, 1, 10], vertical_alignment="center")
with pick_a:
    fighter_a = st.selectbox("Fighter A", options, key="fighter_a", format_func=LABELS.get)
    st.caption(fighter_summary(fighter_a))
with pick_swap:
    st.button("⇄", on_click=_swap, help="Swap fighters", width="stretch")
with pick_b:
    fighter_b = st.selectbox("Fighter B", options_b, key="fighter_b", format_func=LABELS.get)
    st.caption(fighter_summary(fighter_b))

btn_predict, btn_random, _ = st.columns([2, 2, 6])
with btn_predict:
    clicked = st.button("Predict Fight", type="primary", width="stretch")
with btn_random:
    st.button("Random matchup", on_click=_random_matchup, args=(options,), width="stretch",
              icon=":material/casino:", help="Pick two fighters from the filtered list (same division when possible)")

if clicked:
    if fighter_a == fighter_b:
        st.error("Choose two different fighters.")
    else:
        try:
            st.session_state["result"] = predictor.predict_fight(fighter_a, fighter_b)
        except (ValueError, KeyError) as exc:
            st.error(str(exc))

result = st.session_state.get("result")
# Discard a stale prediction if the selection changed since it was made.
if result and (result["fighter_a"]["display_name"], result["fighter_b"]["display_name"]) != (fighter_a, fighter_b):
    result = None

tab_predict, tab_card, tab_compare, tab_perf, tab_insights = st.tabs(
    ["Predict", "Upcoming Card", "Fighter Comparison", "Model Performance", "Model Insights"],
    key="main_tabs", on_change="rerun")


# --------------------------------------------------------------------------- tab 1: predict
def factor_text(group: str) -> str:
    return group[0].upper() + group[1:]


METHOD_LABELS = {"KO/TKO": "KO/TKO", "SUB": "Submission", "DEC": "Decision"}
WEIGHT_CLASS_CHOICES = ([d for d in DIVISIONS] + [f"Women's {d}" for d in WOMEN_DIVISIONS] + ["Catch Weight"])


def method_section(result: dict):
    """Method-of-victory breakdown with bout-context controls (they don't change the win probability)."""
    pa, pb = result["fighter_a"], result["fighter_b"]
    fid_a, fid_b = pa["fighter_id"], pb["fighter_id"]
    ctx = result["method"]["context"]
    st.subheader("Method of victory")
    c_wc, c_rd, c_title = st.columns([4, 2, 2], vertical_alignment="bottom")
    choices = WEIGHT_CLASS_CHOICES if ctx["weight_class"] in WEIGHT_CLASS_CHOICES else \
        WEIGHT_CLASS_CHOICES + [ctx["weight_class"]]
    with c_wc:
        wc = st.selectbox("Bout weight class", choices, index=choices.index(ctx["weight_class"]),
                          key=f"m_wc_{fid_a}_{fid_b}",
                          help="Defaults to the fighters' shared division, or the heavier one.")
    with c_rd:
        rounds = st.segmented_control("Rounds", [3, 5], default=ctx["scheduled_rounds"],
                                      key=f"m_rd_{fid_a}_{fid_b}") or 3
    with c_title:
        title = st.checkbox("Title fight", value=ctx["is_title"], key=f"m_title_{fid_a}_{fid_b}")
    m = predictor.predict_method(fid_a, fid_b, result["prob_a"], predictor.bout_context(fid_a, fid_b, wc, rounds, title))

    ml = m["most_likely"]
    who = pa["name"] if ml["winner"] == "A" else pb["name"]
    st.markdown(f"Most likely single outcome: **{who} by {METHOD_LABELS[ml['method']]}** "
                f"({ml['prob']:.1%})")

    k1, k2, k3 = st.columns(3)
    for col, meth in zip((k1, k2, k3), METHOD_LABELS):
        col.metric(f"Ends by {METHOD_LABELS[meth]}", f"{m['overall'][meth]:.1%}")

    labels = list(METHOD_LABELS.values())
    fig = go.Figure()
    for side, p, colour in [("A", pa, RED), ("B", pb, GOLD)]:
        vals = [m[side][k] for k in METHOD_LABELS]
        fig.add_bar(x=labels, y=vals, name=p["name"], marker_color=colour,
                    text=[f"{v:.1%}" for v in vals], textposition="outside")
    fig.update_layout(barmode="group", yaxis=dict(tickformat=".0%", rangemode="tozero"),
                      title=dict(text="Probability of each outcome (all six sum to 100%)", font=dict(size=14)))
    st.plotly_chart(layout(fig, 330), width="stretch")
    for side, p in [("A", pa), ("B", pb)]:
        g = m[f"given_{side}_wins"]
        st.caption(f"If {p['name']} wins: " + " · ".join(f"{METHOD_LABELS[k]} {g[k]:.0%}" for k in METHOD_LABELS))
    st.caption("Weight class, rounds and title only affect the method breakdown, not the win probability. "
               "DQs and other rare outcomes (under 1% of fights) are not modelled.")


with tab_predict:
    if result is None:
        st.info("Pick two fighters and press **Predict Fight**.")
    else:
        pa, pb = result["fighter_a"], result["fighter_b"]
        ca, cvs, cb = st.columns([5, 1, 5])
        for col, p, prob, cls, colour in [(ca, pa, result["prob_a"], "a", RED), (cb, pb, result["prob_b"], "b", GOLD)]:
            col.markdown(
                f'<div class="fp-card {cls}"><p class="fp-name">{p["name"]}</p>'
                f'<div class="fp-label">Win probability</div>'
                f'<div class="fp-prob" style="color:{colour}">{prob:.1%}</div>'
                f'<div class="fp-label">Prior UFC fights: {p["n_fights"]} &nbsp;·&nbsp; Record {record(p)}</div></div>',
                unsafe_allow_html=True)
        cvs.markdown('<div class="fp-vs">VS</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_bar(y=[""], x=[result["prob_a"]], orientation="h", marker_color=RED, name=pa["name"],
                    text=f"{result['prob_a']:.1%}", textposition="inside", insidetextanchor="start")
        fig.add_bar(y=[""], x=[result["prob_b"]], orientation="h", marker_color=GOLD, name=pb["name"],
                    text=f"{result['prob_b']:.1%}", textposition="inside", insidetextanchor="end")
        fig.add_vline(x=0.5, line_dash="dot", line_color="#ecebe8")
        fig.update_layout(barmode="stack", bargap=0.35, xaxis=dict(range=[0, 1], tickformat=".0%"))
        fig = layout(fig, 150)
        fig.update_layout(legend=dict(traceorder="normal", x=0, xanchor="left"))
        st.plotly_chart(fig, width="stretch")

        st.markdown(f'<p class="fp-disclaimer">{DISCLAIMER} Fighter profiles are as of '
                    f'{result["as_of"]} and include every UFC fight in the data before that date. Model: {result["model_type"].replace("_", " ")}.</p>',
                    unsafe_allow_html=True)
        for w in result["warnings"]:
            st.warning(w)

        if result.get("method"):
            method_section(result)

        st.subheader("Model factors")
        st.caption("How much each group of matchup differences moved this model's estimate "
                   "(probability points, measured by neutralising that group). These are statistical "
                   "associations the model uses, not causes of a result.")
        factors = [f for f in result["factors"] if abs(f["impact"]) >= 0.005]
        fa, fb = st.columns(2)
        for col, side, p in [(fa, "A", pa), (fb, "B", pb)]:
            items = [f for f in factors if f["favors"] == side]
            col.markdown(f"**Factors favouring {p['name']}:**")
            if not items:
                col.markdown("_none of note_")
            for f in items:
                col.markdown(f'<p class="fp-factor">+ {factor_text(f["group"])} '
                             f'<span style="color:{STEEL}">({abs(f["impact"]) * 100:.1f} pts)</span></p>',
                             unsafe_allow_html=True)

        st.subheader("Tale of the tape")
        fa_, fb_ = pa["features"], pb["features"]
        tape = pd.DataFrame({
            "": ["Age", "Height", "Reach", "Stance", "UFC record", "Prior UFC fights",
                 "Wins in last 5", "Sig. strikes landed / min", "Sig. strikes absorbed / min",
                 "Takedowns / 15 min", "Takedown defence", "Days since last fight",
                 "Opponent-strength rating (Glicko)"],
            pa["display_name"]: [fmt_num(pa["age"], 1) if pa["age"] else "unknown", fmt_height(pa["height_in"]),
                         fmt_inches(pa["reach_in"]), pa["stance"], record(pa), pa["n_fights"],
                         int(fa_["wins_last5"]), fmt_num(fa_["slpm"]), fmt_num(fa_["sapm"]),
                         fmt_num(fa_["td15"]), fmt_pct(fa_["td_def"]), pa["days_since_last"] or "—",
                         fmt_glicko(fa_)],
            pb["display_name"]: [fmt_num(pb["age"], 1) if pb["age"] else "unknown", fmt_height(pb["height_in"]),
                         fmt_inches(pb["reach_in"]), pb["stance"], record(pb), pb["n_fights"],
                         int(fb_["wins_last5"]), fmt_num(fb_["slpm"]), fmt_num(fb_["sapm"]),
                         fmt_num(fb_["td15"]), fmt_pct(fb_["td_def"]), pb["days_since_last"] or "—",
                         fmt_glicko(fb_)],
        }).astype(str)
        st.dataframe(tape, hide_index=True, width="stretch", height=35 * len(tape) + 38)


# --------------------------------------------------------------------------- tab: upcoming card
@st.cache_data(ttl=3600, show_spinner="Fetching scheduled events from Wikipedia…")
def cached_events() -> pd.DataFrame:
    from src.upcoming import fetch_scheduled_events
    return fetch_scheduled_events()


@st.cache_data(ttl=3600, show_spinner="Fetching the fight card…")
def cached_card(page: str) -> pd.DataFrame:
    from src.upcoming import fetch_fight_card
    return fetch_fight_card(page)


def card_predictions(card: pd.DataFrame) -> pd.DataFrame:
    from src.upcoming import match_fighter
    names = roster.set_index("fighter_id")["display_name"]
    rows = []
    for r in card.itertuples():
        ids, notes = [], []
        for name in (r.fighter_1, r.fighter_2):
            fid, how = match_fighter(name, r.weight_class, predictor.fighters, roster)
            ids.append(fid)
            if fid.startswith("new_"):
                notes.append(f"{name}: not in UFCStats data")
        is_title = "championship" in (r.notes or "").lower()
        five = is_title or r.Index == 0  # main event (first bout listed) or title fight
        p = predictor.predict_ids(*ids, weight_class=r.weight_class, scheduled_rounds=5 if five else 3,
                                  is_title=is_title)
        m = p["method"]
        for name, fid, n in ((r.fighter_1, ids[0], p["a_prior_fights"]), (r.fighter_2, ids[1], p["b_prior_fights"])):
            if fid.startswith("new_"):
                continue
            if n == 0:
                notes.append(f"{name}: UFC debut")
            elif n < 3:
                notes.append(f"{name}: {n} UFC fight{'s' if n > 1 else ''}")
        rows.append({
            "segment": r.segment, "Weight class": r.weight_class,
            "Fighter 1": r.fighter_1, "F1 win": p["prob_a"], "F2 win": p["prob_b"], "Fighter 2": r.fighter_2,
            "Most likely result": (f"{r.fighter_1 if m['most_likely']['winner'] == 'A' else r.fighter_2} by "
                                   f"{METHOD_LABELS[m['most_likely']['method']]} ({m['most_likely']['prob']:.0%})"
                                   if m else "—"),
            "KO / Sub / Dec": (" / ".join(f"{m['overall'][k]:.0%}" for k in METHOD_LABELS) if m else "—"),
            "Bout": f"{r.weight_class} · {5 if five else 3} rds" + (" · title" if is_title else ""),
            "Notes": "; ".join([n for n in notes] + ([r.notes] if r.notes else [])) or "—",
            "id_1": ids[0], "id_2": ids[1],
            "display_1": names.get(ids[0]), "display_2": names.get(ids[1]),
        })
    return pd.DataFrame(rows)


def _open_bout(display_a: str, display_b: str):
    """Send a card bout to the Predict tab: clear filters, select both fighters, predict."""
    st.session_state.update({
        **FILTER_DEFAULTS, "flt_status": "All-time",
        "fighter_a": display_a, "fighter_b": display_b, "main_tabs": "Predict",
        "result": predictor.predict_fight(display_a, display_b),
    })


with tab_card:
    st.markdown("#### Upcoming card")
    st.caption("Scheduled events and bouts come from Wikipedia, fetched only when you ask, and cached "
               "for an hour. Cards change often, so check official sources before relying on them.")
    if not st.session_state.get("upcoming_on"):
        st.button("Load upcoming events", icon=":material/event:", type="primary",
                  on_click=lambda: st.session_state.update(upcoming_on=True))
    else:
        try:
            events = cached_events()
        except Exception as exc:  # network down, API change, ...
            events = None
            st.error(f"Could not fetch scheduled events from Wikipedia ({exc}).")
        if events is not None and events.empty:
            st.info("No scheduled events found.")
        elif events is not None:
            c_ev, c_refresh = st.columns([5, 1], vertical_alignment="bottom")
            with c_ev:
                ev_i = st.selectbox("Event", range(len(events)), key="card_event",
                                    format_func=lambda i: f"{events.event[i]}  ·  {events.date[i]:%a %d %b %Y}")
            with c_refresh:
                if st.button("Refresh", icon=":material/refresh:", width="stretch"):
                    cached_events.clear()
                    cached_card.clear()
                    st.rerun()
            event = events.iloc[ev_i]
            try:
                card = cached_card(event.page)
            except Exception as exc:
                card = None
                st.error(f"Could not fetch the fight card for {event.event} ({exc}).")
            if card is not None and card.empty:
                st.info("No bouts announced on Wikipedia for this event yet.")
            elif card is not None:
                preds = card_predictions(card)
                days = (event.date - pd.Timestamp.today().date()).days
                when = "today" if days == 0 else f"in {days} day{'s' if days != 1 else ''}"
                st.markdown(f"**{event.event}** · {event.date:%A %d %B %Y} ({when}) · {len(preds)} bouts")
                show_cols = ["Bout", "Fighter 1", "F1 win", "F2 win", "Fighter 2", "Most likely result",
                             "KO / Sub / Dec", "Notes"]
                pct = dict(format="percent", min_value=0.0, max_value=1.0)
                for segment, part in preds.groupby("segment", sort=False):
                    if segment:
                        st.markdown(f"<span class='fp-label'>{segment}</span>", unsafe_allow_html=True)
                    st.dataframe(
                        part[show_cols], hide_index=True, width="stretch",
                        height=35 * (len(part) + 1) + 3,  # show every bout, no inner scrolling
                        column_config={
                            "Bout": st.column_config.TextColumn(width="medium"),
                            "F1 win": st.column_config.ProgressColumn("F1 win", color=RED, width="small", **pct),
                            "F2 win": st.column_config.ProgressColumn("F2 win", color=GOLD, width="small", **pct),
                            "KO / Sub / Dec": st.column_config.TextColumn(width=125),
                            "Notes": st.column_config.TextColumn(width="medium"),
                        })

                openable = preds[preds["display_1"].notna() & preds["display_2"].notna()]
                if len(openable):
                    c_bout, c_open = st.columns([5, 2], vertical_alignment="bottom")
                    with c_bout:
                        b_i = st.selectbox("Full breakdown for a bout", openable.index, key="card_bout",
                                           format_func=lambda i: f"{preds.at[i, 'Fighter 1']} vs. "
                                                                 f"{preds.at[i, 'Fighter 2']}")
                    with c_open:
                        st.button("Open in Predict tab", icon=":material/open_in_new:", width="stretch",
                                  on_click=_open_bout,
                                  args=(openable.at[b_i, "display_1"], openable.at[b_i, "display_2"]))
                st.markdown(f'<p class="fp-disclaimer">{DISCLAIMER} Fighters not in the UFCStats data are '
                            "scored as debutants with neutral (league-average) profiles.</p>",
                            unsafe_allow_html=True)


# --------------------------------------------------------------------------- tab 2: comparison
with tab_compare:
    pa, pb = predictor.profile(fighter_a), predictor.profile(fighter_b)
    fa_, fb_ = pa["features"], pb["features"]
    st.caption("Career-to-date figures use only completed UFC fights in the dataset. Rates are lightly "
               "shrunk toward league averages for fighters with little history.")
    rows = [
        ("Physical", "Age", fmt_num(pa["age"], 1) if pa["age"] else "unknown",
         fmt_num(pb["age"], 1) if pb["age"] else "unknown"),
        ("Physical", "Height", fmt_height(pa["height_in"]), fmt_height(pb["height_in"])),
        ("Physical", "Reach", fmt_inches(pa["reach_in"]), fmt_inches(pb["reach_in"])),
        ("Physical", "Stance", pa["stance"], pb["stance"]),
        ("Record", "UFC record (W-L)", record(pa), record(pb)),
        ("Record", "Prior UFC fights", pa["n_fights"], pb["n_fights"]),
        ("Record", "Wins in last 5", int(fa_["wins_last5"]), int(fb_["wins_last5"])),
        ("Record", "Recent win rate (last 5, shrunk)", fmt_pct(fa_["win_pct_last5"]), fmt_pct(fb_["win_pct_last5"])),
        ("Record", "Current streak", f"{fa_['streak']:+.0f}", f"{fb_['streak']:+.0f}"),
        ("Record", "Opponent-strength rating (Glicko, ± 95% range)", fmt_glicko(fa_), fmt_glicko(fb_)),
        ("Striking", "Sig. strikes landed / min", fmt_num(fa_["slpm"]), fmt_num(fb_["slpm"])),
        ("Striking", "Sig. strikes absorbed / min", fmt_num(fa_["sapm"]), fmt_num(fb_["sapm"])),
        ("Striking", "Sig. strike differential / min", fmt_num(fa_["sig_diff_pm"]), fmt_num(fb_["sig_diff_pm"])),
        ("Striking", "Sig. strike accuracy", fmt_pct(fa_["sig_acc"]), fmt_pct(fb_["sig_acc"])),
        ("Striking", "Sig. strike defence", fmt_pct(fa_["sig_def"]), fmt_pct(fb_["sig_def"])),
        ("Grappling", "Takedowns / 15 min", fmt_num(fa_["td15"]), fmt_num(fb_["td15"])),
        ("Grappling", "Takedown accuracy", fmt_pct(fa_["td_acc"]), fmt_pct(fb_["td_acc"])),
        ("Grappling", "Takedown defence", fmt_pct(fa_["td_def"]), fmt_pct(fb_["td_def"])),
        ("Grappling", "Submission attempts / 15 min", fmt_num(fa_["sub15"]), fmt_num(fb_["sub15"])),
        ("Grappling", "Control time share", fmt_pct(fa_["ctrl_pct"]), fmt_pct(fb_["ctrl_pct"])),
        ("Activity", "Days since last fight", pa["days_since_last"] or "—", pb["days_since_last"] or "—"),
    ]
    # Column keys must be unique even when two different fighters share a real name
    # (e.g. two "Bruno Silva"s) -- display_name is disambiguated, name is not.
    comp = pd.DataFrame(rows, columns=["Category", "Statistic", pa["display_name"], pb["display_name"]]).astype(str)
    st.dataframe(comp, hide_index=True, width="stretch", height=35 * len(comp) + 38)

    chart_stats = [("Strikes landed/min", "slpm"), ("Strikes absorbed/min", "sapm"),
                   ("Takedowns/15", "td15"), ("Sub att/15", "sub15")]
    chart_pcts = [("Strike acc.", "sig_acc"), ("Strike def.", "sig_def"),
                  ("TD acc.", "td_acc"), ("TD def.", "td_def"), ("Control", "ctrl_pct")]
    c1, c2 = st.columns(2)
    for col, stats, title, fmt in [(c1, chart_stats, "Rates", ".2f"), (c2, chart_pcts, "Percentages", ".0%")]:
        fig = go.Figure()
        for p, f, colour in [(pa, fa_, RED), (pb, fb_, GOLD)]:
            fig.add_bar(x=[s[0] for s in stats], y=[f[s[1]] for s in stats], name=p["name"],
                        marker_color=colour, texttemplate=f"%{{y:{fmt}}}", textposition="outside")
        fig.update_layout(title=title, barmode="group")
        if fmt == ".0%":
            fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
        col.plotly_chart(layout(fig, 340), width="stretch")

    r1, r2 = st.columns(2)
    for col, p in [(r1, pa), (r2, pb)]:
        col.markdown(f"**Recent UFC fights — {p['name']}**")
        rf = predictor.recent_fights(p["fighter_id"], 5)
        if rf.empty:
            col.caption("No UFC fights on record.")
        else:
            col.dataframe(rf.drop(columns=["event"]).rename(columns={"sig_str": "sig. str (own / opp)"}).astype(str),
                          hide_index=True, width="stretch")


# --------------------------------------------------------------------------- tab 3: performance
with tab_perf:
    sel = evaluation["selected_model"]
    periods = metadata["periods"]
    tm = metadata["test_metrics"]
    st.markdown(f"**Selected model:** {sel.replace('_', ' ')} &nbsp;·&nbsp; "
                f"**Calibration:** {metadata['calibration_method']} &nbsp;·&nbsp; "
                f"**Trained:** {metadata['trained_at_utc'][:10]}")
    st.caption(evaluation["selection_rule"].capitalize() + ". The test period was used only once, "
               "after the model had been chosen.")

    pf = metadata.get("production_fit") or {}
    if pf.get("refit_on_all_data"):
        st.info(f"**Deployed model:** this configuration refit on all {pf['n_fights']:,} fights from "
                f"{pf['start']} to {pf['end']}, so it also learns from the most recent fights. The metrics "
                f"below come from the evaluation run: trained {periods['train']['start']} → "
                f"{periods['train']['end']}, chosen on {periods['validation']['start']} → "
                f"{periods['validation']['end']}, scored once on {periods['test']['start']} → "
                f"{periods['test']['end']} (`models/evaluated_model.joblib`).")

    policy = metadata.get("staleness_policy")
    if policy:
        # Age = time since the last fight the deployed weights were fit on, NOT since
        # train.py last ran: retraining on unchanged data makes nothing fresher.
        fit_end = pf.get("end") or periods["validation"]["end"]
        age_days = (datetime.now(timezone.utc).date() - datetime.fromisoformat(fit_end).date()).days
        profiles_through = result["as_of"] if result else metadata["data_source"]["latest_fight_date"]
        if pf.get("refit_on_all_data"):
            refresh = ("To refresh it, run `python -m src.fetch_data --latest` then `python -m src.train` "
                       "(the scheduled refresh workflow does this weekly).")
        else:
            refresh = ("To refresh it, move `VALIDATION_START` and `TEST_START` forward in `src/config.py`, "
                       "then run `python -m src.train`; retraining with unchanged split dates refits on the "
                       "same data.")
        if age_days >= policy["retrain_after_days"]:
            st.error(f"The model learned from fights up to {fit_end} ({age_days} days ago), past the "
                     f"{policy['retrain_after_days']}-day retrain guideline. Fighter profiles are current "
                     f"(through {profiles_through}), but the model's weights haven't learned from any fight "
                     f"since {fit_end}. {refresh}")
        elif age_days >= policy["recalibrate_after_days"]:
            st.warning(f"The model learned from fights up to {fit_end} ({age_days} days ago), past the "
                       f"{policy['recalibrate_after_days']}-day recalibration guideline (retrain guideline: "
                       f"{policy['retrain_after_days']} days). {refresh}")
        else:
            st.caption(f"The model learned from fights up to {fit_end} ({age_days} days ago), within the "
                       f"{policy['recalibrate_after_days']}-day recalibration guideline.")

    pcols = st.columns(3)
    for col, key, label in zip(pcols, ["train", "validation", "test"], ["Training", "Validation", "Test"]):
        p = periods[key]
        col.metric(f"{label} period", f"{p['n_fights']:,} fights")
        col.caption(f"{p['start']} → {p['end']}")

    m = st.columns(4)
    m[0].metric("Test accuracy", f"{tm['accuracy']:.1%}")
    m[1].metric("Test log loss", f"{tm['log_loss']:.4f}", help="0.6931 = always predicting 50%. Lower is better.")
    m[2].metric("Test Brier score", f"{tm['brier']:.4f}", help="0.25 = always predicting 50%. Lower is better.")
    m[3].metric("Test ROC-AUC", f"{tm['roc_auc']:.3f}")

    st.subheader("All models on the test period")
    res = pd.DataFrame(evaluation["test_results"])
    res["model"] = res["model"].str.replace("_", " ")
    st.dataframe(res.style.format({"accuracy": "{:.1%}", "log_loss": "{:.4f}", "brier": "{:.4f}",
                                   "roc_auc": "{:.3f}"}), hide_index=True, width="stretch")
    st.caption("Baselines: always 50%; logistic regression on win-rate difference; logistic regression "
               "on experience + win rate. Lower log loss / Brier is better.")

    diag = evaluation["test_diagnostics"][sel]
    g1, g2 = st.columns(2)
    cal = diag["calibration"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=STEEL), name="Perfect calibration")
    fig.add_scatter(x=cal["mean_predicted"], y=cal["fraction_positive"], mode="lines+markers",
                    line=dict(color=RED, width=3), name=sel.replace("_", " "))
    fig.update_layout(title="Calibration (test period)", xaxis_title="Predicted P(Fighter A wins)",
                      yaxis_title="Observed win rate")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    g1.plotly_chart(layout(fig, 380), width="stretch")

    cm = diag["confusion_matrix"]
    fig = go.Figure(go.Heatmap(z=cm["matrix"], x=[f"Pred: {l}" for l in cm["labels"]],
                               y=[f"Actual: {l}" for l in cm["labels"]], colorscale=[[0, "#1d1f24"], [1, RED]],
                               text=cm["matrix"], texttemplate="%{text}", showscale=False))
    fig.update_layout(title="Confusion matrix (threshold 50%)")
    fig.update_yaxes(autorange="reversed")
    g2.plotly_chart(layout(fig, 380), width="stretch")

    h = diag["probability_histogram"]
    edges = h["bin_edges"]
    fig = go.Figure(go.Bar(x=[(edges[i] + edges[i + 1]) / 2 for i in range(len(h["counts"]))], y=h["counts"],
                           width=0.045, marker_color=GOLD))
    fig.update_layout(title="Distribution of predicted P(Fighter A wins), test period",
                      xaxis_title="Predicted probability", yaxis_title="Fights")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(layout(fig, 300), width="stretch")
    gap = diag.get("mean_abs_reversal_gap_before_symmetrisation")
    if gap is not None:
        st.caption(f"Before symmetric averaging, swapping fighters changed P(A) + P(B) away from 1 by "
                   f"{gap:.4f} on average; the app averages both orientations so the two always sum to 100%.")

    mm = evaluation.get("method_model")
    if mm:
        st.subheader("Method-of-victory model")
        st.caption(f"Multinomial logistic regression for KO/TKO vs submission vs decision, given who wins. "
                   f"Trained on {mm['n_train']:,} fights ({mm['min_train_date'][:4]}–2021, after the shift toward "
                   f"decisions had settled); same chronological split and the same pre-fight features. "
                   f"Scored once on {mm['n_test']:,} test fights.")
        t = mm["test"]
        mt = pd.DataFrame({
            "Model": ["Method model", "Baseline: weight-class method rates", "Baseline: overall method rates"],
            "Log loss": [t["model"]["log_loss"], t["baseline_weight_class_rates"]["log_loss"],
                         t["baseline_overall_rates"]["log_loss"]],
            "Accuracy (top method)": [t["model"]["accuracy"], t["baseline_weight_class_rates"]["accuracy"],
                                      t["baseline_overall_rates"]["accuracy"]],
        })
        m1, m2 = st.columns([3, 2])
        m1.dataframe(mt.style.format({"Log loss": "{:.4f}", "Accuracy (top method)": "{:.1%}"}),
                     hide_index=True, width="stretch")
        shares = pd.DataFrame({"Predicted (avg)": mm["test_mean_predicted"], "Actual": mm["test_actual_share"]})
        shares.index = [METHOD_LABELS[k] for k in shares.index]
        m2.dataframe(shares.style.format("{:.1%}"), width="stretch")
        j = mm.get("test_joint_six_way")
        if j:
            st.caption(f"Full six-way outcome (winner × method) test log loss: {j['model_log_loss']:.3f}, versus "
                       f"{j['baseline_winner_model_x_method_rates_log_loss']:.3f} for the win model × average method "
                       f"rates and {j['baseline_uniform_log_loss']:.3f} for a uniform guess (lower is better).")


# --------------------------------------------------------------------------- tab 4: insights
with tab_insights:
    imp = evaluation["importance"]
    perm = imp["group_permutation_log_loss_increase_validation"]
    i1, i2 = st.columns(2)
    pdf = pd.Series(perm).sort_values()
    fig = go.Figure(go.Bar(x=pdf.values, y=pdf.index, orientation="h",
                           marker_color=[RED if v > 0 else STEEL for v in pdf.values]))
    fig.update_layout(title="Feature-group importance (validation)",
                      xaxis_title="Increase in log loss when the group is shuffled")
    i1.plotly_chart(layout(fig, 520), width="stretch")

    coefs = pd.Series(imp["logistic_regression_standardized_coefficients"])
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=[c.removesuffix("_diff") + " (A−B)" for c in top.index], orientation="h",
                           marker_color=[RED if v > 0 else GOLD for v in top.values]))
    fig.update_layout(title="Logistic regression: top standardized coefficients",
                      xaxis_title="Log-odds change per 1 SD of the difference (+ favours A)")
    i2.plotly_chart(layout(fig, 520), width="stretch")
    st.caption("Importance reflects what the model relies on given correlated inputs, not causal effects. "
               "Correlated features (e.g. career vs. last-5 striking) share credit unpredictably.")

    st.subheader("How predictions are produced")
    st.markdown(
        """
1. **Pre-fight histories only.** For every historical fight, each fighter's profile is built from fights on
   *strictly earlier* dates: career-to-date, last-3 and last-5 windows for record, striking, grappling,
   finishing and activity. Age is computed on the fight date. For a new matchup, the same code builds each
   fighter's profile as of the day after the latest fight in the data.
2. **Matchup differences.** The model sees Fighter A minus Fighter B for each statistic (reach difference,
   striking-differential difference, …), so it learns about matchups rather than about column positions.
3. **Order-neutral training.** Training fights are randomly but deterministically oriented and also mirrored
   (B vs A), because the source lists the winner first about two-thirds of the time.
4. **Calibration and symmetry.** Raw probabilities are recalibrated on the validation period, then averaged
   over both orientations: P(A) = ½·[p(A,B) + 1 − p(B,A)], so P(A) + P(B) = 100% exactly.
5. **Chronological validation.** Models are trained on older fights, tuned and calibrated on 2022–2023, and
   scored once on 2024 onward, mimicking real forward-looking use.
"""
    )
    st.subheader("Limitations")
    st.markdown(
        """
- **UFC-only history.** UFCStats covers UFC (and a few related) bouts only. Regional, Bellator, PFL, ONE or
  amateur records are invisible, so newly signed fighters look like debutants regardless of experience.
- **Static physical data.** Height, reach, stance and date of birth come from a current snapshot of each
  fighter's profile, not values recorded at fight time.
- **Missing data is not random.** Fighters with short UFC careers are much more likely to have missing reach
  or date of birth. To avoid leaking that future information, missing values are filled with neutral values
  and the model never sees a missing-data flag. Those fighters' predictions are less informed.
- **Not modelled:** weight class, title fights, 5-round bouts, short-notice replacements, weight cuts,
  injuries, camp changes, judging, betting odds and opponent quality (no strength-of-schedule rating).
- **Name matching.** Fight records identify fighters by name; a few renamed or same-named fighters were
  resolved by hand-checked rules, and a handful without profiles have no physical data.
- **Modest accuracy.** MMA is noisy: expect roughly 60–65% winner accuracy, and treat probabilities as
  rough estimates, never as betting advice.
"""
    )
