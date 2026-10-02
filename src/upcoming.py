"""Upcoming UFC cards, fetched on demand from Wikipedia.

UFCStats only lists completed fights in the CSV data, and its website now sits behind
a JavaScript browser check, so scheduled cards come from Wikipedia's public API:
  - "List of UFC events" -> section "Scheduled events" (event page + date)
  - each event page -> "Fight card" section ({{MMAevent bout ...}} templates)

Nothing here runs at app start-up: the app calls these functions only when the user
asks for upcoming cards, and caches the result.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

from src.build_dataset import NAME_ALIASES
from src.data_loader import normalize_name

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ufc-predictor-mvp/0.1 (local research app; https://github.com/Greco1899/scrape_ufc_stats data)"
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
DIVISION_LBS = {"Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
                "Lightweight": 155, "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205,
                "Heavyweight": 265}


# --------------------------------------------------------------------------- wikipedia
def _wikitext(page: str, section: int | None = None, timeout: float = 15) -> str:
    params = {"action": "parse", "page": page, "prop": "wikitext", "format": "json", "redirects": 1}
    if section is not None:
        params["section"] = section
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    if "error" in data:
        raise RuntimeError(data["error"].get("info", "Wikipedia API error"))
    return data["parse"]["wikitext"]["*"]


def _section_index(page: str, title: str) -> int:
    params = {"action": "parse", "page": page, "prop": "sections", "format": "json"}
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        sections = json.load(resp)["parse"]["sections"]
    for s in sections:
        if s["line"].strip().lower() == title.lower():
            return int(s["index"])
    raise RuntimeError(f"section {title!r} not found on {page!r}")


def _plain(text: str) -> str:
    """Wikitext -> plain text: drop refs/templates, keep link labels."""
    text = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>|'''?|\(c\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


class ParseFailure(RuntimeError):
    """Raised when a Wikipedia page looks like it has the expected content, but our
    regex-based wikitext parser extracted nothing from it -- almost always a sign that
    the page's template markup changed, not that there is genuinely nothing there. We'd
    rather fail loudly here than have the app silently show "no events"/"no bouts"."""


def parse_scheduled_events(wikitext: str, today: date) -> pd.DataFrame:
    """Scheduled UFC events on or after `today`, soonest first: columns event, page, date."""
    all_rows, future_rows = [], []
    for row in wikitext.split("|-")[1:]:
        link = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", row)
        d = re.search(r"\{\{dts\|(\d{4})\|(\w{3})\w*\|(\d{1,2})", row)
        if not (link and d) or d.group(2) not in MONTHS:
            continue
        when = date(int(d.group(1)), MONTHS[d.group(2)], int(d.group(3)))
        all_rows.append(when)
        if when >= today:
            future_rows.append({"event": (link.group(2) or link.group(1)).strip(),
                                "page": link.group(1).strip(), "date": when})
    # "{{dts|" templates are present (the section has rows) but none parsed as a valid
    # (link, date) pair: a parsing break, not "no events" -- distinct from the ordinary
    # case where every parsed row is simply in the past relative to `today`.
    if not all_rows and "{{dts|" in wikitext:
        raise ParseFailure(
            "Found date templates in Wikipedia's 'Scheduled events' table but could not parse any "
            "event out of them. The page format likely changed; src/upcoming.py needs an update."
        )
    return pd.DataFrame(future_rows, columns=["event", "page", "date"]).sort_values("date").reset_index(drop=True)


def fetch_scheduled_events(today: date | None = None) -> pd.DataFrame:
    """Scheduled UFC events (today or later), soonest first: columns event, page, date."""
    wt = _wikitext("List_of_UFC_events", _section_index("List_of_UFC_events", "Scheduled events"))
    return parse_scheduled_events(wt, today or date.today())


def parse_fight_card(wikitext: str) -> pd.DataFrame:
    """Bouts from {{MMAevent card}} / {{MMAevent bout}} templates, in card order."""
    rows, segment = [], ""
    for m in re.finditer(r"\{\{MMAevent (card|bout)\s*\|(.*?)\n?\}\}(?=\s*(?:\{\{|<ref|$|\n))",
                         wikitext, flags=re.S):
        kind, body = m.group(1), m.group(2)
        if kind == "card":
            segment = _plain(body.split("|")[0])
            continue
        fields = [f.strip() for f in re.split(r"\n\|", "\n|" + body.strip())[1:]]
        fields += [""] * (8 - len(fields))
        weight_class, f1, _vs, f2 = (_plain(x) for x in fields[:4])
        if f1 and f2:
            rows.append({"segment": segment, "weight_class": weight_class, "fighter_1": f1, "fighter_2": f2,
                         "notes": _plain(fields[7])})
    # A page with {{MMAevent bout ...}} templates that none of them parsed into a
    # fighter_1/fighter_2 pair means our field-order assumptions broke, not that the
    # event genuinely has zero bouts.
    if not rows and "{{MMAevent bout" in wikitext:
        raise ParseFailure(
            "Found '{{MMAevent bout' templates on this page but parsed zero bouts from them. The "
            "template's field layout likely changed; src/upcoming.py needs an update."
        )
    return pd.DataFrame(rows, columns=["segment", "weight_class", "fighter_1", "fighter_2", "notes"])


def fetch_fight_card(page: str) -> pd.DataFrame:
    return parse_fight_card(_wikitext(page))


# --------------------------------------------------------------------------- name matching
def _name_keys(name: str) -> list[str]:
    """Normalised lookup keys: as written, via the alias table, and without Jr./II/III."""
    keys = [normalize_name(name), normalize_name(NAME_ALIASES.get(name, name))]
    stripped = re.sub(r",?\s+(Jr\.?|Sr\.?|II|III|IV)$", "", name.strip())
    keys.append(normalize_name(stripped))
    return list(dict.fromkeys(k for k in keys if k))


def match_fighter(name: str, weight_class: str, fighters: pd.DataFrame, roster: pd.DataFrame) -> tuple[str, str]:
    """Return (fighter_id, how). Unknown fighters get 'new_<name>' (a debutant profile)."""
    division = (weight_class or "").replace("Women's ", "")
    target_lbs = DIVISION_LBS.get(division, np.nan)
    norm = fighters["name_norm"]
    for key in _name_keys(name):
        cands = fighters[norm == key]
        if len(cands) == 0:
            continue
        if len(cands) == 1:
            return cands["fighter_id"].iloc[0], "matched"
        # Several fighters share the name: prefer one who has fought in this division,
        # then the closest listed weight, then the most recently active.
        r = roster.set_index("fighter_id")
        same_div = [f for f in cands["fighter_id"] if f in r.index and r.loc[f, "last_weight_class"] == weight_class]
        if len(same_div) == 1:
            return same_div[0], "matched (same-name fighters: picked by division)"
        if not np.isnan(target_lbs) and cands["tott_weight_lbs"].notna().any():
            best = cands.loc[(cands["tott_weight_lbs"] - target_lbs).abs().idxmin(), "fighter_id"]
            return best, "matched (same-name fighters: picked by weight)"
        return cands["fighter_id"].iloc[0], "matched (ambiguous name)"
    return "new_" + normalize_name(name), "not in UFCStats data"
