"""Upcoming-card parsing and matching (offline: uses a stored wikitext sample)."""
import pytest

from src.upcoming import match_fighter, parse_fight_card

SAMPLE = """==Fight card==
{{MMAevent}}
{{MMAevent card|Main card ([[Paramount+]])}}
{{MMAevent bout
|Women's Flyweight
|[[Natália Silva (fighter)|Natália Silva]]
|vs.
|[[Wang Cong]]
|
|
|
|For the vacant [[UFC Women's Flyweight Championship]].<ref>x</ref>
}}
{{MMAevent card|Preliminary card|header=no}}
{{MMAevent bout
|Featherweight
|[[Andre Fili]]
|vs.
|[[Kai Kamaka III]]
|
|
|
|
}}
{{MMAevent bout
|Women's Bantamweight
|Alice Pereira
|vs.
|Zzyzx Notarealfighter
|
|
|
|
}}
{{MMAevent end|notes=yes}}
"""


def test_parse_fight_card():
    card = parse_fight_card(SAMPLE)
    assert list(card["fighter_1"]) == ["Natália Silva", "Andre Fili", "Alice Pereira"]
    assert list(card["fighter_2"]) == ["Wang Cong", "Kai Kamaka III", "Zzyzx Notarealfighter"]
    assert list(card["segment"]) == ["Main card (Paramount+)", "Preliminary card", "Preliminary card"]
    assert card.loc[0, "weight_class"] == "Women's Flyweight"
    assert card.loc[0, "notes"] == "For the vacant UFC Women's Flyweight Championship."


def test_match_and_predict_card(predictor):
    card = parse_fight_card(SAMPLE)
    for r in card.itertuples():
        a, how_a = match_fighter(r.fighter_1, r.weight_class, predictor.fighters, predictor.roster)
        b, how_b = match_fighter(r.fighter_2, r.weight_class, predictor.fighters, predictor.roster)
        p = predictor.predict_ids(a, b)
        assert 0 < p["prob_a"] < 1 and abs(p["prob_a"] + p["prob_b"] - 1) < 1e-9
    # accents, aliases (Kai Kamaka III -> Kai Kamaka) and unknown debutants
    assert match_fighter("Natália Silva", "Women's Flyweight", predictor.fighters, predictor.roster)[1] == "matched"
    kai, _ = match_fighter("Kai Kamaka III", "Featherweight", predictor.fighters, predictor.roster)
    assert predictor.profile_row(kai)["n_fights"].iloc[0] >= 4
    new, how = match_fighter("Zzyzx Notarealfighter", "Lightweight", predictor.fighters, predictor.roster)
    assert new.startswith("new_") and predictor.profile_row(new)["n_fights"].iloc[0] == 0


def test_parse_scheduled_events_normal_and_past():
    from datetime import date
    from src.upcoming import parse_scheduled_events

    wt = "|-\n|[[UFC 332]]\n|{{dts|2026|Oct|3}}\n|-\n|[[UFC 333|UFC 333: X vs Y]]\n|{{dts|2026|Oct|24}}\n"
    future = parse_scheduled_events(wt, date(2026, 10, 1))
    assert list(future["event"]) == ["UFC 332", "UFC 333: X vs Y"]
    # All real events simply being in the past is a legitimate empty result, not an error.
    assert parse_scheduled_events(wt, date(2030, 1, 1)).empty
    assert parse_scheduled_events("nothing resembling the expected markup", date(2026, 10, 1)).empty


def test_parse_scheduled_events_raises_on_broken_markup():
    """A page containing {{dts|...}} templates that don't parse into any event is a
    silent-failure trap (template format changed) -- must raise, not return empty."""
    from datetime import date
    from src.upcoming import ParseFailure, parse_scheduled_events

    broken = "|-\n|not a wikilink line\n|{{dts|2026|Oct|3}} with no link markup next to it\n"
    with pytest.raises(ParseFailure):
        parse_scheduled_events(broken, date(2026, 10, 1))


def test_parse_fight_card_raises_on_broken_markup():
    from src.upcoming import ParseFailure, parse_fight_card

    broken = "{{MMAevent}}\n{{MMAevent card|Main card}}\n{{MMAevent bout\n}}\n"
    with pytest.raises(ParseFailure):
        parse_fight_card(broken)
    # A page with no MMAevent templates at all is legitimately "nothing to parse".
    assert parse_fight_card("no templates here at all").empty
