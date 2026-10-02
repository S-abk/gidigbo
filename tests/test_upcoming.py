"""Upcoming-card parsing and matching (offline: uses a stored wikitext sample)."""
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
