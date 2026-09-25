"""Tests fuer die X01-Spiellogik: python -m pytest tests/ -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from game.x01 import X01Game, parse_field, checkout_suggestion  # noqa: E402


def throw(game, *fields):
    for f in fields:
        assert game.add_dart(parse_field(f)), f"Pfeil {f} abgelehnt"


def test_parse_field():
    assert parse_field("T20").points == 60
    assert parse_field("d16").points == 32 and parse_field("d16").is_double
    assert parse_field("20").field == "S20"
    assert parse_field("BULL").points == 25 and not parse_field("BULL").is_double
    assert parse_field("BULLSEYE").points == 50 and parse_field("BULLSEYE").is_double
    assert parse_field("MISS").points == 0
    with pytest.raises(ValueError):
        parse_field("T25")


def test_simple_turns_and_rotation():
    g = X01Game(["A", "B"], 501)
    throw(g, "T20", "T20", "T20")
    assert g.turn.complete and g.turn.points == 180
    assert not g.add_dart(parse_field("S1"))  # kein 4. Pfeil
    assert g.end_turn()
    assert g.players[0].score == 321 and g.current == 1
    assert g.players[0].stats.count_180 == 1
    throw(g, "S20", "S5", "S1")
    g.end_turn()
    assert g.players[1].score == 475 and g.current == 0


def test_bust_rules_double_out():
    g = X01Game(["A"], 301)
    throw(g, "T20", "T20", "T20")
    g.end_turn()  # 121
    throw(g, "T20", "T20")  # 1 Rest -> Bust
    assert g.turn.bust and g.turn.complete
    g.end_turn()
    assert g.players[0].score == 121, "Bust laesst den Rest unveraendert"
    throw(g, "T20", "S20", "S20")  # 21 Rest
    g.end_turn()
    throw(g, "S19", "S2")  # 0 Rest ohne Double -> Bust
    assert g.turn.bust
    g.end_turn()
    assert g.players[0].score == 21
    throw(g, "S20")  # Rest 1 -> sofort Bust
    assert g.turn.bust and g.turn.complete
    g.end_turn()
    assert g.players[0].score == 21
    throw(g, "S5", "D8")  # 21 - 5 - 16 = 0 mit Double -> Finish
    assert g.turn.finished and not g.turn.bust


def test_no_double_out():
    g = X01Game(["A"], 301, double_out=False)
    throw(g, "T20", "T20", "T20")
    g.end_turn()
    throw(g, "T20", "T20", "S1")  # 121 - 121 = 0 -> Finish ohne Double
    assert g.turn.finished
    throw_bust = X01Game(["A"], 301, double_out=False)
    throw(throw_bust, "T20", "T20", "T20")
    throw_bust.end_turn()
    throw(throw_bust, "T20", "T20", "S2")  # -1 -> Bust
    assert throw_bust.turn.bust


def test_leg_and_match():
    g = X01Game(["A", "B"], 301, legs_to_win=2)
    # A checkt 301 in 6 Pfeilen: 180 + 121 (T20 T11 D14)
    throw(g, "T20", "T20", "T20"); g.end_turn()
    throw(g, "S20", "S20", "S20"); g.end_turn()          # B
    throw(g, "T20", "T11", "D14")
    assert g.turn.finished
    g.end_turn()
    assert g.players[0].legs_won == 1 and g.leg == 2
    assert g.current == 1, "Anwurf wechselt zu B"
    assert all(p.score == 301 for p in g.players)
    assert g.players[0].stats.best_leg_darts == 6
    assert g.players[0].stats.highest_finish == 121
    # Leg 2: B beginnt, A gewinnt erneut -> Match
    throw(g, "S1", "S1", "S1"); g.end_turn()               # B
    throw(g, "T20", "T20", "T20"); g.end_turn()           # A 121
    throw(g, "S1", "S1", "S1"); g.end_turn()               # B
    throw(g, "T20", "T11", "D14"); g.end_turn()           # A finish
    assert g.over and g.winner == 0
    assert not g.add_dart(parse_field("S1"))


def test_undo_dart_and_turn():
    g = X01Game(["A", "B"], 501)
    throw(g, "T20", "T20")
    assert g.undo() and len(g.turn.darts) == 1
    throw(g, "T20", "T20"); g.end_turn()
    assert g.current == 1 and g.players[0].score == 321
    assert g.undo(), "letzte Aufnahme wieder oeffnen"
    assert g.current == 0 and g.players[0].score == 501 and len(g.turn.darts) == 3
    assert g.players[0].stats.darts == 0 and g.players[0].stats.count_180 == 0
    g.remove_last_dart()
    throw(g, "S20"); g.end_turn()
    assert g.players[0].score == 361


def test_undo_finished_leg():
    g = X01Game(["A", "B"], 301, legs_to_win=3)
    throw(g, "T20", "T20", "T20"); g.end_turn()
    throw(g, "S20", "S20", "S20"); g.end_turn()
    throw(g, "T20", "T11", "D14"); g.end_turn()
    assert g.leg == 2 and g.players[0].legs_won == 1
    assert g.undo()
    assert g.leg == 1 and g.players[0].legs_won == 0
    assert g.players[0].score == 121 and g.players[1].score == 241
    assert g.current == 0 and len(g.turn.darts) == 3 and g.turn.finished
    g.correct_dart(2, "S14")  # kein Finish mehr
    assert not g.turn.finished and g.remaining_for_current() == 121 - 60 - 33 - 14


def test_correct_dart_changes_bust():
    g = X01Game(["A"], 301)
    throw(g, "T20", "T20", "T20"); g.end_turn()  # 121
    throw(g, "T20", "T19", "S4")  # 121-60-57-4 = 0 ohne Double -> Bust
    assert g.turn.bust
    assert g.correct_dart(2, "MISS")  # Rest 4, kein Bust mehr
    assert not g.turn.bust and g.remaining_for_current() == 4
    assert g.correct_dart(2, "D2")  # 4 mit Double -> Finish
    assert g.turn.finished and not g.turn.bust
    assert g.correct_dart(2, "S2")  # Rest 2, offen
    assert not g.turn.finished and not g.turn.bust and g.remaining_for_current() == 2


def test_checkout_suggestions():
    assert checkout_suggestion(170, 3) == "T20 T20 BULLSEYE"
    assert checkout_suggestion(40, 1) == "D20"
    assert checkout_suggestion(41, 1) is None
    assert checkout_suggestion(41, 2) == "S9 D16"
    assert checkout_suggestion(1, 3) is None
    assert checkout_suggestion(171, 3) is None
    assert checkout_suggestion(100, 2) == "T20 D20"
    assert checkout_suggestion(50, 1) == "BULLSEYE"


def test_state_dict():
    g = X01Game(["Alice", "Bob"], 501)
    throw(g, "T20")
    d = g.to_dict()
    assert d["remaining"] == 441 and d["darts_left"] == 2
    assert d["players"][0]["name"] == "Alice"
    assert d["turn"]["darts"][0]["field"] == "T20"
