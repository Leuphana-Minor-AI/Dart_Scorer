"""X01-Spiellogik (301 / 501 / 701) fuer 1-4 Spieler.

Regeln:
- Jeder Spieler wirft pro Aufnahme bis zu drei Pfeile; die Punkte werden vom Rest abgezogen.
- Double-Out (Standard): Das Leg endet mit einem Double (oder Bullseye), das den Rest exakt
  auf 0 bringt. Bust, wenn der Rest unter 0 faellt, auf 1 faellt oder auf 0 ohne Double.
- Ohne Double-Out: Bust nur bei Rest < 0; Rest 0 beendet das Leg.
- Bei Bust bleibt der Rest der Aufnahme unveraendert, die Aufnahme ist beendet.
- Legs: Wer zuerst `legs_to_win` Legs gewinnt, gewinnt das Match. Anwurf wechselt je Leg.

Die Klasse kennt keine Kamera - Pfeile kommen als Feld-Strings ("T20", "D16", "S5",
"BULL", "BULLSEYE", "MISS") von der Erkennung oder aus der Oberflaeche.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SECTORS = list(range(1, 21))


@dataclass
class Dart:
    field: str          # "T20", "D16", "S5", "BULL", "BULLSEYE", "MISS"
    points: int
    sector: int         # 1-20, 25 (Bull), 0 (Miss)
    multiplier: int     # 1, 2, 3 (Bull: 1 = 25, 2 = 50); Miss: 0
    manual: bool = False
    uncertain: bool = False
    x_mm: Optional[float] = None   # Position auf dem Board (fuer die Anzeige)
    y_mm: Optional[float] = None

    @property
    def is_double(self) -> bool:
        return self.multiplier == 2 and self.sector != 0

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "points": self.points,
            "sector": self.sector,
            "multiplier": self.multiplier,
            "manual": self.manual,
            "uncertain": self.uncertain,
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
        }


def parse_field(text: str, manual: bool = False, uncertain: bool = False) -> Dart:
    """Feld-String in einen Dart umwandeln. Akzeptiert T20, D16, S5, 20 (=S20), BULL, BULLSEYE, DBULL, MISS."""
    t = text.strip().upper()
    if t in ("MISS", "0", "X", ""):
        return Dart("MISS", 0, 0, 0, manual, uncertain)
    if t in ("BULLSEYE", "DBULL", "DB", "50"):
        return Dart("BULLSEYE", 50, 25, 2, manual, uncertain)
    if t in ("BULL", "SBULL", "SB", "25"):
        return Dart("BULL", 25, 25, 1, manual, uncertain)
    mult = {"S": 1, "D": 2, "T": 3}.get(t[0])
    num = t[1:] if mult else t
    if mult is None:
        mult = 1
    if not num.isdigit() or int(num) not in SECTORS:
        raise ValueError(f"Ungueltiges Feld: {text!r}")
    sector = int(num)
    return Dart(f"{'SDT'[mult - 1]}{sector}", sector * mult, sector, mult, manual, uncertain)


# ---------------------------------------------------------------------------
# Checkout-Wege (Double-Out)
# ---------------------------------------------------------------------------

_PREFERRED = ["T20", "T19", "T18", "T17", "T16", "T15", "T14", "T13", "BULLSEYE",
              "S20", "S19", "S18", "S17", "S16", "S15", "S14", "S13", "S12", "S11", "S10",
              "S9", "S8", "S7", "S6", "S5", "S4", "S3", "S2", "S1", "BULL",
              "D20", "D19", "D18", "D17", "D16", "D15", "D14", "D13", "D12", "D11", "D10",
              "D9", "D8", "D7", "D6", "D5", "D4", "D3", "D2", "D1"]
_DOUBLES = ["D20", "D16", "D18", "D19", "D17", "D12", "D10", "D8", "D14", "D15", "D13", "D11",
            "D9", "D7", "D6", "D5", "D4", "D3", "D2", "D1", "BULLSEYE"]
_POINTS = {f: parse_field(f).points for f in _PREFERRED + _DOUBLES}

# Bewaehrte Standardwege haben Vorrang vor dem rechnerischen Ergebnis
_STANDARD_CHECKOUTS: Dict[int, str] = {
    170: "T20 T20 BULLSEYE", 167: "T20 T19 BULLSEYE", 164: "T20 T18 BULLSEYE", 161: "T20 T17 BULLSEYE",
    160: "T20 T20 D20", 158: "T20 T20 D19", 157: "T20 T19 D20", 156: "T20 T20 D18", 155: "T20 T19 D19",
    154: "T20 T18 D20", 153: "T20 T19 D18", 152: "T20 T20 D16", 151: "T20 T17 D20", 150: "T20 T18 D18",
    149: "T20 T19 D16", 148: "T20 T20 D14", 147: "T20 T17 D18", 146: "T20 T18 D16", 145: "T20 T15 D20",
    144: "T20 T20 D12", 143: "T20 T17 D16", 142: "T20 T14 D20", 141: "T20 T19 D12", 140: "T20 T20 D10",
    139: "T20 T13 D20", 138: "T20 T18 D12", 137: "T20 T19 D10", 136: "T20 T20 D8", 135: "T20 T17 D12",
    134: "T20 T14 D16", 133: "T20 T19 D8", 132: "T20 T16 D12", 131: "T20 T13 D16", 130: "T20 T20 D5",
    129: "T19 T16 D12", 128: "T18 T14 D16", 127: "T20 T17 D8", 126: "T19 T19 D6", 125: "BULL T20 D20",
    124: "T20 T16 D8", 123: "T19 T16 D9", 122: "T18 T18 D7", 121: "T20 T11 D14", 120: "T20 S20 D20",
    119: "T19 T12 D13", 118: "T20 S18 D20", 117: "T20 S17 D20", 116: "T20 S16 D20", 115: "T20 S15 D20",
    114: "T20 S14 D20", 113: "T20 S13 D20", 112: "T20 S12 D20", 111: "T20 S11 D20", 110: "T20 S10 D20",
    109: "T20 S9 D20", 108: "T20 S8 D20", 107: "T19 S10 D20", 106: "T20 S6 D20", 105: "T20 S5 D20",
    104: "T18 S10 D20", 103: "T19 S6 D20", 102: "T20 S10 D16", 101: "T17 S10 D20", 100: "T20 D20",
    99: "T19 S10 D16", 98: "T20 D19", 97: "T19 D20", 96: "T20 D18", 95: "T19 D19", 94: "T18 D20",
    93: "T19 D18", 92: "T20 D16", 91: "T17 D20", 90: "T18 D18", 89: "T19 D16", 88: "T20 D14",
    87: "T17 D18", 86: "T18 D16", 85: "T15 D20", 84: "T20 D12", 83: "T17 D16", 82: "BULLSEYE D16",
    81: "T19 D12", 80: "T20 D10", 79: "T19 D11", 78: "T18 D12", 77: "T19 D10", 76: "T20 D8",
    75: "T17 D12", 74: "T14 D16", 73: "T19 D8", 72: "T16 D12", 71: "T13 D16", 70: "T18 D8",
    69: "T19 D6", 68: "T20 D4", 67: "T17 D8", 66: "T10 D18", 65: "T19 D4", 64: "T16 D8",
    63: "T13 D12", 62: "T10 D16", 61: "T15 D8", 60: "S20 D20", 59: "S19 D20", 58: "S18 D20",
    57: "S17 D20", 56: "T16 D4", 55: "S15 D20", 54: "S14 D20", 53: "S13 D20", 52: "S12 D20",
    51: "S11 D20", 50: "S10 D20", 49: "S9 D20", 48: "S16 D16", 47: "S15 D16", 46: "S6 D20",
    45: "S13 D16", 44: "S12 D16", 43: "S11 D16", 42: "S10 D16", 41: "S9 D16", 40: "D20",
    39: "S7 D16", 38: "D19", 37: "S5 D16", 36: "D18", 35: "S3 D16", 34: "D17", 33: "S1 D16",
    32: "D16", 31: "S15 D8", 30: "D15", 29: "S13 D8", 28: "D14", 27: "S19 D4", 26: "D13",
    25: "S9 D8", 24: "D12", 23: "S7 D8", 22: "D11", 21: "S5 D8", 20: "D10", 19: "S3 D8",
    18: "D9", 17: "S1 D8", 16: "D8", 15: "S7 D4", 14: "D7", 13: "S5 D4", 12: "D6", 11: "S3 D4",
    10: "D5", 9: "S1 D4", 8: "D4", 7: "S3 D2", 6: "D3", 5: "S1 D2", 4: "D2", 3: "S1 D1", 2: "D1",
}


def checkout_suggestion(remaining: int, darts_left: int, double_out: bool = True) -> Optional[str]:
    """Empfohlener Weg zum Finish mit den verbleibenden Pfeilen, oder None."""
    if remaining <= 0 or darts_left <= 0:
        return None
    if not double_out:
        if remaining <= 60 and darts_left >= 1:
            for f in _PREFERRED:
                if _POINTS[f] == remaining:
                    return f
        return None
    if remaining < 2 or remaining > 170:
        return None
    route = _STANDARD_CHECKOUTS.get(remaining)
    if route and len(route.split()) <= darts_left:
        return route
    # rechnerisch: 1 Pfeil (Double), 2 Pfeile, 3 Pfeile
    for d in _DOUBLES:
        if _POINTS[d] == remaining and darts_left >= 1:
            return d
    if darts_left >= 2:
        for a in _PREFERRED:
            for d in _DOUBLES:
                if _POINTS[a] + _POINTS[d] == remaining:
                    return f"{a} {d}"
    if darts_left >= 3:
        for a in _PREFERRED:
            for b in _PREFERRED:
                for d in _DOUBLES:
                    if _POINTS[a] + _POINTS[b] + _POINTS[d] == remaining:
                        return f"{a} {b} {d}"
    return None


# ---------------------------------------------------------------------------
# Spielzustand
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    player: int
    leg: int
    score_before: int
    darts: List[Dart] = field(default_factory=list)
    bust: bool = False
    finished: bool = False   # Leg mit dieser Aufnahme beendet

    @property
    def points(self) -> int:
        return 0 if self.bust else sum(d.points for d in self.darts)

    @property
    def complete(self) -> bool:
        return self.bust or self.finished or len(self.darts) >= 3

    def to_dict(self) -> dict:
        return {
            "player": self.player,
            "leg": self.leg,
            "score_before": self.score_before,
            "darts": [d.to_dict() for d in self.darts],
            "points": self.points,
            "bust": self.bust,
            "finished": self.finished,
            "complete": self.complete,
        }


@dataclass
class PlayerStats:
    darts: int = 0
    points: int = 0
    first9_darts: int = 0
    first9_points: int = 0
    highest_turn: int = 0
    count_180: int = 0
    highest_finish: int = 0
    best_leg_darts: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "darts": self.darts,
            "points": self.points,
            "average": round(self.points / self.darts * 3, 2) if self.darts else 0.0,
            "first9_average": round(self.first9_points / self.first9_darts * 3, 2) if self.first9_darts else 0.0,
            "highest_turn": self.highest_turn,
            "count_180": self.count_180,
            "highest_finish": self.highest_finish,
            "best_leg_darts": self.best_leg_darts,
        }


@dataclass
class Player:
    name: str
    score: int
    legs_won: int = 0
    stats: PlayerStats = field(default_factory=PlayerStats)
    leg_darts: int = 0   # Pfeile im laufenden Leg


class X01Game:
    def __init__(
        self,
        players: List[str],
        start_score: int = 501,
        double_out: bool = True,
        legs_to_win: int = 1,
    ) -> None:
        if not 1 <= len(players) <= 4:
            raise ValueError("1 bis 4 Spieler")
        if start_score not in (301, 501, 701, 1001):
            raise ValueError("Startwert muss 301, 501, 701 oder 1001 sein")
        self.start_score = start_score
        self.double_out = double_out
        self.legs_to_win = max(1, legs_to_win)
        self.players = [Player(name=n.strip() or f"Spieler {i + 1}", score=start_score) for i, n in enumerate(players)]
        self.leg = 1
        self.leg_starter = 0
        self.current = 0
        self.turn: Turn = Turn(player=0, leg=1, score_before=start_score)
        self.history: List[Turn] = []
        self.winner: Optional[int] = None
        self.leg_winner: Optional[int] = None   # Sieger des zuletzt beendeten Legs (Anzeige)
        self.created = time.time()
        self.last_event = ""

    # ---- Abfragen -------------------------------------------------------
    @property
    def player(self) -> Player:
        return self.players[self.current]

    @property
    def over(self) -> bool:
        return self.winner is not None

    def remaining_for_current(self) -> int:
        """Rest des aktuellen Spielers unter Beruecksichtigung der offenen Aufnahme."""
        if self.turn.bust:
            return self.turn.score_before
        return self.turn.score_before - sum(d.points for d in self.turn.darts)

    # ---- Aktionen -------------------------------------------------------
    def add_dart(self, dart: Dart) -> bool:
        """Pfeil zur laufenden Aufnahme hinzufuegen. False, wenn keine Aufnahme offen ist."""
        if self.over or self.turn.complete:
            return False
        self.turn.darts.append(dart)
        self._evaluate_turn()
        return True

    def add_miss(self) -> bool:
        return self.add_dart(Dart("MISS", 0, 0, 0, manual=True))

    def correct_dart(self, index: int, field_text: str) -> bool:
        """Feld des index-ten Pfeils der laufenden Aufnahme ersetzen (auch nach Bust/Finish)."""
        if self.over and not self.turn.finished:
            return False
        if not 0 <= index < len(self.turn.darts):
            return False
        old = self.turn.darts[index]
        new = parse_field(field_text, manual=True, uncertain=False)
        new.x_mm, new.y_mm = old.x_mm, old.y_mm
        self.turn.darts[index] = new
        # Neu bewerten: Bust/Finish koennen sich aendern
        was_finished = self.turn.finished
        self.turn.bust = False
        self.turn.finished = False
        if was_finished:
            self._revert_leg_finish()
        self._evaluate_turn()
        self.last_event = f"Korrektur: {old.field} -> {self.turn.darts[index].field}"
        return True

    def remove_last_dart(self) -> bool:
        if not self.turn.darts:
            return False
        was_finished = self.turn.finished
        self.turn.darts.pop()
        self.turn.bust = False
        self.turn.finished = False
        if was_finished:
            self._revert_leg_finish()
        self._evaluate_turn()
        return True

    def end_turn(self) -> bool:
        """Aufnahme abschliessen (Pfeile gezogen / 'Weiter'). Leer -> nichts passiert."""
        if self.over and not self.turn.finished:
            return False
        if not self.turn.darts and not self.turn.bust:
            return False
        turn = self.turn
        self.history.append(turn)
        self._apply_stats(turn)
        if turn.finished:
            self._finish_leg(turn.player)
        else:
            self.player.score = turn.score_before - turn.points
            self.player.leg_darts += len(turn.darts)
            self.current = (self.current + 1) % len(self.players)
            self.turn = Turn(player=self.current, leg=self.leg, score_before=self.player.score)
        return True

    def undo(self) -> bool:
        """Letzten Pfeil zuruecknehmen; ist die Aufnahme leer, letzte Aufnahme wieder oeffnen."""
        if self.turn.darts:
            return self.remove_last_dart()
        if not self.history:
            return False
        turn = self.history.pop()
        self._unapply_stats(turn)
        if turn.finished:
            self._revert_leg_finish_committed(turn)
        else:
            self.players[turn.player].score = turn.score_before
            self.players[turn.player].leg_darts -= len(turn.darts)
        self.current = turn.player
        turn.finished = False
        turn.bust = False
        self.turn = turn
        self._evaluate_turn()
        self.last_event = "Aufnahme wieder geoeffnet"
        return True

    # ---- Intern ---------------------------------------------------------
    def _evaluate_turn(self) -> None:
        t = self.turn
        rem = t.score_before - sum(d.points for d in t.darts)
        t.bust = False
        t.finished = False
        if rem < 0:
            t.bust = True
        elif rem == 0:
            if self.double_out:
                if t.darts and t.darts[-1].is_double:
                    t.finished = True
                else:
                    t.bust = True
            else:
                t.finished = True
        elif rem == 1 and self.double_out:
            t.bust = True
        if t.finished and not self.over:
            self.last_event = "Game shot!"
        elif t.bust:
            self.last_event = "Bust"

    def _apply_stats(self, turn: Turn) -> None:
        s = self.players[turn.player].stats
        n = len(turn.darts)
        s.darts += n
        s.points += turn.points
        leg_darts_before = self.players[turn.player].leg_darts
        if leg_darts_before < 9:
            take = min(n, 9 - leg_darts_before)
            s.first9_darts += take
            # Punkte anteilig (Bust = 0)
            if not turn.bust:
                s.first9_points += sum(d.points for d in turn.darts[:take])
        s.highest_turn = max(s.highest_turn, turn.points)
        if turn.points == 180:
            s.count_180 += 1
        if turn.finished:
            s.highest_finish = max(s.highest_finish, turn.points)

    def _unapply_stats(self, turn: Turn) -> None:
        s = self.players[turn.player].stats
        n = len(turn.darts)
        s.darts -= n
        s.points -= turn.points
        leg_darts_after = self.players[turn.player].leg_darts
        leg_darts_before = leg_darts_after - n if not turn.finished else leg_darts_after
        if leg_darts_before < 9:
            take = min(n, 9 - leg_darts_before)
            s.first9_darts -= take
            if not turn.bust:
                s.first9_points -= sum(d.points for d in turn.darts[:take])
        if turn.points == 180:
            s.count_180 -= 1
        # highest_turn / highest_finish werden aus der Historie neu berechnet
        s.highest_turn = max((t.points for t in self.history if t.player == turn.player), default=0)
        s.highest_finish = max((t.points for t in self.history if t.player == turn.player and t.finished), default=0)

    def _finish_leg(self, winner: int) -> None:
        p = self.players[winner]
        p.score = 0
        p.leg_darts += len(self.turn.darts)
        p.legs_won += 1
        p.stats.best_leg_darts = min(p.stats.best_leg_darts or 10**9, p.leg_darts)
        self.leg_winner = winner
        if p.legs_won >= self.legs_to_win:
            self.winner = winner
            self.last_event = f"{p.name} gewinnt das Match!"
            self.turn = Turn(player=winner, leg=self.leg, score_before=0)
            return
        self.leg += 1
        self.leg_starter = (self.leg_starter + 1) % len(self.players)
        for pl in self.players:
            pl.score = self.start_score
            pl.leg_darts = 0
        self.current = self.leg_starter
        self.turn = Turn(player=self.current, leg=self.leg, score_before=self.start_score)
        self.last_event = f"{p.name} gewinnt Leg {self.leg - 1}"

    def _revert_leg_finish(self) -> None:
        """Offene Aufnahme war als Finish markiert, wird aber korrigiert (Leg noch nicht abgeschlossen)."""
        # Nichts persistiert, solange end_turn() nicht aufgerufen wurde.
        return

    def _revert_leg_finish_committed(self, turn: Turn) -> None:
        """Ein abgeschlossenes Leg per Undo zuruecknehmen."""
        winner = turn.player
        self.winner = None
        self.leg_winner = None
        p = self.players[winner]
        p.legs_won -= 1
        # Leg-Zaehler und Scores wiederherstellen
        self.leg = turn.leg
        self.leg_starter = (self.leg_starter - 1) % len(self.players) if self.leg != turn.leg or True else self.leg_starter
        # Scores aller Spieler aus der Historie dieses Legs rekonstruieren
        for i, pl in enumerate(self.players):
            pl.score = self.start_score
            pl.leg_darts = 0
        for t in self.history:
            if t.leg == turn.leg:
                pl = self.players[t.player]
                pl.score = t.score_before - t.points
                pl.leg_darts += len(t.darts)
        p.stats.best_leg_darts = min((self._leg_darts_of(pl_idx, lg) for pl_idx, lg in self._won_legs(winner)), default=None)
        # Der Leg-Starter des zurueckgenommenen Legs bleibt der, der es begonnen hat
        self.leg_starter = self._starter_of_leg(turn.leg)

    def _won_legs(self, player: int) -> List[Tuple[int, int]]:
        return [(t.player, t.leg) for t in self.history if t.finished and t.player == player]

    def _leg_darts_of(self, player: int, leg: int) -> int:
        return sum(len(t.darts) for t in self.history if t.player == player and t.leg == leg)

    def _starter_of_leg(self, leg: int) -> int:
        return (leg - 1) % len(self.players)

    # ---- Serialisierung -------------------------------------------------
    def to_dict(self) -> dict:
        rem = self.remaining_for_current()
        darts_left = 0 if self.turn.complete else 3 - len(self.turn.darts)
        return {
            "mode": f"{self.start_score}",
            "start_score": self.start_score,
            "double_out": self.double_out,
            "legs_to_win": self.legs_to_win,
            "leg": self.leg,
            "current": self.current,
            "winner": self.winner,
            "leg_winner": self.leg_winner,
            "over": self.over,
            "last_event": self.last_event,
            "players": [
                {
                    "name": p.name,
                    "score": p.score,
                    "legs_won": p.legs_won,
                    "leg_darts": p.leg_darts,
                    "stats": p.stats.to_dict(),
                }
                for p in self.players
            ],
            "turn": self.turn.to_dict(),
            "remaining": rem,
            "darts_left": darts_left,
            "checkout": checkout_suggestion(rem, darts_left, self.double_out) if not self.over else None,
            "history": [t.to_dict() for t in self.history[-12:]],
        }
