from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from app.services.amt.basic_pitch import NoteEvent
from app.services.guitar.fretboard import STANDARD_TUNING, pitch_to_fret_options
from app.services.guitar.open_chords import matches_open_chord

MAX_FRET_SPAN = 5
MAX_FRET_SPAN_HIGH = 6
MIN_FRET_SPAN = 4
MAX_FRET = 24

_CANDIDATES_PER_NOTE = 6
_CANDIDATES_PER_CHORD = 14
_ONSET_GROUP_WINDOW_S = 0.02


@dataclass(frozen=True)
class HandPosition:
    base_fret: int
    span: int
    finger_assignments: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FretPosition:
    string: int
    fret: int
    finger: int | None = None


@dataclass(frozen=True)
class TabEvent:
    time_s: float
    positions: list[FretPosition]
    is_chord: bool
    suggested_hand_position: int | None = None


@dataclass(frozen=True)
class TabOptimizationResult:
    events: list[TabEvent]
    total_cost: float
    position_changes: int
    impossible_transitions: list[tuple[int, int]]


@dataclass(frozen=True)
class _Candidate:
    positions: list[FretPosition]
    base_fret: int
    span: int
    cost: float
    avg_string: float
    avg_fret: float
    hand_position: HandPosition


def _group_note_events(
    note_events: Iterable[NoteEvent],
    *,
    onset_window_s: float = _ONSET_GROUP_WINDOW_S,
) -> list[tuple[float, list[int]]]:
    events = sorted(note_events, key=lambda e: float(e.start_time_s))
    if not events:
        return []

    grouped: list[tuple[float, list[int]]] = []
    current_time = float(events[0].start_time_s)
    current_pitches: list[int] = [int(events[0].pitch_midi)]

    for ev in events[1:]:
        t = float(ev.start_time_s)
        if t - current_time <= float(onset_window_s):
            current_pitches.append(int(ev.pitch_midi))
        else:
            grouped.append((current_time, list(current_pitches)))
            current_time = t
            current_pitches = [int(ev.pitch_midi)]

    grouped.append((current_time, list(current_pitches)))
    return grouped


def _base_fret(positions: list[FretPosition]) -> int:
    fretted = [p.fret for p in positions if p.fret > 0]
    if not fretted:
        return 0
    return int(min(fretted))


def _span_for_positions(positions: list[FretPosition], base_fret: int) -> int:
    fretted = [p.fret for p in positions if p.fret > 0]
    if not fretted:
        return 0
    return int(max(fretted) - int(base_fret))


def _max_span_for_base(base_fret: int) -> int:
    return MAX_FRET_SPAN_HIGH if int(base_fret) >= 12 else MAX_FRET_SPAN


def _finger_assignments(
    positions: list[FretPosition],
    base_fret: int,
) -> dict[int, int]:
    assignments: dict[int, int] = {}
    for pos in positions:
        if pos.fret <= 0:
            continue
        finger = int(pos.fret) - int(base_fret) + 1
        finger = max(1, min(4, finger))
        assignments[int(pos.string)] = finger
    return assignments


def _register_penalty(
    pitches: list[int],
    positions: list[FretPosition],
    tuning: tuple[int, ...],
) -> float:
    if not pitches or not positions:
        return 0.0
    tuning_list = list(tuning)
    penalty = 0.0
    for pitch, pos in zip(pitches, positions):
        idx = 6 - int(pos.string)
        if idx < 0 or idx >= len(tuning_list):
            continue
        open_pitch = int(tuning_list[idx])
        fret = int(pitch) - open_pitch
        penalty += abs(float(fret) - float(pos.fret)) * 0.05
        ideal_string = 6 - int(idx)
        penalty += abs(int(pos.string) - int(ideal_string)) * 0.2
    return float(penalty)


def _string_order_penalty(pitches: list[int], positions: list[FretPosition]) -> float:
    if len(pitches) < 2:
        return 0.0
    idx_sorted = sorted(range(len(pitches)), key=lambda i: int(pitches[i]))
    strings = [positions[i].string for i in idx_sorted]
    penalty = 0.0
    for i in range(1, len(strings)):
        if int(strings[i]) > int(strings[i - 1]):
            penalty += 1.0
    return float(penalty) * 0.8


def transition_feasibility(
    pos_from: list[tuple[int, int]],
    pos_to: list[tuple[int, int]],
    time_gap_s: float,
    tempo_bpm: float,
) -> float:
    if not pos_from or not pos_to:
        return 0.0
    frets_from = [p[1] for p in pos_from if p[1] > 0]
    frets_to = [p[1] for p in pos_to if p[1] > 0]
    if not frets_from or not frets_to:
        return 0.0
    move = abs(float(np.mean(frets_to)) - float(np.mean(frets_from)))
    tempo = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 120.0
    beat_dur = 60.0 / tempo
    fast = float(time_gap_s) < min(0.2, 0.35 * beat_dur)
    if fast and move > 5.0:
        return (move - 5.0) * 4.0
    return 0.0


def _candidate_cost(
    pitches: list[int],
    positions: list[FretPosition],
    tuning: tuple[int, ...],
) -> tuple[float, int, int, HandPosition]:
    base = _base_fret(positions)
    span = _span_for_positions(positions, base)
    max_span = _max_span_for_base(base)
    if span > max_span:
        return 1e9, base, span, HandPosition(base_fret=base, span=span, finger_assignments={})

    span_penalty = 0.0
    if span > MIN_FRET_SPAN:
        span_penalty = float(span - MIN_FRET_SPAN) * 2.0

    open_bonus = -0.6 if any(p.fret == 0 for p in positions) and base <= 4 else 0.0
    register_penalty = _register_penalty(pitches, positions, tuning)
    order_penalty = _string_order_penalty(pitches, positions)
    base_penalty = float(base) * 0.08

    assignments = _finger_assignments(positions, base)
    hand_pos = HandPosition(base_fret=int(base), span=int(span), finger_assignments=assignments)

    cost = base_penalty + span_penalty + register_penalty + order_penalty + open_bonus
    return float(cost), int(base), int(span), hand_pos


def _positions_to_fret_positions(positions: list[tuple[int, int]]) -> list[FretPosition]:
    return [FretPosition(string=int(s), fret=int(f)) for s, f in positions]


def _note_candidates(
    pitch: int,
    tuning: tuple[int, ...],
) -> list[_Candidate]:
    options = pitch_to_fret_options(int(pitch), tuning, max_fret=MAX_FRET)
    if not options:
        return []

    ranked: list[tuple[float, tuple[int, int]]] = []
    for string_num, fret in options:
        cost = float(fret) * 0.05
        if fret == 0:
            cost -= 0.5
        ranked.append((cost, (string_num, fret)))

    ranked.sort(key=lambda x: x[0])
    ranked = ranked[:_CANDIDATES_PER_NOTE]

    candidates: list[_Candidate] = []
    for _cost, pos in ranked:
        positions = [FretPosition(string=pos[0], fret=pos[1])]
        cost, base, span, hand_pos = _candidate_cost([pitch], positions, tuning)
        candidates.append(
            _Candidate(
                positions=positions,
                base_fret=base,
                span=span,
                cost=cost,
                avg_string=float(pos[0]),
                avg_fret=float(pos[1]),
                hand_position=hand_pos,
            )
        )
    return candidates


def _chord_candidates(
    pitches: list[int],
    chord_label: str,
    tuning: tuple[int, ...],
) -> list[_Candidate]:
    if not pitches:
        return []

    matched, open_positions = matches_open_chord(pitches, chord_label, tuning=tuning)
    candidates: list[_Candidate] = []
    if matched:
        positions = _positions_to_fret_positions(open_positions)
        cost, base, span, hand_pos = _candidate_cost(pitches, positions, tuning)
        candidates.append(
            _Candidate(
                positions=positions,
                base_fret=base,
                span=span,
                cost=cost - 1.0,
                avg_string=float(np.mean([p.string for p in positions])),
                avg_fret=float(np.mean([p.fret for p in positions])),
                hand_position=hand_pos,
            )
        )
        return candidates

    per_pitch_options: list[list[tuple[int, int]]] = []
    for pitch in pitches:
        options = pitch_to_fret_options(int(pitch), tuning, max_fret=MAX_FRET)
        if not options:
            return []
        ranked: list[tuple[float, tuple[int, int]]] = []
        for string_num, fret in options:
            cost = float(fret) * 0.05
            if fret == 0:
                cost -= 0.3
            ranked.append((cost, (string_num, fret)))
        ranked.sort(key=lambda x: x[0])
        per_pitch_options.append([pos for _c, pos in ranked[:4]])

    best: list[_Candidate] = []

    def backtrack(
        idx: int,
        used_strings: set[int],
        current: list[tuple[int, int]],
    ) -> None:
        if idx >= len(pitches):
            positions = _positions_to_fret_positions(list(current))
            cost, base, span, hand_pos = _candidate_cost(pitches, positions, tuning)
            if cost >= 1e8:
                return
            candidate = _Candidate(
                positions=positions,
                base_fret=base,
                span=span,
                cost=cost,
                avg_string=float(np.mean([p.string for p in positions])),
                avg_fret=float(np.mean([p.fret for p in positions])),
                hand_position=hand_pos,
            )
            best.append(candidate)
            return

        for string_num, fret in per_pitch_options[idx]:
            if int(string_num) in used_strings:
                continue
            used_strings.add(int(string_num))
            current.append((int(string_num), int(fret)))
            backtrack(idx + 1, used_strings, current)
            current.pop()
            used_strings.remove(int(string_num))

    backtrack(0, set(), [])
    if not best:
        return []

    best.sort(key=lambda c: c.cost)
    return best[:_CANDIDATES_PER_CHORD]


def _build_candidates(
    pitches: list[int],
    *,
    chord_label: str,
    tuning: tuple[int, ...],
) -> list[_Candidate]:
    if not pitches:
        return []
    if len(pitches) == 1:
        return _note_candidates(int(pitches[0]), tuning)
    return _chord_candidates(list(pitches), chord_label, tuning)


def _normalize_events(
    events: Iterable[tuple[float, list[int], str | None]],
) -> list[tuple[float, list[int], str]]:
    out: list[tuple[float, list[int], str]] = []
    for entry in events:
        time_s, pitches, label = entry
        out.append((float(time_s), list(pitches), str(label or "")))
    out.sort(key=lambda e: e[0])
    return out


def optimize_tab_positions_for_events(
    events: Iterable[tuple[float, list[int], str | None]],
    *,
    tuning: tuple[int, ...] = STANDARD_TUNING,
    tempo_bpm: float = 120.0,
) -> TabOptimizationResult:
    normalized = _normalize_events(events)
    if not normalized:
        return TabOptimizationResult(events=[], total_cost=0.0, position_changes=0, impossible_transitions=[])

    candidates_per_event: list[list[_Candidate]] = []
    for _time_s, pitches, label in normalized:
        candidates = _build_candidates(pitches, chord_label=label, tuning=tuning)
        if not candidates:
            candidates = [_Candidate(positions=[], base_fret=0, span=0, cost=50.0, avg_string=0.0, avg_fret=0.0,
                                     hand_position=HandPosition(base_fret=0, span=0, finger_assignments={}))]
        candidates_per_event.append(candidates)

    dp_costs: list[list[float]] = []
    dp_prev: list[list[int]] = []
    dp_costs.append([c.cost for c in candidates_per_event[0]])
    dp_prev.append([-1 for _ in candidates_per_event[0]])

    for i in range(1, len(normalized)):
        cur_costs: list[float] = []
        cur_prev: list[int] = []
        time_gap = float(normalized[i][0]) - float(normalized[i - 1][0])
        for j, cand in enumerate(candidates_per_event[i]):
            best_cost = None
            best_idx = -1
            for k, prev in enumerate(candidates_per_event[i - 1]):
                move_cost = abs(float(cand.base_fret) - float(prev.base_fret)) * 0.6
                move_cost += abs(float(cand.avg_string) - float(prev.avg_string)) * 0.4
                feasibility = transition_feasibility(
                    [(p.string, p.fret) for p in prev.positions],
                    [(p.string, p.fret) for p in cand.positions],
                    time_gap_s=time_gap,
                    tempo_bpm=tempo_bpm,
                )
                total = dp_costs[i - 1][k] + cand.cost + move_cost + feasibility
                if best_cost is None or total < best_cost:
                    best_cost = total
                    best_idx = k
            cur_costs.append(float(best_cost if best_cost is not None else cand.cost))
            cur_prev.append(int(best_idx))
        dp_costs.append(cur_costs)
        dp_prev.append(cur_prev)

    last_costs = dp_costs[-1]
    end_idx = int(np.argmin(last_costs))
    best_path: list[int] = [end_idx]
    for i in range(len(normalized) - 1, 0, -1):
        end_idx = dp_prev[i][end_idx]
        if end_idx < 0:
            end_idx = 0
        best_path.append(end_idx)
    best_path.reverse()

    tab_events: list[TabEvent] = []
    impossible: list[tuple[int, int]] = []
    position_changes = 0
    total_cost = float(last_costs[best_path[-1]])

    for i, (time_s, pitches, _label) in enumerate(normalized):
        cand = candidates_per_event[i][best_path[i]]
        positions = [
            FretPosition(
                string=int(pos.string),
                fret=int(pos.fret),
                finger=cand.hand_position.finger_assignments.get(int(pos.string)),
            )
            for pos in cand.positions
        ]
        if i > 0:
            prev = candidates_per_event[i - 1][best_path[i - 1]]
            if cand.base_fret != prev.base_fret:
                position_changes += 1
            gap = float(time_s) - float(normalized[i - 1][0])
            penalty = transition_feasibility(
                [(p.string, p.fret) for p in prev.positions],
                [(p.string, p.fret) for p in cand.positions],
                time_gap_s=gap,
                tempo_bpm=tempo_bpm,
            )
            if penalty > 0.0:
                impossible.append((i - 1, i))

        tab_events.append(
            TabEvent(
                time_s=float(time_s),
                positions=positions,
                is_chord=len(pitches) > 1,
                suggested_hand_position=int(cand.base_fret) if cand.base_fret > 0 else None,
            )
        )

    return TabOptimizationResult(
        events=tab_events,
        total_cost=total_cost,
        position_changes=position_changes,
        impossible_transitions=impossible,
    )


def optimize_tab_positions(
    note_events: list[NoteEvent],
    tuning: tuple[int, ...] = STANDARD_TUNING,
) -> list[list[tuple[int, int]]]:
    grouped = _group_note_events(note_events)
    events = [(time_s, pitches, None) for time_s, pitches in grouped]
    result = optimize_tab_positions_for_events(events, tuning=tuning, tempo_bpm=120.0)
    return [[(p.string, p.fret) for p in ev.positions] for ev in result.events]
