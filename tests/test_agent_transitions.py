"""Observation lines, the action showcase, the observation window, the "Last change" text and the action tracker
(`witness_agent.transitions`, `witness_agent.history`, `witness_agent.actions`).

`agent/units/perception_transition_line`, `perception_showcase`, `perception_window` and
`perception_action_tracker` hold expected outputs for small inputs. Their nodes are `[col, row]` and their grids
`[cols, rows]`; the tests convert both to the harness's (row, col) order. The level-complete line is also checked
against the first user message of every level of the recorded episode `agent/episodes/rl_tw03_s1041_full`.
"""
from __future__ import annotations

import pytest

from golden import STORE
from test_agent_board import semantics_from_json
from witness_agent.actions import ActionTracker, move_vote
from witness_agent.history import ObservationWindow
from witness_agent.transitions import MovementTracker, breadcrumb, showcase, transition_line
from witness_agent.view import BoardView, Pos


def rc(node: list[int]) -> Pos:
    return (node[1], node[0])


def dims_of(grid: list[int]) -> tuple[int, int]:
    return (grid[1], grid[0])


def view(path: tuple[Pos, ...], start: Pos, dims: tuple[int, int] = (3, 3)) -> BoardView:
    """A board without markers: only the path, the selected start and the size matter here."""
    rows, cols = dims
    return BoardView(rows=rows, cols=cols, path=path, start=start, starts=(start,), end=(0, cols), dots=(),
                     dot_fields=(), triangles={}, erasers=frozenset(), pieces={}, stars={}, squares={},
                     framed=frozenset(), markers=(), broken_h=frozenset(), broken_v=frozenset())


def stored_view(state: dict, grid: list[int]) -> BoardView:
    return view(tuple(rc(p) for p in state["path"]), rc(state["start"]), dims_of(grid))


def ids(cases: list[dict]) -> list[str]:
    return [c["id"] for c in cases]


# ------------------------------------------------------------------------------------------ observation lines
LINE_CASES = STORE.unit_cases("perception_transition_line")


@pytest.mark.parametrize("case", LINE_CASES, ids=ids(LINE_CASES))
def test_observation_line_of_one_step(case):
    inp = case["input"]
    line = transition_line(inp["action"], stored_view(inp["pre"], inp["grid"]), stored_view(inp["post"], inp["grid"]),
                           state_changed=inp["frame_changed"], random_fallback=inp["random_fallback"])
    assert line == case["expected"]["line"]


# --------------------------------------------------------------------------------------------------- showcase
SHOWCASE_CASES = STORE.unit_cases("perception_showcase")
ALIASING = [c for c in SHOWCASE_CASES if c["id"].startswith("aliasing")]
RULES = [c for c in SHOWCASE_CASES if not c["id"].startswith("aliasing")]


def showcase_of(case: dict) -> str:
    inp = case["input"]
    snapshots = {key: stored_view(state, inp["grid"]) for key, state in inp["snapshots"].items()}
    return showcase([tuple(t) for t in inp["transitions"]], snapshots, dims_of(inp["grid"]))


@pytest.mark.parametrize("case", RULES, ids=ids(RULES))
def test_showcase_shows_the_latest_one_edge_move_of_each_action(case):
    assert showcase_of(case) == case["expected"]["text"]


@pytest.mark.parametrize("case", ALIASING, ids=ids(ALIASING))
def test_showcase_reads_the_view_stored_for_the_before_key(case):
    assert showcase_of(case) == case["expected"]["text"]


def test_showcase_requires_a_stored_view_for_every_key():
    with pytest.raises(KeyError):
        showcase([("k0", 1, "kx")], {"k0": view(((2, 0),), (2, 0))}, (3, 3))


def test_showcase_skips_a_start_switch_whose_new_cursor_is_one_edge_from_the_old_one():
    snapshots = {"k0": view(((0, 0),), (0, 0), (2, 2)), "k1": view(((1, 1), (0, 1)), (1, 1), (2, 2))}
    assert showcase([("k0", 4, "k1")], snapshots, (2, 2)) == ""


# ----------------------------------------------------------------------------------------- window, breadcrumb
WINDOW_CASES = STORE.unit_cases("perception_window")


@pytest.mark.parametrize("case", WINDOW_CASES, ids=ids(WINDOW_CASES))
def test_observation_window_shows_the_last_eight_lines(case):
    window = ObservationWindow()
    for step, action, line, level_complete in case["input"]["entries"]:
        if level_complete:
            assert (step, action) == (0, 5)
            window.add_breadcrumb(line)
        else:
            window.add(step, action, line)
    assert window.text() == case["expected"]["text"]


def test_level_complete_line_opens_the_window_of_every_new_level():
    ep = STORE.episode("rl_tw03_s1041_full")
    assert ep.levels_completed == ep.max_levels          # the last completion ends the episode: no later call
    for level in range(1, ep.levels_completed):
        first = next(c for c in ep.calls if c.level == level)
        window = ObservationWindow()
        window.add_breadcrumb(breadcrumb(level, sum(1 for s in ep.steps if s[0] == level - 1), ep.max_levels))
        assert window.text() in first.user


# ------------------------------------------------------------------------------------------------ last change
def test_last_change_is_empty_after_a_level_entry_and_after_a_clear():
    start, up, right = view(((2, 0),), (2, 0)), view(((2, 0), (1, 0)), (2, 0)), view(((2, 0), (1, 0), (1, 1)), (2, 0))
    movement = MovementTracker()
    movement.reset(start)
    assert movement.text == ""
    movement.update(up, 1)
    assert movement.text == "agent moved from (2,0) to (1,0)"
    movement.clear()
    assert movement.text == ""
    movement.update(right, 4)                   # compared with the state before the clear
    assert movement.text == "agent moved from (1,0) to (1,1)"
    movement.reset(start)
    assert movement.text == ""


def test_last_change_after_a_start_switch_names_the_new_start_and_the_cursor():
    movement = MovementTracker()
    movement.reset(view(((2, 2),), (2, 2)))
    movement.update(view(((1, 1), (2, 1)), (1, 1)), 2)
    assert movement.text == "start switched to (1,1); cursor at (2,1)"


# ------------------------------------------------------------------------------------------------- move votes
def test_move_vote_runs_from_the_old_cursor_to_the_new_one():
    assert move_vote(view(((2, 0),), (2, 0)), view(((2, 0), (1, 0)), (2, 0))) == ((2, 0), (1, 0))


def test_move_vote_is_none_when_the_cursor_did_not_move():
    state = view(((2, 0), (1, 0)), (2, 0))
    assert move_vote(state, state) is None


def test_move_vote_after_a_start_switch_runs_from_the_new_start():
    assert move_vote(view(((2, 2),), (2, 2)), view(((0, 0), (0, 1)), (0, 0))) == ((0, 0), (0, 1))


def test_move_vote_is_none_when_a_start_switch_leaves_only_the_new_start():
    assert move_vote(view(((0, 0),), (0, 0)), view(((2, 2),), (2, 2))) is None


# --------------------------------------------------------------------------------------------- action tracker
TRACKER_CASES = STORE.unit_cases("perception_action_tracker")


@pytest.mark.parametrize("case", TRACKER_CASES, ids=ids(TRACKER_CASES))
def test_action_tracker_semantics(case):
    tracker = ActionTracker()
    for op in case["input"]["ops"]:
        if op[0] == "new_level":
            tracker.new_level()
            continue
        _, action, before, after, move, grid = op
        tracker.record(action, before, after, None if move is None else (rc(move[0]), rc(move[1])), dims_of(grid))
    assert tracker.semantics() == semantics_from_json(case["expected"]["semantics"])
