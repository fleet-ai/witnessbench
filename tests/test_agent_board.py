"""The board view and the board block (`witness_agent.view`, `witness_agent.board`), and the texts of every
reflection call of a scripted episode per training game.

`agent/perception/<game>` holds one scripted episode per training game that enters every level (call cap
`max_reflection_calls`): its steps and, per reflection call, the board block, the action-semantics text and the
observations text the harness must produce. The test replays the steps on the game and rebuilds the three texts
from the harness's perception objects (the action-semantics text through `prompts.kas_text`), called exactly as the
episode loop calls them.
`agent/units/perception_actions_block` and `perception_action_tracker` hold expected Actions lines for given action
semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import pytest
from arcengine import ActionInput, GameAction, GameState

from golden import STORE, PerceptionEpisode, Step, first_diff
from witness_agent.actions import ActionSemantics, ActionTracker, MoveSemantics, move_vote
from witness_agent.board import board_block, budget_line
from witness_agent.history import ObservationWindow, TransitionTable
from witness_agent.loop import MAX_STEPS_PER_LEVEL
from witness_agent.prompts import kas_text
from witness_agent.transitions import MovementTracker, breadcrumb, showcase, transition_line
from witness_agent.view import BoardView, PieceView, board_view, state_key
from witness_env import BoardTruth, CellMarkers, NodeMarkers, Polyomino, make_game

# The budget line of a board block at a level's start, with no call answered and no call cap.
NO_BUDGET_USED = budget_line(0, MAX_STEPS_PER_LEVEL, 0, None)
# Legend meanings in their fixed order; None stands for the game's cell markers (sorted by name).
LEGEND_ORDER = ("agent", "trail", "start", "end", "mandatory_dot", "triangle_count", "eraser", "polyomino", "star",
                "colored_square", None, "two_or_more_objects_in_one_cell (each listed under Cell content)",
                "broken_H_edge", "broken_V_edge")
NODE_GLYPHS = frozenset("@+SEO|-")      # the glyphs of nodes and broken edges, listed only when on the board


def semantics_from_json(stored: dict) -> ActionSemantics:
    """The `ActionSemantics` of its fixture form: `{"moves": [MoveSemantics fields, ...], "confirm_presses": n}`."""
    return ActionSemantics(moves=tuple(MoveSemantics(**m) for m in stored["moves"]),
                           confirm_presses=stored["confirm_presses"])


def act(game, action: int):
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


# ------------------------------------------------------------------------------------------------- board view
BASE = dict(cols=3, rows=2, path=((0, 2), (1, 2), (1, 1)), start=(0, 2), starts=((0, 2),), end=(3, 0))


def test_board_view_swaps_every_coordinate_to_row_col():
    view = board_view(BoardTruth(**BASE))
    assert (view.path, view.start, view.starts, view.end) == (((2, 0), (2, 1), (1, 1)), (2, 0), ((2, 0),), (0, 3))
    assert (view.cursor, view.dims, view.rows, view.cols) == ((1, 1), (2, 3), 2, 3)


def test_board_view_keys_broken_edges_by_their_left_or_top_node():
    view = board_view(BoardTruth(**BASE, breakpoints=frozenset({((0, 0), (1, 0)), ((2, 1), (2, 2))})))
    assert (view.broken_h, view.broken_v) == ({(0, 0)}, {(1, 2)})


def test_board_view_normalises_polyomino_offsets():
    view = board_view(BoardTruth(**BASE, polyominoes={(1, 0): Polyomino(shape=((2, 1), (3, 1), (2, 2)),
                                                                          rotatable=True)}))
    assert dict(view.pieces) == {(0, 1): PieceView(offsets=((0, 0), (0, 1), (1, 0)), rotatable=True)}


def test_board_view_lists_node_marker_nodes_after_the_dots_without_repeats():
    view = board_view(BoardTruth(**BASE, dots=((1, 1),), node_markers=(
        NodeMarkers("knots", ((1, 1), (2, 0))), NodeMarkers("beads", ((2, 0), (0, 1))))))
    assert view.dots == ((1, 1), (0, 2), (1, 0))
    assert view.dot_fields == (("knots", ((1, 1), (0, 2))), ("beads", ((0, 2), (1, 0))))


def test_board_view_sorts_cell_markers_by_name():
    view = board_view(BoardTruth(**BASE, cell_markers=(CellMarkers("vanes", "v", {(0, 0): 1}),
                                                       CellMarkers("posts", "p", {(1, 1): None}))))
    assert [m.name for m in view.markers] == ["posts", "vanes"]
    assert [dict(m.values) for m in view.markers] == [{(1, 1): None}, {(0, 0): 1}]


def test_board_view_mappings_are_read_only():
    view = board_view(BoardTruth(**BASE, triangles={(0, 0): 2}))
    assert isinstance(view.triangles, MappingProxyType) and dict(view.triangles) == {(0, 0): 2}
    with pytest.raises(TypeError):
        view.triangles[(1, 1)] = 1


# ---------------------------------------------------------------------------------------------- Actions lines
def actions_lines(sem: ActionSemantics) -> str:
    """The lines after "Actions:" of the board block of the first tw01 state (no last change is shown)."""
    block = board_block(board_view(make_game("tw01").truth()), level=0, level_total=10, semantics=sem, movement="",
                        budget=NO_BUDGET_USED)
    return block.split("\nActions:\n", 1)[1]


ACTIONS_CASES = [pytest.param(c["input"]["semantics"], c["expected"]["actions_block"], id=c["id"])
                 for c in STORE.unit_cases("perception_actions_block")]
TRACKER_CASES = [pytest.param(c["expected"]["semantics"], c["expected"]["actions_block"], id=c["id"])
                 for c in STORE.unit_cases("perception_action_tracker")]


@pytest.mark.parametrize(("stored", "expected"), ACTIONS_CASES + TRACKER_CASES)
def test_actions_lines_of_the_board_block(stored, expected):
    assert actions_lines(semantics_from_json(stored)) == expected


# ---------------------------------------------------------------------- the path over the start and end nodes
def block_of(view: BoardView) -> str:
    return board_block(view, level=0, level_total=10, semantics=ActionTracker().semantics(), movement="",
                       budget=NO_BUDGET_USED)


def node_glyphs(block: str) -> list[str]:
    """The node glyphs of the ASCII board of a board block, one string per node row."""
    lines = block.split("\n")
    rows = lines[lines.index("Board:") + 2:]
    return [row[3::2] for row in rows[:rows.index("")] if row[:2].strip()]


def line_starting(block: str, prefix: str) -> str:
    return next(line for line in block.split("\n") if line.startswith(prefix))


def test_the_cursor_on_the_end_node_hides_the_end_glyph():
    game = make_game("tw01")
    act(game, 0)
    entry = block_of(board_view(game.truth()))
    for action in STORE.solutions()["tw01"][0][:-1]:          # the first level's solution without its CONFIRM
        act(game, action)
    view = board_view(game.truth())
    block = block_of(view)
    assert view.cursor == view.end == (0, 2)
    assert node_glyphs(block)[0] == "++@"
    assert "E=end" in line_starting(entry, "Legend: ") and "E=end" not in line_starting(block, "Legend: ")
    assert line_starting(block, "End:") == line_starting(entry, "End:") == "End:   (0,2)"


def test_a_path_through_the_end_node_hides_the_end_glyph():
    truth = BoardTruth(cols=3, rows=2, path=((3, 1), (3, 0), (2, 0)), start=(3, 1), starts=((3, 1),), end=(3, 0))
    block = block_of(board_view(truth))
    assert node_glyphs(block) == ["  @+", "   +", "    "]
    assert line_starting(block, "Legend: ") == "Legend: @=agent  +=trail"
    assert line_starting(block, "End:") == "End:   (0,3)"


def test_an_alternate_start_shows_its_glyph_and_the_selected_start_is_marked():
    truth = BoardTruth(cols=3, rows=2, path=((3, 2), (3, 1)), start=(3, 2), starts=((2, 0), (3, 2)), end=(3, 0))
    block = block_of(board_view(truth))
    assert node_glyphs(block) == ["  SE", "   @", "   +"]
    assert line_starting(block, "Legend: ") == "Legend: @=agent  +=trail  S=start  E=end"
    assert line_starting(block, "Starts: ") == "Starts: (0,2), (2,3) [selected]"


# ---------------------------------------------------------------------------- scripted episodes, call by call
@dataclass
class _State:
    key: str                            # the state key of the view
    view: BoardView


class _Replay:
    """The perception objects of one episode, driven exactly as the episode loop drives them."""

    def __init__(self, ep: PerceptionEpisode) -> None:
        self.game = make_game(ep.game_id)
        self.fd = act(self.game, 0)
        self.level_total = min(self.fd.win_levels, ep.max_levels)
        self.level = self.level_step = 0
        self.call_cap = ep.max_reflection_calls
        self.tracker, self.movement = ActionTracker(), MovementTracker()
        self.executed: list[Step] = []
        self.enter_level(self.observe())

    def observe(self) -> _State:
        view = board_view(self.game.truth())
        return _State(state_key(view), view)

    def enter_level(self, cur: _State) -> None:
        self.cur = cur
        self.table, self.window, self.snapshots = TransitionTable(), ObservationWindow(), {cur.key: cur.view}
        self.movement.reset(cur.view)

    def run_step(self, record: Step) -> None:
        level, level_step, action, actor, noop_reset = record
        assert (level, level_step) == (self.level, self.level_step), record
        self.executed.append(record)
        pre = self.cur
        if not noop_reset:
            self.fd = act(self.game, action)
        post = pre if noop_reset else self.observe()
        self.level_step += 1
        if self.fd.levels_completed > self.level:          # a completed level: enter the next one
            self.level += 1
            if self.level < self.level_total:
                self.tracker.new_level()
                self.enter_level(post)
                self.window.add_breadcrumb(breadcrumb(self.level, self.level_step, self.level_total))
            self.level_step = 0
            return
        self.table.record(pre.key, action, post.key)
        self.window.add(self.level_step, action, transition_line(action, pre.view, post.view,
                                                                 state_changed=pre.key != post.key,
                                                                 random_fallback=actor == "random"))
        self.snapshots[post.key] = post.view
        if noop_reset:
            self.movement.clear()
        else:
            self.movement.update(post.view, action)
        self.tracker.record(action, pre.key, post.key, move_vote(pre.view, post.view), post.view.dims)
        self.cur = post

    def board(self, calls: int) -> str:
        """The board block of a call made after `calls` answered calls."""
        budget = budget_line(self.level_step, MAX_STEPS_PER_LEVEL, calls, self.call_cap)
        return board_block(self.cur.view, level=self.level, level_total=self.level_total,
                           semantics=self.tracker.semantics(), movement=self.movement.text, budget=budget)

    def kas(self) -> str:
        return kas_text(self.tracker.semantics())

    def observations(self) -> str:
        cases = showcase(self.table.transitions(), self.snapshots, self.cur.view.dims)
        window = self.window.text()
        return cases + "\n\n" + window if cases else window


def check_legend(block: str) -> None:
    """The legend lists every glyph of the ASCII board in the fixed order; a glyph it lists that is not on the board
    is the glyph of a cell object covered by '&'; none of X, T, #, !, % appears on the board."""
    lines = block.split("\n")
    legend = next(line for line in lines if line.startswith("Legend: "))[len("Legend: "):]
    board = lines[lines.index("Board:") + 2:]
    board = board[:board.index("")]
    lattice = {ch for row in board for ch in row[3:]} - {" "}
    assert not lattice & set("XT#!%")
    entries = [entry.split("=", 1) for entry in legend.split("  ")]
    listed = {g for glyphs, _ in entries for g in glyphs.split("/")}
    covered = listed - lattice
    assert lattice <= listed and not covered & NODE_GLYPHS and (not covered or "&" in lattice)
    ranks = [(LEGEND_ORDER.index(meaning if meaning in LEGEND_ORDER else None),
              "" if meaning in LEGEND_ORDER else meaning) for _, meaning in entries]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize("game_id", STORE.perception_games())
def test_board_action_semantics_and_observations_of_every_reflection_call_match_the_scripted_episode(game_id):
    ep = STORE.perception_episode(game_id)
    replay = _Replay(ep)
    steps = iter(ep.steps)
    pending = next(steps, None)
    for i, call in enumerate(ep.calls):
        while pending is not None and pending[:2] != (call.level, call.level_step):
            replay.run_step(pending)
            pending = next(steps, None)
        board = replay.board(i)
        for name, want, got in (("board", call.board_state, board),
                                ("action semantics", call.action_semantics, replay.kas()),
                                ("observations", call.observations, replay.observations())):
            assert got == want, f"call {i} {name}: " + first_diff(want, got)
        check_legend(board)
    while pending is not None:
        replay.run_step(pending)
        pending = next(steps, None)
    assert tuple(replay.executed) == ep.steps
    assert replay.level == ep.levels_completed
    stop = ("max_levels" if replay.level >= ep.max_levels else
            "game_won" if replay.fd.state == GameState.WIN else "policy_stop")
    assert stop == ep.stop_reason
