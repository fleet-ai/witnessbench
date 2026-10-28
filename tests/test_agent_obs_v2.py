"""The prompt contract of the harness (observation version 2; `test_agent_obs_v2_golden.py` pins every prompt byte for
byte).
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import replace

import numpy as np
import pytest
from arcengine import ActionInput, GameAction

import obs_support
from obs_support import frame_key, lines, plan_text, play_answers
from witness_agent import EpisodeLimits, run_episode
from witness_agent.actions import ActionTracker
from witness_agent.board import RESERVED_GLYPHS, board_block, budget_line
from witness_agent.loop import MAX_STEPS_PER_LEVEL
from witness_agent.transitions import revealed_changes
from witness_agent.view import board_view, state_key
from witness_env import GAME_IDS, BoardTruth, CellMarkers, LegendEntry, NodeMarkers, RevealNodes, make_game


# ------------------------------------------------------------------------------------------------- frame keys
# The frame key of `obs_support` tells states apart by their pixels; the tests below compare the state keys with it.
ZERO_FRAME_KEY = "bb7df04e1b0a"          # key of the all-zero 64x64 frame
TW01_ENTRY_KEY = "89dca6994183"          # key of the first level's entry frame of tw01


def test_frame_key_depends_only_on_the_pixel_values():
    zeros = [[0] * 64 for _ in range(64)]
    keys = {frame_key(zeros), frame_key(np.zeros((64, 64), np.int8)), frame_key(np.zeros((64, 64), np.int64))}
    assert keys == {ZERO_FRAME_KEY}


def test_frame_key_of_the_first_entry_frame():
    frame = make_game("tw01").perform_action(ActionInput(id=GameAction.RESET)).frame[-1]
    assert frame_key(frame) == TW01_ENTRY_KEY
    assert frame_key(np.asarray(frame).tolist()) == TW01_ENTRY_KEY


# ------------------------------------------------------------------------------------------------- state keys
SLOW = os.environ.get("WITNESS_SLOW_TESTS") == "1"


def _truth(**changes) -> BoardTruth:
    """A snapshot that uses every field: two starts, dots, a dot field, a cell marker, flagged cells, lamps, a revealed
    marker and two legend entries."""
    base = dict(cols=3, rows=2, path=((0, 2), (1, 2)), start=(0, 2), starts=((0, 2), (3, 0)), end=(3, 2),
                dots=((2, 1), (1, 0)), node_markers=(NodeMarkers("pins", ((2, 2), (1, 1))),),
                cell_markers=(CellMarkers("blue_dot", "b", {(1, 1): None, (0, 0): None}),
                              CellMarkers("grey_bar", "g", {(2, 0): 3})),
                flagged_cells=frozenset({(0, 0)}), reveal_nodes=(RevealNodes("lamps", {(2, 0): False, (0, 1): True}),),
                revealed_markers=(CellMarkers("red_dot", "r", {(1, 0): None}),),
                legend=(LegendEntry("blue_dot", 9, "small_square"), LegendEntry("blue_dot", 6, "hollow_ring")))
    return BoardTruth(**{**base, **changes})


def _key(truth: BoardTruth) -> str:
    return state_key(board_view(truth))


def test_state_key_ignores_flags_and_snapshot_order_and_changes_with_everything_else_shown():
    key = _key(_truth())
    assert re.fullmatch(r"[0-9a-f]{12}", key)
    reordered = _truth(starts=((3, 0), (0, 2)), dots=((1, 0), (2, 1)),
                       node_markers=(NodeMarkers("pins", ((1, 1), (2, 2))),),
                       cell_markers=(CellMarkers("grey_bar", "g", {(2, 0): 3}),
                                     CellMarkers("blue_dot", "b", {(0, 0): None, (1, 1): None})),
                       reveal_nodes=(RevealNodes("lamps", {(0, 1): True, (2, 0): False}),),
                       legend=(LegendEntry("blue_dot", 6, "hollow_ring"), LegendEntry("blue_dot", 9, "small_square")))
    assert _key(reordered) == key
    assert _key(_truth(flagged_cells=frozenset())) == _key(_truth(flagged_cells=frozenset({(1, 1), (2, 0)}))) == key
    changed = [
        _truth(path=((0, 2),)), _truth(path=((3, 0),), start=(3, 0)), _truth(dots=((2, 1),)),
        _truth(reveal_nodes=(RevealNodes("lamps", {(2, 0): True, (0, 1): True}),)),
        _truth(revealed_markers=()), _truth(revealed_markers=(CellMarkers("red_dot", "r", {(1, 1): None}),)),
        _truth(legend=(LegendEntry("blue_dot", 9, "small_square"),)),
        _truth(cell_markers=(CellMarkers("blue_dot", "b", {(1, 1): None}),)),
        _truth(cell_markers=(CellMarkers("blue_dot", "b", {(1, 1): None, (0, 0): None}),
                             CellMarkers("grey_bar", "g", {(2, 0): 4}))),
        _truth(squares={(0, 1): 6}), _truth(breakpoints=frozenset({((0, 0), (1, 0))})),
    ]
    keys = [_key(t) for t in changed]
    assert key not in keys and len(set(keys)) == len(keys)


def test_v2_state_keys_partition_states_like_frame_keys_on_every_public_level():
    """On every level of every public game the state keys tell states apart exactly as the frame keys do, except:
    tw07 levels 3 and 6 and tw11 level 4 draw one frame for two states (a path walked one way or the other, or a
    one-node path at either of two starts), which the frame keys take for one state; on the tw09 levels whose start is
    on the leftmost or rightmost node column (levels 1-6, 8 and 9) the entry frame and the frame redrawn at the start
    differ in one pixel with the same board, which the state keys take for one state. (WITNESS_SLOW_TESTS=1: four walks
    of 300 steps per level instead of one of 100.)"""
    episodes, steps = (4, 300) if SLOW else (1, 100)
    found = {}
    for game_id in GAME_IDS:
        cls = obs_support.game_class(game_id)
        for index in range(obs_support.level_count(cls)):
            seen = [s for ep in range(episodes) for s in obs_support.walk_level(cls, index, steps, ep)]
            kinds = obs_support.partition_conflicts(seen)
            if kinds:
                found[(game_id, index + 1)] = kinds
    one_frame, one_state = {"one frame, two states"}, {"one state, two frames"}
    assert found == {("tw07", 3): one_frame, ("tw07", 6): one_frame, ("tw11", 4): one_frame,
                     **{("tw09", level): one_state for level in (1, 2, 3, 4, 5, 6, 8, 9)}}


def test_v2_tells_apart_the_states_that_share_a_frame_after_a_start_switch():
    """On tw07's third level the one-node path at the start (0,0) has the frame of another state but is a state of
    its own, so the entry state keeps its path and no ACTION4 move from (0,0) is shown as an example (the step
    switched the start), and ACTION3 does not undo ACTION4."""
    tw07 = obs_support.SOLUTIONS["tw07"]
    answers = [plan_text(tw07[0]), plan_text(tw07[1]), plan_text([4, 3])]
    policy = obs_support.FuzzPolicy("unused", answers)
    policy.queue.append("")                                  # the fourth call ends nothing; the cap stops first
    result = run_episode(make_game("tw07"), seed=0, limits=EpisodeLimits(10, 4), policy=policy)
    assert [(s.level, s.action) for s in result.steps[-3:-1]] == [(2, 4), (2, 3)]
    lines = policy.prompts[3][1].split("\n")
    assert "  #1 ACTION4: start switched to (0,0) (was (2,2)) | @(0,0)→(0,1) | trail +1" in lines
    assert not any(line.startswith("  ACTION4: @") for line in lines)


# ---------------------------------------------------------------------------------------------- board rendering
def block(truth: BoardTruth, movement: str = "") -> str:
    """The board block of `truth` (level 1 of 1, nothing executed yet)."""
    return board_block(board_view(truth), level=0, level_total=1, semantics=ActionTracker().semantics(),
                       movement=movement, budget=budget_line(0, MAX_STEPS_PER_LEVEL, 0, None))


def board_lines(text: str) -> list[str]:
    lines = text.split("\n")
    rows = lines[lines.index("Board:") + 1:]
    return rows[:rows.index("")]


def legend_entries(text: str) -> list[str]:
    return next(line for line in text.split("\n") if line.startswith("Legend: "))[len("Legend: "):].split("  ")


BOX = dict(cols=3, rows=3, path=((0, 3),), start=(0, 3), starts=((0, 3),), end=(3, 0))


def test_v2_two_kinds_of_object_in_one_cell_show_an_ampersand_and_the_legend_lists_every_kind_in_view():
    truth = BoardTruth(**BOX, squares={(0, 0): 6, (2, 2): 10}, triangles={(1, 1): 2},
                       erasers=frozenset({(1, 1)}),
                       cell_markers=(CellMarkers("blue_dot", "b", {(0, 0): None}),
                                     CellMarkers("grey_bar", "g", {(2, 0): None})))
    text = block(truth)
    assert board_lines(text)[2:7:2] == ["    &   g ", "      &   ", "        Q "]
    assert legend_entries(text) == ["@=agent", "E=end", "2=triangle_count", "R=eraser", "Q=colored_square",
                                  "b=blue_dot", "g=grey_bar",
                                  "&=two_or_more_objects_in_one_cell (each listed under Cell content)"]
    content = text.split("\nCell content:\n")[1].split("\n\n")[0].split("\n")
    assert content == ["  (1,1): triangle count=2", "  (1,1): eraser", "  (0,0): square color=6",
                       "  (2,2): square color=10", "  (0,0): blue_dot", "  (0,2): grey_bar"]


def test_v2_colour_values_and_other_values_are_grouped_under_their_own_headings():
    truth = BoardTruth(**BOX, squares={(0, 0): 6}, stars={(1, 0): 9},
                       cell_markers=(CellMarkers("grey_bar", "g", {(2, 0): 3, (2, 1): 1, (1, 1): 3}),
                                     CellMarkers("blue_dot", "b", {(0, 2): 9, (1, 2): 9}, values_are_colours=True),
                                     CellMarkers("pin", "p", {(2, 2): None})))
    text = block(truth)
    assert text.split("\n\nColor groups")[1].split("\nPath so far")[0] == (
        " (cells sharing the same color):\n"
        "  Squares of color 6: (0,0)\n"
        "  Stars of color 9:   (0,1)\n"
        "  Blue_dot of color 9: (2,0), (2,1)\n"
        "\n"
        "Value groups (cells sharing the same value):\n"
        "  Grey_bar with value 1: (1,2)\n"
        "  Grey_bar with value 3: (0,2), (1,1)\n")
    only_values = BoardTruth(**BOX, cell_markers=(CellMarkers("grey_bar", "g", {(2, 0): 3}),))
    assert "Color groups" not in block(only_values) and "Value groups" in block(only_values)


@pytest.mark.parametrize("glyph", sorted(RESERVED_GLYPHS))
def test_v2_rejects_a_marker_glyph_reserved_by_the_board(glyph):
    truth = BoardTruth(**BOX, cell_markers=(CellMarkers("thing", glyph, {(1, 1): None}),))
    with pytest.raises(ValueError, match=f"^cell marker glyphs reserved by the board: {re.escape(glyph)}$"):
        block(truth)


def test_v2_rejects_a_glyph_of_two_markers_revealed_ones_included():
    shared = BoardTruth(**BOX, cell_markers=(CellMarkers("bells", "g", {(0, 0): None}),),
                        revealed_markers=(CellMarkers("violet_tick", "g", {(1, 0): None}),))
    with pytest.raises(ValueError, match="^cell marker glyphs of more than one marker: g$"):
        block(shared)
    with pytest.raises(ValueError, match="^cell marker glyphs of more than one marker: g$"):
        block(replace(shared, cell_markers=shared.cell_markers + (CellMarkers("gongs", "g", {}),),
                      revealed_markers=()))
    block(replace(shared, revealed_markers=(CellMarkers("violet_tick", "u", {(1, 0): None}),)))


def test_no_public_game_declares_a_reserved_glyph_or_one_glyph_twice():
    for game_id in GAME_IDS:
        cls = obs_support.game_class(game_id)
        for index in range(obs_support.level_count(cls)):
            game = obs_support.single_level(cls, index)
            game.perform_action(ActionInput(id=GameAction.RESET))
            truth = game.truth()
            glyphs = [m.glyph for m in truth.cell_markers + truth.revealed_markers]
            assert not set(glyphs) & RESERVED_GLYPHS and len(set(glyphs)) == len(glyphs), (game_id, index + 1)
            block(truth)


def _shuffled(truth: BoardTruth, rng: random.Random) -> BoardTruth:
    """`truth` with every list and mapping in another order (the same state)."""
    def order(items):
        items = list(items)
        rng.shuffle(items)
        return tuple(items)

    def mapping(m):
        return dict(order(m.items()))

    return replace(truth, starts=order(truth.starts), dots=order(truth.dots),
                   node_markers=order(NodeMarkers(f.name, order(f.nodes)) for f in truth.node_markers),
                   cell_markers=order(CellMarkers(m.name, m.glyph, mapping(m.values), m.values_are_colours)
                                      for m in truth.cell_markers),
                   squares=mapping(truth.squares), stars=mapping(truth.stars), triangles=mapping(truth.triangles),
                   polyominoes=mapping(truth.polyominoes),
                   reveal_nodes=order(RevealNodes(r.name, mapping(r.lit)) for r in truth.reveal_nodes),
                   revealed_markers=order(CellMarkers(m.name, m.glyph, mapping(m.values), m.values_are_colours)
                                          for m in truth.revealed_markers),
                   legend=order(truth.legend))


def test_v2_board_text_does_not_depend_on_the_order_of_any_snapshot_list():
    truth = _truth(node_markers=(NodeMarkers("pins", ((2, 2), (1, 1))), NodeMarkers("bells", ((3, 1), (0, 1)))),
                   squares={(2, 1): 6, (0, 1): 6, (1, 0): 10}, triangles={(2, 0): 1, (0, 0): 2},
                   reveal_nodes=(RevealNodes("lamps", {(2, 0): False, (0, 1): True}),
                                 RevealNodes("bulbs", {(1, 2): True, (0, 0): False})),
                   legend=(LegendEntry("blue_dot", 9, "small_square"), LegendEntry("blue_dot", 6, "hollow_ring"),
                           LegendEntry("grey_bar", 3, "arrow")))
    rng = random.Random(7)
    texts = {block(_shuffled(truth, rng)) for _ in range(20)}
    assert texts == {block(truth)}


# ------------------------------------------------------------------------------------- rings and rejected submits
ROUTE = obs_support.rejected_route(obs_support.RingGame)        # [1, 1, 4, 4]: tw01 level 1 to the end, a dot missed
SUBMIT_5 = ("  #5 ACTION5: submit REJECTED — submitted path (5 nodes): (2,0) → (1,0) → (0,0) → (0,1) → (0,2); "
            "cells ringed in red: (0,0), (0,1), (1,0); path reset to start (2,0)")
DETAIL_5 = ("    submit at action 5: submitted path (5 nodes): (2,0) → (1,0) → (0,0) → (0,1) → (0,2); cells ringed "
            "in red: (0,0), (0,1), (1,0)")
RING_LINE = "Cells ringed in red by the last CONFIRM (rejected; path reset): (0,0), (0,1), (1,0)"


def test_v2_ring_line_is_one_sorted_line_after_the_actions_and_no_board_glyph():
    flagged = _truth(flagged_cells=frozenset({(2, 1), (0, 0), (1, 0)}))
    text = block(flagged, movement="CONFIRM rejected — path reset to start (2,0)")
    tail = text.split("\nActions:\n")[1].split("\n")
    assert tail[-3:] == ["Cells ringed in red by the last CONFIRM (rejected; path reset): (0,0), (0,1), (1,2)", "",
                         "Last change: CONFIRM rejected — path reset to start (2,0)"]
    assert tail[-4] == "" and tail[-5].startswith("  ACTION5 ")
    assert board_lines(text) == board_lines(block(_truth(flagged_cells=frozenset())))
    assert "!" not in text and "shown as" not in text and "violation" not in text.lower()
    assert "ringed" not in block(_truth(flagged_cells=frozenset()))


def test_v2_blocked_move_after_a_flagging_confirm_is_a_noop():
    """The blocked move right after a flagging CONFIRM removes the rings from the frame but leaves the
    state as it was; the harness says so and counts the move as a no-op (the state key holds no flags)."""
    answers = [plan_text([*ROUTE, 5]), plan_text([3])]
    users, run = play_answers(obs_support.RingGame(), answers)
    assert lines(users[2], "  #6 ") == ["  #6 ACTION3: blocked — cursor stays at (2,0) | red rings cleared"]
    assert "  ACTION3: noop=100% [observed ×1 this level]" in users[2].split("\n")
    assert lines(users[2], "Last change") == ["Last change: agent stayed at (2,0) | red rings cleared"]
    assert run.table.outcome_counts(3) == (0, 1)


def test_v2_rings_reach_the_next_prompt_when_the_confirm_is_not_the_last_action():
    """The ring line shows the cells ringed by the last action only; the history line and the
    attempt's line keep them."""
    users, _ = play_answers(obs_support.RingGame(), [plan_text([*ROUTE, 5])])
    assert RING_LINE in users[1].split("\n")
    assert SUBMIT_5 in users[1].split("\n")
    users, _ = play_answers(obs_support.RingGame(), [plan_text([*ROUTE, 5, 3])])
    assert "ringed in red by" not in users[1]
    assert SUBMIT_5 in users[1].split("\n")
    assert lines(users[1], "- attempt", "    submit") == [
        "- attempt #1: [1,1,4,4,5,3] → submit rejected at action 5/6", DETAIL_5]


def test_v2_never_says_nothing_changed_when_anything_visible_changed():
    """A RESET or a CONFIRM at the start right after a flagging CONFIRM removes the rings, and the texts say so; they
    say "nothing changed" only when nothing visible changed."""
    for tail, history, last in (
            ([0], "  #6 RESET: RESET — path already at start (2,0) | red rings cleared",
             "Last change: RESET — path already at start (2,0) | red rings cleared"),
            ([5], "  #6 ACTION5: submit REJECTED — path is only the start node (2,0) | red rings cleared",
             "Last change: CONFIRM rejected — cursor at start (2,0) | red rings cleared")):
        answers = [plan_text([*ROUTE, 5]), plan_text(tail)]
        users, _ = play_answers(obs_support.RingGame(), answers)
        assert lines(users[2], "  #6 ", "Last change") == [last, history]
    users, _ = play_answers(obs_support.RingGame(), [plan_text([*ROUTE, 5, 0, 0, 5])])
    assert lines(users[1], "  #7 ", "  #8 ") == [
        "  #7 RESET: RESET — path already at start (2,0); nothing changed",
        "  #8 ACTION5: submit REJECTED — nothing changed (path is only the start node (2,0))"]


def test_v2_reset_from_a_one_node_path_at_another_start_says_the_path_was_cleared():
    """tw07 level 3: a blocked ACTION4 switches the start to (0,0), ACTION3 backtracks to the one-node path there and
    RESET returns to the selected start (2,2): the path was cleared."""
    tw07 = obs_support.SOLUTIONS["tw07"]
    answers = [plan_text(tw07[0]), plan_text(tw07[1]), plan_text([4, 3, 0])]
    users, _ = play_answers(make_game("tw07"), answers)
    assert lines(users[3], "  #3 ", "Last change") == ["Last change: RESET — path cleared, cursor at start (2,2)",
                                                      "  #3 RESET: RESET — path cleared, cursor at start (2,2)"]


def test_v2_ignores_action7_so_the_move_after_a_flagging_confirm_clears_the_rings():
    """The harness does not take ACTION7, which is not one of the game's actions: the 7 of the plan is ignored
    with a notice, and the blocked move after the flagging CONFIRM redraws the frame without the rings."""
    users, run = play_answers(obs_support.RingGame(), [plan_text([*ROUTE, 5, 7, 3])])
    assert lines(users[1], "  #5 ", "  #6 ", "  #7 ") == [
        SUBMIT_5, "  #6 ACTION3: blocked — cursor stays at (2,0) | red rings cleared"]
    assert "- ignored plan token '7' — not an available action id" in users[1].split("\n")
    assert run.table.outcome_counts(7) == (0, 0) and run.table.outcome_counts(3) == (0, 1)


def test_v2_identical_attempt_warning_survives_a_flagging_confirm():
    """Two identical attempts from the start state are flagged although the first left rings behind."""
    answers = [plan_text([*ROUTE, 5]), plan_text([*ROUTE, 5])]
    warning = "⚠ Your last two attempts were IDENTICAL action sequences from the same starting state."
    users, _ = play_answers(obs_support.RingGame(), answers)
    assert warning in users[2].split("\n")
    assert lines(users[2], "- attempt", "    submit") == [
        "- attempt #1: [1,1,4,4,5] → submit rejected at action 5/5", DETAIL_5,
        "- attempt #2: [1,1,4,4,5] → submit rejected at action 5/5", DETAIL_5]


def test_v2_every_rejected_submit_of_a_plan_has_its_own_attempt_line():
    """One plan, two rejected submits: each keeps its line under the plan's line, which still ends with
    the plan-cap note."""
    actions = [*ROUTE, 5, 4, 1, 5] + [3] * 10
    users, _ = play_answers(obs_support.RingGame(), [plan_text(actions)])
    assert lines(users[1], "- attempt", "    submit") == [
        "- attempt #1: [1,1,4,4,5,4,1,5,3,3,3,3,3,3,3,3] → 2 submits rejected (last at action 8/16) (2 trailing "
        "action(s) cut by the plan cap)", DETAIL_5,
        "    submit at action 8: submitted path (3 nodes): (2,0) → (2,1) → (1,1)"]


def test_v2_rejected_confirm_line_names_the_submitted_path():
    """On ft20 level 1, a 20-node path drawn over three plans and then submitted can be read back in full
    from the next prompt, although most of its steps left the observation window."""
    moves = [3] * 4 + [1] + [4] * 4 + [1] + [3] * 4 + [1] + [4] * 4
    answers = [plan_text(moves[:7]), plan_text(moves[7:14]), plan_text([*moves[14:], 5])]
    users, _ = play_answers(make_game("ft20"), answers)
    path = ("(4,4) → (4,3) → (4,2) → (4,1) → (4,0) → (3,0) → (3,1) → (3,2) → (3,3) → (3,4) → (2,4) → (2,3) → (2,2) "
            "→ (2,1) → (2,0) → (1,0) → (1,1) → (1,2) → (1,3) → (1,4)")
    assert lines(users[3], "  #20 ") == [
        f"  #20 ACTION5: submit REJECTED — submitted path (20 nodes): {path}; path reset to start (4,4)"]
    assert lines(users[3], "- ", "    submit")[:3] == [
        "- chunk before attempt #1 (no submit): [4,4,1,3,3,3,3] → ran to its end — continue from the resulting state",
        "- attempt #1: [1,4,4,4,4,5] → submit rejected at action 6/6",
        f"    submit at action 6: submitted path (20 nodes): {path}"]


# ------------------------------------------------------------------------------------------ the reveal channel
def test_v2_node_marks_have_their_own_line_looks_and_glyph_and_are_never_mandatory_dots():
    text = block(_truth())                       # lamps (0,2) not yet reached and (1,0) reached, in (row, col)
    node_rows = board_lines(text)[1::2]
    assert node_rows[0][3:] == "  O Z S" and node_rows[1][3:] == "Z O O  "
    assert "Z=lamps" in legend_entries(text)
    assert lines(text, "Node marks") == ["Node marks (lamps): (0,2) light_gray_plus, (1,0) white_block"]
    assert lines(text, "Mandatory dots") == [
        "Mandatory dots: (0,1) [not visited], (1,1) [not visited], (1,2) [not visited], (2,2) [not visited]"]
    assert not re.search(r"\b(un)?lit\b", text) and "#" not in board_lines(text)[1] and "*" not in text


def test_v2_revealed_markers_appear_as_cell_markers():
    text = block(_truth())
    assert "r=red_dot" in legend_entries(text)
    assert board_lines(text)[2][3:] == " b r g "                # cells (0,0), (0,1), (0,2)
    assert "  (0,1): red_dot" in text.split("\n")
    hidden = block(_truth(revealed_markers=()))
    assert "red_dot" not in hidden and board_lines(hidden)[2][3:] == " b   g "


LAMP_RESET = ("RESET — path already at start (2,0) | lamps (1,2), (2,1) now light_gray_plus | blue_dot no longer shown "
              "at (1,1) | red rings cleared")


def test_v2_reveal_state_follows_the_engine_through_confirm_reset_and_level_change():
    """A lamp reached shows white_block and reveals its cell marker; a rejected CONFIRM keeps both; RESET puts
    every lamp out and the marker (and its legend entry) disappears; the next level starts from its own lamps. Every
    change of look is told in the step's line and in the last change."""
    answers = [plan_text([4]), plan_text([1, 4, 1, 5]), plan_text([0]), plan_text(obs_support.SOLUTIONS["tw01"][0])]
    users, _ = play_answers(obs_support.LampGame(), answers)
    entry, lit, rejected, reset, level_2 = users[:5]
    assert lines(entry, "Node marks") == ["Node marks (lamps): (1,2) light_gray_plus, (2,1) light_gray_plus"]
    assert "blue_dot" not in entry
    assert lines(lit, "Node marks") == ["Node marks (lamps): (1,2) light_gray_plus, (2,1) white_block"]
    assert lines(lit, "  #1 ", "Last change") == [
        "Last change: agent moved from (2,0) to (2,1) | lamps (2,1) now white_block | blue_dot now shown at (1,1)",
        "  #1 ACTION4: @(2,0)→(2,1) | trail +1 | lamps (2,1) now white_block | blue_dot now shown at (1,1)"]
    assert "b=blue_dot" in legend_entries(lit) and "  (1,1): blue_dot" in lit.split("\n")
    assert lines(rejected, "Node marks") == ["Node marks (lamps): (1,2) white_block, (2,1) white_block"]
    assert "  (1,1): blue_dot" in rejected.split("\n")
    assert "Cells ringed in red by the last CONFIRM (rejected; path reset): (1,1)" in rejected.split("\n")
    assert lines(rejected, "  #3 ") == ["  #3 ACTION4: @(1,1)→(1,2) | trail +1 | lamps (1,2) now white_block"]
    assert lines(reset, "Node marks") == lines(entry, "Node marks")
    assert lines(reset, "  #6 ", "Last change") == [f"Last change: {LAMP_RESET}", f"  #6 RESET: {LAMP_RESET}"]
    assert "  (1,1): blue_dot" not in reset.split("\n") and legend_entries(reset) == legend_entries(entry)
    assert lines(level_2, "Node marks") == ["Node marks (lamps): (0,3) light_gray_plus, (3,0) light_gray_plus"]


def test_v2_a_marker_revealed_and_hidden_again_within_one_plan_reaches_the_text():
    """A plan that walks onto a lamp and then RESETs: the next board shows no revealed marker, but the step lines and
    the last change say where it was shown and that it is no longer shown."""
    users, _ = play_answers(obs_support.LampGame(), [plan_text([4, 0])])
    gone = "lamps (2,1) now light_gray_plus | blue_dot no longer shown at (1,1)"
    assert lines(users[1], "  #", "Last change") == [
        f"Last change: RESET — path cleared, cursor at start (2,0) | {gone}",
        "  #1 ACTION4: @(2,0)→(2,1) | trail +1 | lamps (2,1) now white_block | blue_dot now shown at (1,1)",
        f"  #2 RESET: RESET — path cleared, cursor at start (2,0) | {gone}"]
    assert "b=blue_dot" not in legend_entries(users[1])


def test_v2_revealed_changes_are_sorted_and_carry_values():
    hidden = board_view(_truth(revealed_markers=()))
    shown = board_view(_truth(revealed_markers=(
        CellMarkers("red_dot", "r", {(2, 1): None, (1, 0): None}),
        CellMarkers("grey_ring", "q", {(0, 1): 4, (0, 0): 2}),
        CellMarkers("blue_ring", "u", {(1, 1): 9}, values_are_colours=True))))
    assert revealed_changes(hidden, shown) == [
        "blue_ring now shown at (1,1) color=9", "grey_ring now shown at (0,0) value=2, (1,0) value=4",
        "red_dot now shown at (0,1), (1,2)"]
    assert revealed_changes(shown, hidden) == [
        "blue_ring no longer shown at (1,1) color=9", "grey_ring no longer shown at (0,0) value=2, (1,0) value=4",
        "red_dot no longer shown at (0,1), (1,2)"]
    moved = board_view(_truth(revealed_markers=(CellMarkers("red_dot", "r", {(2, 1): None, (0, 0): None}),)))
    assert revealed_changes(board_view(_truth()), moved) == ["red_dot now shown at (0,0), (1,2)",
                                                             "red_dot no longer shown at (0,1)"]
    assert revealed_changes(shown, shown) == []


# ------------------------------------------------------------------------------------------------ legend strip
def test_v2_legend_strip_entries_are_one_sorted_line():
    text = block(_truth())
    strip = ("Legend strip (drawn outside the board): blue_dot color=6 next to hollow_ring; blue_dot color=9 next to "
             "small_square")
    assert lines(text, "Legend strip") == [strip]
    after_groups = text.split("\nValue groups (cells sharing the same value):\n")[1].split("\n")
    assert after_groups[:3] == ["  Grey_bar with value 3: (0,2)", "", strip]
    assert "Legend strip" not in block(_truth(legend=()))
    users, _ = play_answers(obs_support.StripGame(), [plan_text([4])])
    assert all(lines(u, "Legend strip") == [strip] for u in users)
    assert "  Blue_dot of color 9: (0,0)" in users[0].split("\n")


# ------------------------------------------------------------------------------------ the recorded episodes
RECORDED = sorted(name for name in obs_support.specs_by_name()
                  if name.startswith("level_") or (name.startswith("multi_") and (SLOW or name.endswith("_s0"))))


@pytest.mark.parametrize("name", RECORDED + sorted(n for n in obs_support.specs_by_name() if n.startswith("made_")))
def test_v2_never_says_nothing_changed_of_a_step_that_changed_what_the_board_shows(name):
    result, _, snapshots, frames = obs_support.played(name)
    tw09 = obs_support.specs_by_name()[name].game_id == "tw09"
    assert obs_support.check_nothing_changed(result, snapshots, frames, frame_exempt=tw09) > 0


@pytest.mark.parametrize("game_id", [*GAME_IDS, *obs_support.MADE_GAMES])
def test_v2_equal_state_keys_mean_equal_board_text_apart_from_rings_actions_and_last_change(game_id):
    cls = obs_support.MADE_GAMES.get(game_id) or obs_support.game_class(game_id)
    for index in range(obs_support.level_count(cls)):
        for episode in range(4 if SLOW else 1):
            obs_support.check_keys_and_texts(obs_support.walk_level(cls, index, 300 if SLOW else 100, episode))


class _ShufflingTruth:
    """Makes `game.truth()` list and map everything in a new random order on every call (`_shuffled`)."""

    def __init__(self, game, seed: int):
        rng, original = random.Random(seed), game.truth
        game.truth = lambda: _shuffled(original(), rng)
        self.game = game


@pytest.mark.parametrize("game_id", ["tw07", "tw12", "tw13", "ft06", "ft11", "ft12", "ft20", "fv01", "fv03",
                                     *obs_support.MADE_GAMES])
def test_v2_prompts_do_not_depend_on_the_order_of_any_snapshot_list(game_id):
    def game():
        return obs_support.MADE_GAMES[game_id]() if game_id in obs_support.MADE_GAMES else make_game(game_id)

    prompts = []
    for g in (game(), _ShufflingTruth(game(), 1).game, _ShufflingTruth(game(), 2).game):
        factory = obs_support.MADE_GAMES.get(game_id) or (lambda: obs_support.single_level(
            obs_support.game_class(game_id), 0))
        policy = obs_support.FuzzPolicy(f"order:{game_id}", obs_support.prelude(factory, game_id),
                                        obs_support.SOLUTIONS.get(game_id))
        result = run_episode(g, seed=0, limits=EpisodeLimits(g.win_score, 30), policy=policy)
        prompts.append([(c.system, c.user) for c in result.calls])
    assert prompts[0] == prompts[1] == prompts[2]
