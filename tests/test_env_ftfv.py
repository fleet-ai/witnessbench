"""The training games ft01-ft06, ft10-ft16, ft20 and the validation games fv01-fv04 (`witness_env.games`): level
files, config checks, frames, reported markers, rules.

Golden fixtures used: `env/entry_frames.json` (entry frame of every level), `env/marker_layers.json` (the marker layer
alone, drawn for configs that are not levels of the game), `env/rule_cases.json` (whether the game's rule accepts a
path on small boards that are not levels of the game), `env/traces/<game>.json` (scripted action sequences with the
engine's answer to every action; the validation games' sequences never complete a level and never reach the end node
with an accepted path), `env/solutions.json` (a solving action sequence per training level) and
`env/path_counts.json` (per training level: its end-reaching paths and how many of them CONFIRM accepts). Frames are
compared through the first 16 hex digits of the sha1 of their int8 bytes. A few more rule cases on small boards are
derived by hand.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, GameAction

import golden
from golden import sha16
from witness_env import GAME_IDS, WitnessGame, make_game
from witness_env.base import Board, load_level_configs
from witness_env.grid import CURSOR, START

GAMES = tuple(g for g in GAME_IDS if g.startswith(("ft", "fv")))
TRAINING = tuple(g for g in GAMES if g in golden.TRAINING_GAMES)
LEVEL_COUNTS = golden.LEVEL_COUNTS
# ft12's squares and gargoyles carry a color tag in the level file; the game reports the tag's palette color, and
# its gargoyles are the only cell markers whose values are colors.
TAG_COLOURS = {0: 6, 1: 10}
COLOUR_VALUED = {("ft12", "gargoyles")}
# Per game, what `truth()` declares: how squares are reported (None: never; "raw": the level-file colors;
# "tag": the palette color of the tag), the node-marker names in order, and the cell markers in order as
# (name, board glyph).
DECLARATIONS = {
    "ft01": ("raw", (), (("censers", "c"),)),
    "ft02": ("raw", (), (("hearths", "h"),)),
    "ft03": ("raw", (), (("granaries", "g"),)),
    "ft04": ("raw", (), (("beacon", "b"),)),
    "ft05": ("raw", (), (("coffers", "c"),)),
    "ft06": (None, (), (("buoys", "b"),)),
    "ft10": (None, ("knots", "beads"), ()),
    "ft11": (None, (), (("vanes", "v"), ("rift_vanes", "r"), ("posts", "p"))),
    "ft12": ("tag", (), (("gargoyles", "g"),)),
    "ft13": (None, (), (("abaci", "a"), ("counters", "c"))),
    "ft14": ("raw", (), (("braziers", "b"),)),
    "ft15": (None, (), (("cairns", "c"),)),
    "ft16": (None, (), (("flints", "f"), ("straws", "t"))),
    "ft20": (None, (), (("banners", "b"),)),
    "fv01": (None, ("wickets_rank1", "wickets_rank2", "wickets_rank3"), ()),
    "fv02": (None, (), (("weirs", "w"),)),
    "fv03": (None, (), (("estates", "t"),)),
    "fv04": (None, (), (("wattles", "w"),)),
}
RESET, UP, DOWN, LEFT, RIGHT, CONFIRM, ACTION7 = 0, 1, 2, 3, 4, 5, 7


def act(game: WitnessGame, action: int) -> FrameData:
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def entry_frame(game_id: str, level: int) -> np.ndarray:
    """The frame of `level` (0-based) of a fresh game right after entering it."""
    game = make_game(game_id)
    game.set_level(level)
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)


def config_cells(value: Any) -> dict[tuple[int, int], int | None]:
    """A level-file marker value as {cell: value}: a "c,r" -> int object, one [c, r] cell, or a list of cells
    (plain markers, value None)."""
    if isinstance(value, dict):
        return {tuple(int(n) for n in key.split(",")): v for key, v in value.items()}
    if value and isinstance(value[0], int):
        return {(value[0], value[1]): None}
    return {(c, r): None for c, r in value}


# --------------------------------------------------------------------------------------- level files, configs
@pytest.mark.parametrize("game_id", GAMES)
def test_reset_reports_the_level_count_and_the_first_entry_frame(game_id):
    fd = act(make_game(game_id), RESET)
    assert fd.win_levels == LEVEL_COUNTS[game_id]
    assert [sha16(f) for f in fd.frame] == golden.STORE.entry_frames()[game_id][:1]


@pytest.mark.parametrize("game_id", GAMES)
def test_level_file_holds_the_level_count_and_exactly_the_marker_keys_of_its_game(game_id):
    configs = load_level_configs(game_id)
    assert len(configs) == LEVEL_COUNTS[game_id]
    required = golden.REQUIRED_MARKER_KEYS[game_id]
    optional = golden.MARKER_KEYS[game_id] - required
    for level, config in enumerate(configs):
        assert {"cols", "rows", "start", "end"} | required <= set(config), level
        assert set(config) <= {"cols", "rows", "start", "end", "breakpoints"} | required | optional, level


def _first_level_class(game_id: str, mutate) -> type:
    """A subclass of the game whose only level is its first level config after `mutate(config)`."""
    config = json.loads(json.dumps(load_level_configs(game_id)[0]))
    mutate(config)
    return type(f"_Mutated{game_id}", (type(make_game(game_id)),), {"_level_configs": lambda self: [config]})


def _set(key, value):
    return lambda config: config.__setitem__(key, value)


def _delete(key):
    return lambda config: config.pop(key)


def _domain_error(game_id: str, key: str, value: int, allowed: list[int]) -> tuple:
    return (f"{game_id}-{key}-{value}", game_id, _set(key, {"0,0": value}),
            f"{game_id} level 1: {key}: cell 0,0 has value {value}, expected one of {allowed}")


CONFIG_ERRORS = [   # (id, game, mutation of its first level config, error message)
    ("missing-key", "ft01", _delete("censers"), "ft01 level 1: missing key 'censers'"),
    ("missing-required-wicket", "fv01", _delete("wickets_rank2"), "fv01 level 1: missing key 'wickets_rank2'"),
    ("misspelled-key", "ft01", _set("breakpoint", []), "ft01 level 1: unknown config key 'breakpoint'"),
    ("unknown-wicket-rank", "fv01", _set("wickets_rank4", [[0, 0]]),
     "fv01 level 1: unknown config key 'wickets_rank4'"),
    *[(f"{g}-board-too-large-for-the-glyphs", g, lambda c: c.update(cols=6, rows=6),
       f"{g} level 1: board 6x6: cell size 8 is below 9") for g in GAMES],
    ("three-flints", "ft16", _set("flints", [[0, 0], [1, 0], [2, 0]]),
     "ft16 level 1: flints: expected exactly two cells, got [[0, 0], [1, 0], [2, 0]]"),
    ("one-flint", "ft16", _set("flints", [[0, 0]]), "ft16 level 1: flints: expected exactly two cells, got [[0, 0]]"),
    ("two-nodes-for-one-wicket", "fv01", _set("wickets_rank1", [[0, 0], [1, 1]]),
     "fv01 level 1: wickets_rank1: expected exactly one node, got [[0, 0], [1, 1]]"),
    ("no-node-for-one-wicket", "fv01", _set("wickets_rank1", []),
     "fv01 level 1: wickets_rank1: expected exactly one node, got []"),
    ("boolean-coordinate", "ft01", _set("censers", [[True, 0]]),
     "ft01 level 1: censers: expected [col, row] with two ints, got [True, 0]"),
    ("boolean-node", "ft10", _set("beads", [[0, False]]),
     "ft10 level 1: beads: expected [col, row] with two ints, got [0, False]"),
    ("boolean-value", "ft06", _set("buoys", {"0,0": True}),
     "ft06 level 1: buoys: cell 0,0 has value True, expected an int"),
    *[(f"{g}-text-square-colour", g, _set("squares", {"0,0": "6"}),
       f"{g} level 1: squares: cell 0,0 has value '6', expected an int")
      for g in ("ft01", "ft02", "ft03", "ft04", "ft05", "ft14")],
    *[_domain_error("ft06", "buoys", v, [1, 2, 3, 4]) for v in (0, 5)],
    *[_domain_error("ft11", "vanes", v, [0, 1, 2, 3]) for v in (-1, 4)],
    _domain_error("ft12", "squares", 2, [0, 1]),
    _domain_error("ft12", "gargoyles", 2, [0, 1]),
    _domain_error("ft13", "abaci", 5, [2, 3, 4]),
    _domain_error("ft13", "counters", 3, [1, 2]),
    *[_domain_error("ft15", "cairns", v, [1, 2, 3]) for v in (0, 4)],
    *[_domain_error("ft20", "banners", v, [0, 1, 2, 3, 4, 5]) for v in (-1, 6)],
    _domain_error("fv02", "weirs", 4, [0, 1, 2, 3]),
    *[_domain_error("fv03", "estates", v, [1, 2, 3, 4]) for v in (0, 5)],
]


@pytest.mark.parametrize("game_id, mutate, message", [c[1:] for c in CONFIG_ERRORS],
                         ids=[c[0] for c in CONFIG_ERRORS])
def test_malformed_level_config_fails_construction_with_its_message(game_id, mutate, message):
    game_class = _first_level_class(game_id, mutate)
    with pytest.raises(ValueError) as exc:
        game_class()
    assert str(exc.value) == message


@pytest.mark.parametrize("game_id", sorted({c[1] for c in CONFIG_ERRORS}))
def test_unmodified_first_level_config_constructs(game_id):
    assert _first_level_class(game_id, lambda config: None)().win_score == 1


# ----------------------------------------------------------------------------------------------------- frames
@pytest.mark.parametrize("game_id", GAMES)
def test_entry_frames_match_the_golden_fixture_of_entry_frames(game_id):
    expected = golden.STORE.entry_frames()[game_id]
    assert len(expected) == LEVEL_COUNTS[game_id]
    assert [sha16(entry_frame(game_id, level)) for level in range(len(expected))] == expected


@pytest.mark.parametrize("game_id", GAMES)
def test_failed_confirm_at_the_start_re_renders_the_entry_frame(game_id):
    for level, digest in enumerate(golden.STORE.entry_frames()[game_id]):
        game = make_game(game_id)
        game.set_level(level)
        fd = act(game, CONFIRM)
        assert [sha16(f) for f in fd.frame] == [digest], level
        assert (fd.levels_completed, game.level_index) == (0, level)


@pytest.mark.parametrize("game_id", GAMES)
def test_marker_layers_of_non_level_configs_match_the_golden_fixture_of_marker_layers(game_id):
    cls = type(make_game(game_id))
    cases = golden.STORE.marker_layers()[game_id]
    assert cases
    for i, case in enumerate(cases):
        board = Board.from_config(case.config)
        frame = board.grid.new_frame()
        cls._draw_markers(frame, board, cls._parse_markers(case.config, board), frozenset())
        assert sha16(frame) == case.sha16, i


def test_cursor_sprite_follows_the_path_tip():
    assert all(type(make_game(game_id)).CURSOR_SPRITE_FOLLOWS_TIP is True for game_id in GAMES)
    game = make_game("ft01")
    (entry,) = act(game, RESET).frame
    frame = np.asarray(act(game, RIGHT).frame[0])
    truth = game.truth()
    assert len(truth.path) == 2
    grid = Board.from_config(load_level_configs("ft01")[0]).grid
    (tip_x, tip_y), (start_x, start_y) = grid.node_xy(truth.path[-1]), grid.node_xy(truth.start)
    # The tip is CURSOR in the frame's background anyway; a cursor sprite left at the start would cover the start's
    # center pixel, so the START color there shows that the sprite moved.
    assert (frame[tip_y, tip_x], frame[start_y, start_x]) == (CURSOR, START)
    (rejected,) = act(game, CONFIRM).frame
    assert sha16(rejected) == sha16(entry)


# ------------------------------------------------------------------------------------------- reported markers
@pytest.mark.parametrize("game_id", GAMES)
def test_truth_of_every_level_declares_the_markers_of_its_level_config(game_id):
    squares_as, node_names, cell_declaration = DECLARATIONS[game_id]
    game = make_game(game_id)
    for level, config in enumerate(load_level_configs(game_id)):
        game.set_level(level)
        truth = game.truth()
        squares = config_cells(config["squares"]) if squares_as else {}
        if squares_as == "tag":
            squares = {cell: TAG_COLOURS[tag] for cell, tag in squares.items()}
        assert dict(truth.squares) == squares, level
        assert [(m.name, m.nodes) for m in truth.node_markers] == [
            (name, tuple((c, r) for c, r in config[name])) for name in node_names if config.get(name)], level
        assert [(m.name, m.glyph, m.values_are_colours) for m in truth.cell_markers] == [
            (name, glyph, (game_id, name) in COLOUR_VALUED) for name, glyph in cell_declaration], level
        for marker in truth.cell_markers:
            values = config_cells(config[marker.name])
            if (game_id, marker.name) in COLOUR_VALUED:
                values = {cell: TAG_COLOURS[tag] for cell, tag in values.items()}
            assert dict(marker.values) == values, (level, marker.name)
            assert all(v is None or type(v) is int for v in marker.values.values()), (level, marker.name)
        assert (truth.dots, dict(truth.stars), dict(truth.triangles), dict(truth.polyominoes), truth.erasers,
                truth.framed_cells) == ((), {}, {}, {}, frozenset(), frozenset()), level
        assert truth.starts == (tuple(config["start"]),), level


# ----------------------------------------------------------------------------------------- scripted sequences
@pytest.mark.parametrize("game_id", GAMES)
def test_scripted_actions_match_the_golden_fixture_of_env_traces(game_id):
    trace = golden.STORE.env_trace(game_id)
    assert trace.solution_free == (game_id not in TRAINING)
    game = make_game(game_id)
    cls = type(game)
    configs = load_level_configs(game_id)
    boards = [Board.from_config(config) for config in configs]
    markers = [cls._parse_markers(config, board) for config, board in zip(configs, boards)]
    for i, step in enumerate(trace.steps):
        before = game.truth()
        if trace.solution_free and step.action == CONFIRM:
            assert before.path[-1] != before.end, i                      # never a CONFIRM on the end node
        fd = act(game, step.action)
        truth = game.truth()
        got = (tuple(sha16(f) for f in fd.frame), fd.levels_completed, fd.state.name, game.level_index, truth.path,
               truth.start)
        assert got == (step.frames, step.levels_completed, step.state, step.level, step.path, step.start), i
        if trace.solution_free and truth.path[-1] == truth.end:
            level = game.level_index
            assert not cls._wins(boards[level], markers[level], truth.path), i


@pytest.mark.parametrize("game_id", TRAINING)
def test_stored_solutions_played_in_a_row_win_every_level_and_enter_the_next(game_id):
    solutions = golden.STORE.solutions()[game_id]
    entry = golden.STORE.entry_frames()[game_id]
    assert len(solutions) == LEVEL_COUNTS[game_id]
    game = make_game(game_id)
    act(game, RESET)
    for level, solution in enumerate(solutions):
        results = [act(game, a) for a in solution]
        final = level == len(solutions) - 1
        assert [len(fd.frame) for fd in results] == [1] * (len(solution) - 1) + [1 if final else 2], level
        assert [fd.levels_completed for fd in results] == [level] * (len(solution) - 1) + [level + 1], level
        assert results[-1].state.name == ("WIN" if final else "NOT_FINISHED")
        if not final:
            assert sha16(results[-1].frame[-1]) == entry[level + 1], level


@pytest.mark.parametrize("game_id", TRAINING)
def test_each_stored_solution_wins_its_level_on_its_last_action_only(game_id):
    for level, solution in enumerate(golden.STORE.solutions()[game_id]):
        game = make_game(game_id)
        game.set_level(level)
        results = [act(game, a) for a in solution]
        assert [fd.levels_completed for fd in results] == [0] * (len(solution) - 1) + [1], level
        final = level == LEVEL_COUNTS[game_id] - 1
        assert (len(results[-1].frame), results[-1].state.name) == ((1, "WIN") if final else (2, "NOT_FINISHED"))


def test_after_winning_the_last_level_only_reset_returns_a_frame():
    game = make_game("ft02")
    act(game, RESET)
    for solution in golden.STORE.solutions()["ft02"]:
        results = [act(game, a) for a in solution]
    assert (len(results[-1].frame), results[-1].state.name, results[-1].levels_completed) == (1, "WIN", 10)
    for action in (UP, DOWN, LEFT, RIGHT, CONFIRM, ACTION7):
        fd = act(game, action)
        assert (fd.frame, fd.state.name) == ([], "WIN"), action
    fd = act(game, RESET)
    assert (len(fd.frame), fd.state.name, game.level_index) == (1, "NOT_FINISHED", 0)


# ------------------------------------------------------------------------------------------------------ rules
@pytest.mark.parametrize("game_id", TRAINING)
def test_path_counts_match_the_golden_fixture_of_path_counts(game_id):
    rows = golden.STORE.path_counts()[game_id]
    assert [row["level"] for row in rows] == list(range(LEVEL_COUNTS[game_id]))
    game = make_game(game_id)
    configs = load_level_configs(game_id)
    for row in rows:
        board = Board.from_config(configs[row["level"]])
        markers = type(game)._parse_markers(configs[row["level"]], board)
        assert golden.count_paths(game, board, markers) == (row["paths"], row["accepted"]), row["level"]


@pytest.mark.parametrize("game_id", GAMES)
def test_rule_outcomes_on_small_boards_match_the_golden_fixture_of_rule_cases(game_id):
    cls = type(make_game(game_id))
    cases = golden.STORE.rule_cases()[game_id]
    assert cases
    for i, case in enumerate(cases):
        board = Board.from_config(case.config)
        assert cls._wins(board, cls._parse_markers(case.config, board), case.path) is case.accepted, i


RULE_CASES = [   # (id, game, cols, rows, path, config fragment, expected)
    # The path cuts the gargoyle's view to the east; the only square of the other color is in its own cell.
    ("ft12-gargoyle-does-not-look-at-its-own-cell", "ft12", 2, 2, ((1, 0), (1, 1), (1, 2)),
     {"gargoyles": {"0,0": 0}, "squares": {"0,0": 1}}, True),
    # Ranks visited in the order 1, 3, 2: rank 1 comes before rank 3, but rank 3 comes before rank 2.
    ("fv01-every-consecutive-pair-of-ranks-is-compared", "fv01", 2, 2, ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
     {"wickets_rank1": [[1, 0]], "wickets_rank2": [[2, 2]], "wickets_rank3": [[2, 1]]}, False),
]


@pytest.mark.parametrize("game_id, cols, rows, path, fragment, expected", [c[1:] for c in RULE_CASES],
                         ids=[c[0] for c in RULE_CASES])
def test_rule_outcome_on_a_small_board(game_id, cols, rows, path, fragment, expected):
    config = fragment | {"cols": cols, "rows": rows, "start": list(path[0]), "end": list(path[-1])}
    cls = type(make_game(game_id))
    board = Board.from_config(config)
    assert cls._wins(board, cls._parse_markers(config, board), path) is expected
