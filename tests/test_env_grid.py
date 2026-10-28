"""Board geometry, pixel drawing and lattice topology (`witness_env.grid`).

Hand-derived values come from the geometry formulas; `env/grid.json.gz` holds the expected geometry, drawn
pixels and region splits for every board size that occurs in the level files.
"""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import numpy as np
import pytest

from golden import STORE
from witness_env.grid import (BACKGROUND, CELL, FACING, LINE, Grid, count_sides, edge, edge_between, paint,
                              path_edges, region_of, rotate_offsets, split_regions)

# (cols, rows) -> (cell_size, offset_x, offset_y, half) for every board size of the level files.
LEVEL_BOARD_SIZES = {
    (1, 2): (26, 18, 4, 12), (1, 3): (17, 22, 4, 8), (1, 4): (12, 25, 5, 5), (2, 2): (26, 4, 4, 12),
    (2, 3): (17, 13, 4, 8), (2, 5): (10, 20, 4, 4), (3, 2): (17, 4, 13, 8), (3, 3): (17, 4, 4, 8),
    (3, 4): (12, 12, 5, 5), (4, 3): (12, 5, 12, 5), (4, 4): (12, 5, 5, 5), (4, 5): (10, 9, 4, 4),
    (5, 2): (10, 4, 20, 4), (5, 3): (10, 4, 15, 4), (5, 4): (10, 4, 9, 4), (5, 5): (10, 4, 4, 4),
    (6, 5): (8, 4, 9, 3), (6, 6): (8, 4, 4, 3),
}


@pytest.fixture(scope="module")
def expected() -> dict:
    return STORE.grid()


def changed_pixels(frame: np.ndarray, blank: np.ndarray) -> dict[tuple[int, int], int]:
    """{(x, y): color} of every pixel of `frame` that differs from `blank`."""
    ys, xs = np.nonzero(frame != blank)
    return {(int(x), int(y)): int(frame[y, x]) for x, y in zip(xs, ys)}


def drawn(grid: Grid, draw) -> dict[tuple[int, int], int]:
    """The pixels that `draw(frame)` changes on a blank frame of `grid`."""
    frame = grid.new_frame()
    draw(frame)
    return changed_pixels(frame, grid.new_frame())


def test_board_geometry_of_every_board_size_in_the_level_files(expected):
    level_sizes = set()
    for p in (files("witness_env") / "levels").iterdir():
        level_sizes |= {(c["cols"], c["rows"]) for c in json.loads(p.read_text(encoding="utf-8"))}
    assert level_sizes == set(LEVEL_BOARD_SIZES)
    assert sorted((s["cols"], s["rows"]) for s in expected["sizes"]) == sorted(LEVEL_BOARD_SIZES)
    for s in expected["sizes"]:
        grid = Grid(s["cols"], s["rows"])
        assert (grid.cell_size, grid.offset_x, grid.offset_y, grid.half) == LEVEL_BOARD_SIZES[(grid.cols, grid.rows)]
        assert (grid.cell_size, grid.offset_x, grid.offset_y) == (s["cell_size"], s["offset_x"], s["offset_y"])
        assert [[c, r, *grid.node_xy((c, r))] for c, r, _, _ in s["nodes"]] == s["nodes"]
        assert [[c, r, *grid.cell_center((c, r))] for c, r, _, _ in s["cells"]] == s["cells"]
        assert len(s["nodes"]) == (grid.cols + 1) * (grid.rows + 1) and len(s["cells"]) == grid.cols * grid.rows


@pytest.mark.parametrize("cols, rows", [(0, 1), (1, 0), (10, 1), (1, 10), (2, True)])
def test_board_size_outside_one_to_nine_is_rejected(cols, rows):
    with pytest.raises(ValueError, match="is outside 1..9"):
        Grid(cols, rows)


def test_largest_board_size_is_accepted():
    grid = Grid(9, 9)
    assert (grid.cell_size, grid.half, grid.offset_x, grid.offset_y) == (5, 2, 4, 4)


def test_node_and_cell_pixel_positions():
    g33, g12 = Grid(3, 3), Grid(1, 2)
    assert g33.node_xy((0, 0)) == (4, 4)
    assert g33.node_xy((3, 3)) == (58, 58)
    assert g33.cell_center((1, 2)) == (31, 49)
    assert g12.node_xy((1, 2)) == (45, 58)
    assert g12.cell_center((0, 1)) == (32, 45)


def test_blank_frame_colour_counts_and_digest(expected):
    by_hand = {(1, 2): {0: 1352, 3: 2556, 5: 188}, (3, 3): {0: 2601, 3: 1071, 5: 424},
               (4, 4): {0: 2304, 3: 1287, 5: 505}, (6, 6): {0: 2304, 3: 1071, 5: 721},
               (2, 5): {0: 1000, 3: 2808, 5: 288}}
    for s in expected["sizes"]:
        grid = Grid(s["cols"], s["rows"])
        frame = grid.new_frame()
        assert frame.shape == (64, 64) and frame.dtype == np.int8
        cols, rows, size = grid.cols, grid.rows, grid.cell_size
        cells = cols * rows * size * size
        lines = (cols + 1) * (rows + 1) + size * (cols * (rows + 1) + rows * (cols + 1))
        values, counts = np.unique(frame, return_counts=True)
        assert dict(zip(values.tolist(), counts.tolist())) == {CELL: cells, BACKGROUND: 4096 - cells - lines,
                                                               LINE: lines}
        if (cols, rows) in by_hand:
            assert dict(zip(values.tolist(), counts.tolist())) == by_hand[(cols, rows)]
        assert hashlib.sha1(frame.tobytes()).hexdigest() == s["grid_sha1"]


DRAWING_METHODS = ("draw_dot", "draw_edge", "draw_gap", "draw_square", "draw_star", "draw_triangles",
                   "draw_polyomino", "draw_eraser")


def as_tuples(value):
    """A JSON value with every list turned into a tuple, recursively (nodes, cells, edges, shapes)."""
    return tuple(as_tuples(v) for v in value) if isinstance(value, list) else value


def test_every_drawing_call_changes_exactly_the_expected_pixels(expected):
    assert len(expected["draws"]) == 1379
    assert {case["call"] for case in expected["draws"]} == set(DRAWING_METHODS)
    mismatches = []
    for i, case in enumerate(expected["draws"]):
        grid = Grid(case["cols"], case["rows"])
        args = {name: as_tuples(value) for name, value in case["args"].items()}
        pixels = drawn(grid, lambda f: getattr(grid, case["call"])(f, **args))
        got = sorted([x, y, colour] for (x, y), colour in pixels.items())
        if got != case["pixels"]:
            mismatches.append(f"case {i}: {case['cols']}x{case['rows']} {case['call']} {case['args']}")
    assert mismatches == []


def test_glyph_pixel_sets():
    g33, g66 = Grid(3, 3), Grid(6, 6)
    assert set(drawn(g33, lambda f: g33.draw_gap(f, ((0, 0), (1, 0))))) == {(x, 4) for x in range(7, 20)}
    assert set(drawn(g33, lambda f: g33.draw_gap(f, ((0, 0), (0, 1))))) == {(4, y) for y in range(7, 20)}
    assert set(drawn(g33, lambda f: g33.draw_triangles(f, (1, 1), 1))) == {(30, 31), (31, 30), (31, 31), (32, 31)}
    assert set(drawn(g33, lambda f: g33.draw_triangles(f, (1, 1), 2))) == {
        (28, 31), (29, 30), (29, 31), (30, 31), (32, 31), (33, 30), (33, 31), (34, 31)}
    assert set(drawn(g33, lambda f: g33.draw_triangles(f, (1, 1), 3))) == {
        (27, 31), (28, 30), (28, 31), (29, 31), (30, 31), (31, 30), (31, 31), (32, 31), (33, 31), (34, 30), (34, 31),
        (35, 31)}
    assert set(drawn(g33, lambda f: g33.draw_polyomino(f, (0, 0), [(0, 0), (0, 1), (1, 1)]))) == {
        (11, 11), (11, 12), (11, 13), (11, 14), (12, 11), (12, 12), (12, 13), (12, 14), (13, 13), (13, 14), (14, 13),
        (14, 14)}
    assert set(drawn(g66, lambda f: g66.draw_triangles(f, (0, 0), 3))) == {
        (6, 9), (7, 8), (7, 9), (8, 9), (9, 8), (9, 9), (10, 9), (11, 8), (11, 9), (12, 9)}
    assert set(drawn(g66, lambda f: g66.draw_polyomino(f, (0, 0), [(0, 0), (1, 0), (2, 0), (3, 0)]))) == {
        (7, 9), (8, 9), (9, 9), (10, 9)}
    assert drawn(g33, lambda f: g33.draw_dot(f, (1, 1), 7)) == {
        (21, 22): 7, (22, 21): 7, (22, 22): 7, (23, 22): 7, (22, 23): 7}
    assert set(drawn(g33, lambda f: g33.draw_star(f, (1, 1), 10))) == {
        (29, 31), (30, 30), (30, 31), (30, 32), (31, 29), (31, 30), (31, 31), (31, 32), (31, 33), (32, 30), (32, 31),
        (32, 32), (33, 31)}
    assert set(drawn(g33, lambda f: g33.draw_eraser(f, (1, 1)))) == {
        (31, 31), (31, 32), (31, 33), (30, 30), (32, 30), (29, 29), (33, 29)}
    assert set(drawn(g33, lambda f: g33.draw_square(f, (1, 1), 6))) == {(x, y) for x in (30, 31, 32)
                                                                        for y in (30, 31, 32)}


def test_edge_drawing_is_symmetric_and_includes_both_node_pixels():
    grid = Grid(3, 3)
    forward = drawn(grid, lambda f: grid.draw_edge(f, (0, 0), (1, 0), 9))
    assert drawn(grid, lambda f: grid.draw_edge(f, (1, 0), (0, 0), 9)) == forward
    assert forward == {(x, 4): 9 for x in range(4, 23)}
    assert drawn(grid, lambda f: grid.draw_edge(f, (0, 0), (0, 1), 9)) == {(4, y): 9 for y in range(4, 23)}


def test_paint_skips_pixels_outside_the_frame_and_later_writes_win():
    frame = Grid(3, 3).new_frame()
    before = frame.copy()
    paint(frame, [(-1, 0), (0, -1), (64, 5), (5, 64), (3, 3), (3, 3)], 7)
    paint(frame, [(3, 3)], 8)
    assert changed_pixels(frame, before) == {(3, 3): 8}


def test_edge_normalisation_and_path_edges():
    assert edge((1, 0), (0, 0)) == ((0, 0), (1, 0))
    assert path_edges([(1, 0), (0, 0), (0, 1)]) == {((0, 0), (1, 0)), ((0, 0), (0, 1))}
    assert path_edges([(2, 2)]) == frozenset()


@pytest.mark.parametrize("path, count", [
    ([(0, 0), (1, 0)], 1),
    ([(0, 0), (1, 0), (1, 1)], 2),
    ([(0, 1), (0, 0), (1, 0), (1, 1)], 3),
    ([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], 4),
])
def test_count_sides_counts_the_sides_on_the_path(path, count):
    assert count_sides((0, 0), path_edges(path)) == count


LEFT_COLUMN_CUT = (frozenset({(0, 0), (0, 1), (0, 2)}), frozenset({(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)}))


@pytest.mark.parametrize("cols, rows, edges, regions", [
    (3, 3, path_edges([(1, 0), (1, 1), (1, 2), (1, 3)]), LEFT_COLUMN_CUT),
    (3, 3, path_edges([(0, 1), (1, 1), (1, 2), (2, 2), (3, 2)]),
     (frozenset({(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)}), frozenset({(0, 1), (0, 2), (1, 2), (2, 2)}))),
    (3, 3, path_edges([(0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0)]),
     (frozenset((c, r) for c in range(3) for r in range(3)),)),
    (3, 3, frozenset({((0, 1), (3, 1))}), (frozenset((c, r) for c in range(3) for r in range(3)),)),
    (2, 2, path_edges([(0, 2), (1, 2), (1, 1), (0, 1), (0, 0), (1, 0), (2, 0)]),
     (frozenset({(0, 0), (1, 0), (1, 1)}), frozenset({(0, 1)}))),
], ids=["column_cut", "staircase_cut", "border_path", "non_lattice_edge", "enclosed_cell"])
def test_region_split_hand_cases(cols, rows, edges, regions):
    got = split_regions(cols, rows, edges)
    assert got == regions
    assert type(got) is tuple and all(type(r) is frozenset for r in got)


def test_region_split_path_edges_and_edge_counts_on_sample_paths(expected):
    kinds = set()
    for case in expected["regions"]:
        cols, rows = case["cols"], case["rows"]
        path = [tuple(n) for n in case["path"]]
        edges = path_edges(path)
        assert [[list(a), list(b)] for a, b in sorted(edges)] == case["edges"], case["seed"]
        assert [sorted([c, r] for c, r in region) for region in split_regions(cols, rows, edges)] == case["regions"]
        counts = {(c, r): n for c, r, n in case["edge_counts"]}
        assert sorted(counts) == sorted((c, r) for c in range(cols) for r in range(rows))
        assert {cell: count_sides(cell, edges) for cell in counts} == counts
        kinds.add(case["kind"])
    assert len(expected["regions"]) == 18 * 24
    assert kinds == {"single", "border", "snake", "staircase", "random"}


def test_cell_and_orientation_helpers():
    assert edge_between((1, 2), (2, 2)) == edge_between((2, 2), (1, 2)) == ((2, 2), (2, 3))
    assert edge_between((1, 2), (1, 1)) == ((1, 2), (2, 2))
    assert region_of((0, 1), LEFT_COLUMN_CUT) == {(0, 0), (0, 1), (0, 2)}
    assert region_of((3, 0), LEFT_COLUMN_CUT) == frozenset()
    assert FACING == ((0, -1), (1, 0), (0, 1), (-1, 0))
    assert [rotate_offsets(((1, 0), (1, -1)), f) for f in range(4)] == [
        ((0, -1), (-1, -1)), ((1, 0), (1, -1)), ((0, 1), (1, 1)), ((-1, 0), (-1, 1))]
    table = ((0, 0), (1, 0), (2, 0), (-1, -1), (-1, 1), (2, -2), (3, 1))
    assert rotate_offsets(table, 1) == table
    for facing in (-1, 4):
        with pytest.raises(IndexError):
            rotate_offsets(table, facing)
