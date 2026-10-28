"""Shared pytest setup: the CPU stand-in for the miles package under `tests/stubs/` is what `import miles`
resolves to in the tests; every test starts with WITNESS_GAME_CATALOG (`no_game_catalog`) and the proxy variables
(`no_proxy`) unset, whatever the shell sets; the `write_catalog` fixture names a test-local game catalog module in
WITNESS_GAME_CATALOG."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "stubs"))

# A game catalog module (`witness_agent.catalog`) listing "zz01": tw01's levels under that id.
CATALOG_SOURCE = '''
from witness_env.base import load_level_configs
from witness_env.games.tw01 import Tw01

PRIVATE_GAME_IDS = ("zz01",)


class Zz01(Tw01):
    GAME_ID = "zz01"

    def _level_configs(self):
        return load_level_configs("tw01")


def make_game(game_id):
    if game_id not in PRIVATE_GAME_IDS:
        raise KeyError(f"unknown catalog game id {game_id!r}")
    return Zz01()
'''


@pytest.fixture(autouse=True)
def no_game_catalog(monkeypatch):
    """Unset WITNESS_GAME_CATALOG for the test (and the subprocesses it starts), so a catalog the shell names for a
    benchmark run does not reach the public tests; tests that need a catalog set it themselves."""
    monkeypatch.delenv("WITNESS_GAME_CATALOG", raising=False)


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    """Unset the proxy variables httpx reads, so a proxy the shell sets does not answer the requests that the tests
    expect to fail locally (a host name that is too long)."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def write_catalog(tmp_path, monkeypatch):
    """A function writing a catalog module (the source given, by default `CATALOG_SOURCE`, also its attribute
    `default`) under a fresh name, putting its directory on the import path and naming it in WITNESS_GAME_CATALOG;
    it returns the module name. The modules are dropped from `sys.modules` afterwards."""
    names = []

    def write(source: str = CATALOG_SOURCE) -> str:
        name = f"zz_catalog_{len(names)}"
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
        names.append(name)
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setenv("WITNESS_GAME_CATALOG", name)
        return name

    write.default = CATALOG_SOURCE
    yield write
    for name in names:
        sys.modules.pop(name, None)
