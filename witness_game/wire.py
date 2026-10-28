"""The JSON form of a `GameFrame` in the `witness.game.v1` protocol: plain JSON values only, with no pickle, no
dynamic imports, no engine objects and no class names.

A dataclass becomes an object keyed by field name, a mapping a list of [key, value] pairs, a tuple a list, and a
frozenset a sorted list. Decoding accepts only the declared types of `GameFrame`, keeps the order of every sequence
and gives back the same immutable types (read-only mappings, tuples, frozensets).
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import cache
from types import MappingProxyType, UnionType
from typing import get_args, get_origin, get_type_hints

from .client import GameFrame


@cache
def _hints(cls):
    return get_type_hints(cls)


def _encode(value):
    if is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return [[_encode(k), _encode(v)] for k, v in value.items()]
    if isinstance(value, (tuple, frozenset)):
        return [_encode(v) for v in (sorted(value) if isinstance(value, frozenset) else value)]
    if value is None or type(value) in (int, str, bool):
        return value
    raise ValueError('unsupported observation value')


def _decode(value, schema):
    origin, args = get_origin(schema), get_args(schema)
    if origin is UnionType:
        for option in args:
            try:
                return _decode(value, option)
            except ValueError:
                pass
        raise ValueError('invalid observation union')
    if is_dataclass(schema):
        hints = _hints(schema)
        if not isinstance(value, dict) or set(value) != set(hints):
            raise ValueError('invalid observation fields')
        return schema(**{k: _decode(value[k], t) for k, t in hints.items()})
    if origin is Mapping:
        if not isinstance(value, list):
            raise ValueError('invalid observation mapping')
        result = {}
        for pair in value:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('invalid observation mapping entry')
            key, item = _decode(pair[0], args[0]), _decode(pair[1], args[1])
            if key in result:
                raise ValueError('duplicate observation mapping key')
            result[key] = item
        return MappingProxyType(result)
    if origin in (tuple, frozenset):
        if not isinstance(value, list):
            raise ValueError('invalid observation sequence')
        if origin is tuple and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ValueError('invalid observation tuple length')
            return tuple(_decode(v, t) for v, t in zip(value, args))
        items = [_decode(v, args[0]) for v in value]
        if origin is frozenset and len(set(items)) != len(items):
            raise ValueError('duplicate observation set entry')
        return origin(items)
    if schema in (str, int, bool, type(None)) and type(value) is schema:
        return value
    raise ValueError('invalid observation scalar')


def encode_frame(frame: GameFrame) -> dict:
    """`frame` as plain JSON values (see the module docstring). Raises ValueError for a value that is not a
    dataclass, a mapping, a tuple, a frozenset, None, an int, a str or a bool."""
    return _encode(frame)


def decode_frame(value: dict) -> GameFrame:
    """The `GameFrame` that `encode_frame` encoded. Raises ValueError for a value that does not match the types of
    `GameFrame` exactly, a negative level count, a game without levels, an empty board or an empty path."""
    frame = _decode(value, GameFrame)
    if frame.levels_completed < 0 or frame.win_levels < 1 or frame.truth.rows < 1 or frame.truth.cols < 1:
        raise ValueError('invalid game dimensions or level count')
    if not frame.truth.path:
        raise ValueError('empty game path')
    return frame
