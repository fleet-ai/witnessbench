"""Test stand-in for the miles interfaces used by witness_rl."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any

from miles.utils.types import Sample


@dataclass(frozen=True)
class GenerateFnInput:
    """What the per-sample rollout function receives; `state` carries `args` and `tokenizer`."""

    state: Any
    sample: Sample
    sampling_params: dict[str, Any]
    evaluation: bool

    @property
    def args(self) -> Namespace:
        return self.state.args


@dataclass(frozen=True)
class GenerateFnOutput:
    """What the per-sample rollout function returns: one sample or a list of samples."""

    samples: Sample | list[Sample]
