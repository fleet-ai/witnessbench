"""Test stand-in for the miles interfaces used by witness_rl."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from types import MappingProxyType


@dataclass
class Sample:
    """One training sequence: prompt and sampled tokens, per-token log-probabilities and loss mask, reward,
    status and free-form metadata."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    group_index: int | None = None
    index: int | None = None
    rollout_id: int | None = None          # shared by every sample of one rollout
    prompt: str = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None
    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)

    def update_from_meta_info(self, args, meta_info: dict) -> None:
        """Take over what the server reports about one response.

        Appends `meta_info["weight_version"]` when present and sets the status from the finish reason ("length"
        -> TRUNCATED, "abort" -> ABORTED, "stop" -> COMPLETED; any other value leaves it unchanged). Reads
        `args.sglang_speculative_algorithm` (so `args` must define it) and raises NotImplementedError when it is
        set, since speculative-decoding statistics are not modeled here.
        """
        if args.sglang_speculative_algorithm:
            raise NotImplementedError("test stand-in: speculative decoding is not modeled")
        if "weight_version" in meta_info:
            self.weight_versions += [meta_info["weight_version"]]
        finish = meta_info["finish_reason"]["type"]
        if finish in _STATUS_BY_FINISH:
            self.status = _STATUS_BY_FINISH[finish]


_STATUS_BY_FINISH = MappingProxyType({"length": Sample.Status.TRUNCATED, "abort": Sample.Status.ABORTED,
                                      "stop": Sample.Status.COMPLETED})
