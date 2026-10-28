"""Test stand-in for the miles interfaces used by witness_rl."""
from __future__ import annotations

import copy
from typing import Any

from miles.utils.types import Sample


def compute_request_payload(args, input_ids: list[int],
                            sampling_params: dict[str, Any]) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    """The body of an inference-server request for `input_ids`, and None; or (None, TRUNCATED) when no new
    token fits.

    The number of new tokens is the `max_new_tokens` sampling parameter (default `args.rollout_max_response_len`),
    lowered to the room left by `args.rollout_max_context_len` when that is set. `sampling_params` is not changed.
    """
    params = copy.deepcopy(sampling_params)
    budget = params.pop("max_new_tokens", args.rollout_max_response_len)
    if args.rollout_max_context_len:
        budget = min(budget, args.rollout_max_context_len - len(input_ids))
    if budget <= 0:
        return None, Sample.Status.TRUNCATED
    body = dict(input_ids=input_ids, sampling_params={**params, "max_new_tokens": budget}, return_logprob=True,
                return_routed_experts=args.use_rollout_routing_replay,
                return_indexer_topk=args.use_rollout_indexer_replay)
    return body, None


async def update_sample_from_response(args, sample: Sample, payload: dict[str, Any], output: dict[str, Any],
                                      update_loss_mask: bool = False) -> None:
    """Append one SGLang response to `sample`.

    A sample without tokens and response starts from the request's input ids. The sampled token ids and their
    log-probabilities come from `output["meta_info"]["output_token_logprobs"]` (entries `(logprob, token id,
    ...)`), the text from `output["text"]`; with `update_loss_mask` every sampled token gets mask 1. Status and
    weight version are then taken from the meta info (`Sample.update_from_meta_info`).
    """
    if sample.response == "" and not sample.tokens:
        sample.tokens = payload["input_ids"]
    meta_info = output["meta_info"]
    pairs = meta_info.get("output_token_logprobs") or []
    sample.tokens = sample.tokens + [p[1] for p in pairs]
    sample.response_length += len(pairs)
    sample.response += output["text"]
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs.extend(p[0] for p in pairs)
    if update_loss_mask:
        if sample.loss_mask is None:
            sample.loss_mask = []
        sample.loss_mask.extend(1 for _ in pairs)
    sample.update_from_meta_info(args, meta_info)
