# Evaluation

Two evaluators play task files through the [canonical harness](harness.md) and report the **levels beaten** per
(game, seed) episode:

- `witness_agent.evaluate` for a model you serve yourself;
- `witness_agent.evaluate_api` for a model on OpenRouter.

Both write the same report, episode records and traces, and `witness_agent.evaluate_api` also keeps a journal of the
replies ([data formats](data-formats.md)). The leaderboard scores (RHAE and OAE, see [Metrics](#metrics)) are
computed by the maintainers from these per-level records and the human baselines, which are not part of this
repository.

An episode plays one game from its first level. Whenever no planned action is left, the harness makes a
*reflection call*: it sends the model a system prompt and a user message and reads the next plan from the reply
([harness](harness.md#how-an-episode-runs)).

Every seed plays the same levels: the seed only fixes the random move that the harness makes after a reply without a
valid plan. The seed is not sent to the model, so two runs with the same seed can differ.

The two evaluators use different settings, so their numbers are not directly comparable:

| | `witness_agent.evaluate` | `witness_agent.evaluate_api` |
|---|---|---|
| thinking | off, checked before the first episode | reasoning effort `max` (`--effort`) |
| output limit per reflection call | 8,192 tokens | 128,000 tokens, reasoning included (`--max-tokens`) |
| sampling | temperature 1.0; top_p and top_k are the server's defaults | the provider's defaults: no sampling parameter is sent |
| a reply cut off at the output limit | read like any other reply | takes no action, and the next prompt says so |
| default seeds (`--seeds`) | 0, 1, 2 | 0 to 4 |

## A model you serve yourself

To evaluate a model in Hugging Face format, for example a training checkpoint converted as described in
[training](training.md), serve it with SGLang on one GPU and run the evaluator. Run this inside the training container
([setup](training.md#setup)), which provides SGLang and httpx:

```bash
scripts/run_validation.sh HF_DIR runs/val
```

The script starts the server with `scripts/serve_policy.sh`, waits until it answers, checks its sampling defaults,
runs the evaluator and stops the server.

The output directory (`runs/val` above) must not exist yet; it receives the server log (`server.log`) and the
evaluator's outputs. Extra arguments go to the evaluator, for example `--games tw08 fv01 --seeds 0`. `PORT` (default
30000) and `CUDA_VISIBLE_DEVICES` select the port and the GPU.

To run the steps yourself, start the server in one shell and, once it answers, the evaluator in another:

```bash
scripts/serve_policy.sh HF_DIR

python -m witness_agent.evaluate --base-url http://127.0.0.1:30000/v1 --out runs/val
```

Any OpenAI-compatible server works if it needs no API key, applies the chat template with `enable_thinking=false`, and
by default samples with top_p 1.0 and top_k -1 (no top-p or top-k cutoff).

The evaluator plays every validation game with seeds 0, 1 and 2. An episode plays up to 5 levels, with at most 250
reflection calls in all and 300 actions per level. Before the first episode, the evaluator checks that the server
answers with thinking disabled.

It writes `results.txt`, `results.json`, one result file per episode (`episodes/<game>_seed<seed>.json`) and one
trace directory per episode (`traces/<game>/seed<seed>/`). The report gives the levels beaten per game and seed, the
total per seed and the mean and sample standard deviation over seeds. A seed's total is at most 40 (8 games of 5
levels); in the paper's runs tw12 had 4 levels, so the maximum was 39.

- An episode that ends with an error is reported as aborted, with its error, and left out of its game's mean. Its
  seed gets no total, and the mean over seeds uses the complete seeds only. Report only runs in which every episode
  completed.
- Exit status: 0 when every episode completed, 1 when some episode aborted, 2 for invalid arguments or an output
  directory that already holds results, 3 when the server check fails.
- Options: `--games` and `--seeds` select a subset, `--model` sets the served model name (default `witness-policy`),
  `--no-traces` skips the traces, and `--tasks PATH` reads other task rows.
- The default task file, `data/val.jsonl` of this checkout, is found only when `witness_agent` is imported from the
  checkout, as after the editable install; otherwise pass `--tasks`.
- All episodes run at once, one thread each. The observation version and, when `WITNESS_GAME_CATALOG` is set, the
  catalog module ([adding games](harness.md#adding-games)) are recorded in `results.json`.

## A model on OpenRouter

`witness_agent.evaluate_api` evaluates a model through [OpenRouter](https://openrouter.ai) on any task file. It plays
every (game, seed) episode with the canonical harness and sends every request only to providers with a
[zero-data-retention](https://openrouter.ai/docs/guides/features/zdr) endpoint. It keeps a journal of every reply, so
that an interrupted run can be resumed.

```bash
export OPENROUTER_API_KEY=...          # read from the environment only, never written anywhere

# Check the model and its providers, print the request and the number of episodes, and stop.
python -m witness_agent.evaluate_api --model anthropic/claude-opus-5.5 --tasks data/val.jsonl --out runs/opus55_val \
    --dry-run

# Play the episodes.
python -m witness_agent.evaluate_api --model anthropic/claude-opus-5.5 --tasks data/val.jsonl --out runs/opus55_val
```

Before the first episode, the evaluator checks the model and its zero-data-retention providers against OpenRouter's
public endpoint lists (no model is called) and builds the request. `--dry-run` prints that request and the number of
episodes, then stops; it needs no API key.

The check fails when the model has no zero-data-retention endpoint, or when an admitted endpoint does not accept the
`reasoning` parameter or allows fewer output tokens than `--max-tokens` (the error names that endpoint):

- A model whose endpoints do not accept `reasoning` cannot be run with this evaluator; write a policy for it instead
  ([your own agent](#your-own-agent-or-platform)).
- For an endpoint with a lower output limit, lower `--max-tokens` or admit only the other providers with
  `--providers`.

`--effort` is sent as given: neither the check nor `--dry-run` validates it.

- Other defaults: every provider with a zero-data-retention endpoint (`--providers` admits only the ones named) and
  at most 8 requests in flight (`--concurrency`). `--games` plays a subset of the task rows.
- No temperature is sent, so each endpoint samples at its provider's default. The request admits only endpoints that
  accept every parameter it sends, and the zero-data-retention endpoints of some models, such as
  `anthropic/claude-opus-5.5`, accept no temperature.
- A reply cut off at the output limit counts as a reflection call, takes no action, and the next prompt tells the
  model so. When more than 5% of a run's calls are cut off, the evaluator warns; raise the output limit (or lower
  the effort) before the run is scored.
- Every reply is appended to `journal/<game>_seed<seed>.jsonl` as it arrives.
- `--resume` continues a run in the same `--out` directory. Completed episodes are kept, and every other one is
  played again from the start, its journal answering the calls already made (each replayed call must carry exactly
  the prompts it was recorded with).
- Resume a run with the command that started it plus `--resume`: nothing checks that the model, the options and the
  task rows are the same, and `results.json` records the settings of the resuming command.
- The output directory also receives what `witness_agent.evaluate` writes. `results.json` records the request as
  sent (no messages, no key) and the totals of the replies: calls, cut-off and retried calls, tokens, cost, and the
  models and providers that served them.
- Exit status: 0 when every episode completed, 1 when some episode aborted, 2 for invalid inputs and 3 when the
  model and provider check fails (also when `OPENROUTER_API_KEY` is not set).

## Your own agent or platform

To evaluate an agent of your own, keep the harness and replace only the policy: a function `policy(system, user)`
that returns the reply as a string, or as a `witness_agent.episode.Completion` that also reports whether the reply
was cut off ([harness](harness.md#policies)).

This loop plays every row of a task file with seeds 0, 1 and 2 and writes the traces in the layout of
`witness_agent.evaluate`:

```python
from pathlib import Path

from witness_agent import EpisodeLimits, run_episode, write_trace
from witness_agent.catalog import make_game
from witness_agent.episode import load_tasks


def policy(system: str, user: str) -> str:
    return "<keep/>"   # stand-in: send the two messages to your model and return its reply


for task in load_tasks(Path("data/val.jsonl")):
    for seed in (0, 1, 2):
        result = run_episode(make_game(task.game_id), seed=seed, policy=policy,
                             limits=EpisodeLimits(task.max_levels, task.max_reflection_calls))
        write_trace(result, Path("runs/my_agent/traces") / task.game_id / f"seed{seed}")
        print(task.game_id, seed, result.levels_completed)
```

`make_game` from `witness_agent.catalog` also finds the games of `WITNESS_GAME_CATALOG`
([adding games](harness.md#adding-games)).

If the policy raises, `run_episode` raises `EpisodeAborted`, whose `partial` holds the episode so far; retry transient
errors inside the policy, as `HttpPolicy` does.

A platform that runs the games in another process passes `run_episode` a `witness_game.RemoteGameClient` instead of
the game ([harness](harness.md#playing-a-game-in-another-process)). The two evaluators show how the records, traces
and journals are written.

The leaderboard's test protocol plays 20 levels with at most 1,000 reflection calls per episode,
`EpisodeLimits(20, 1000)`. The cap covers the whole episode, so on the test games the first five levels can use more
than the 250 calls that validation allows.

## Metrics

Both evaluators report **levels beaten**: the number of levels, counted from level 1, that an episode solved before
it ended.

The leaderboard reports RHAE (relative human action efficiency, taken from ARC-AGI-3) and OAE (objective action
efficiency), computed by the maintainers from the per-level records of each episode:

- A solved level $`\ell`$ scores $`\mathrm{score}_\ell = \min\big((S^\ell_\mathrm{ref} / S^\ell_\mathrm{agent})^2,\ 1.15\big)`$,
  where $`S^\ell_\mathrm{agent}`$ is the agent's action count on the level (failed attempts and resets included). An
  unsolved or unreached level scores 0.
- For RHAE the reference count $`S^\ell_\mathrm{ref}`$ is the mean action count of successful human plays of the
  level; for OAE it is the level's optimal action count.
- A game scored over its first $`L`$ levels weights each level by its index and is capped at the weighted completion
  fraction, so efficiency on one level can make up for inefficiency on another but never for an unsolved level:

```math
\mathrm{Score}_g = 100 \cdot \min\left(\frac{\sum_{\ell=1}^{L} \ell \cdot \mathrm{score}_\ell}{\sum_{\ell=1}^{L} \ell},\ \frac{\sum_{\ell=1}^{L} \ell \cdot \mathbf{1}[\ell\ \text{solved}]}{\sum_{\ell=1}^{L} \ell}\right)
```

- The paper's headline number is RHAE-L5 ($`L = 5`$), next to RHAE-L20 ($`L = 20`$) over every level; the leaderboard
  says which of these scores it reports. Scores range from 0 to 100.

The human baselines are not part of this repository, and neither are the optimal action counts of the validation and
test games.

## The hidden test set and the leaderboard

The test games and their rules are not published and are not in this repository. The leaderboard on
[witnessbench.ai](https://witnessbench.ai/#model-evals) is computed on them by the maintainers with the test protocol
above. The paper's results were measured with harness v1, this release has v2 (see the
[note in the README](../README.md#-leaderboard)).

To have a model or agent evaluated there,
[open an issue](https://github.com/fleet-ai/witnessbench/issues/new?template=leaderboard-submission.yml) with the
leaderboard submission form. The form asks for the model or agent, how the maintainers can run it (an OpenRouter
model id, or a link to the weights or the code of your policy or agent) and its settings (for example, reasoning
effort and output-token limit).
