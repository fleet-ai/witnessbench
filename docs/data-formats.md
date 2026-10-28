# Data formats

## Task files

`data/train.jsonl`, `data/train_r1.jsonl`, `data/train_r2.jsonl` and `data/val.jsonl` hold one JSON object per line:

```json
{"prompt": "tw01", "metadata": {"game_id": "tw01", "max_levels": 10, "max_reflection_calls": 300}}
```

A row and its `metadata` have exactly these keys, and `prompt` must equal `metadata.game_id` (miles requires the key;
the code does not otherwise use it).

An episode of the row stops once `max_levels` levels are completed, and it may make at most `max_reflection_calls`
[reflection calls](harness.md#how-an-episode-runs); the plan of the last one runs to its end.

`witness_agent.episode.load_tasks(Path("data/val.jsonl"))` reads and checks a task file (it takes a `pathlib.Path`).

## Level files

`witness_env/levels/<game>.json` is a JSON array with one configuration object per level. Every object has:

- `cols` and `rows` (board size in cells);
- `end` (a node `[col, row]`), and `start` (a node) or `starts` (a list of nodes);
- optionally `breakpoints` (broken edges as node pairs);
- the marker keys of its game (for example `dots`, `squares`, `censers`). Maps from cells to values, such as square
  colors, use `"col,row"` keys.

For example, level 1 of tw02 is a board one cell wide and two cells tall, with the start at its bottom-left node, the
end at its top-right node and squares of colors 6 and 10 in its two cells:

```json
{"cols":1,"rows":2,"start":[0,2],"end":[1,0],"squares":{"0,0":6,"0,1":10}}
```

Level files give nodes and cells as `[col, row]`; the prompts give them as `(row, col)`, so this start is `(2,0)` in
the prompt.

## Episode traces

`witness_agent.write_trace(result, directory)` writes three files.

`calls.jsonl` has one line per reflection call:

- `index` (0-based);
- `level` (0-based: the levels completed when the call was made) and `level_step` (the actions already taken on that
  level);
- `system` and `user` (the two prompts);
- `response` (the reply text as the policy returned it);
- `plan` (the action ids read from the reply, at most 16, or `null` when the reply asked for no action or was cut off
  at the output limit);
- `meta`, only when the policy returned a `Completion`: the finish reason, whether the reply was cut off, token usage
  and cost, the model and provider that served it, the number of requests made for it and the seconds they took.

`steps.jsonl` has one line per action:

- `level` (0-based) and `step` (0-based index of the action within its level);
- `action` (0 RESET, 1-4 moves, 5 CONFIRM);
- `actor` (`plan`, or `random` for the random fallback move);
- `noop_reset` (`true` for a RESET that was not sent to the engine).

`summary.json` has:

- `game_id`, `seed`, `max_levels`, `levels_completed`, `reflection_calls`, `steps` and `observation_version`;
- `random_fallback_steps` (the actions of the random fallback);
- `stop_reason`;
- `resets`: one `{"level", "step"}` object per RESET the engine executed, where `step` counts the level's actions up
  to and including the RESET;
- `completions`, only when some call has `meta`: the numbers of calls, cut-off calls, retried calls and extra
  requests, the summed token counts and cost, and the models and providers that served them.

The stop reason is one of:

- `max_levels`: the episode completed `max_levels` levels, also when the last of them was the game's last level;
- `game_won`: the game's last level was completed before `max_levels` levels were, as under `EpisodeLimits(20, 1000)`
  on a public game;
- `level_budget`: a level's 300 actions were used up;
- `call_budget`: the reflection calls allowed by `EpisodeLimits` were used up;
- `policy_stop`: the policy returned `None`.

An episode that ended with an error has `aborted`, and the evaluator's record of it gives the error.

## Evaluator outputs

- `results.txt`: the report in text form.
- `results.json`: `settings`, `games`, `seed_totals`, `max_total`, `mean_total`, `std_total` and `episodes`.
  - Each entry of `games` has `game_id`, `max_levels`, `levels_beaten` per seed, `mean` and `std`. Its `max_levels` is
    the most levels an episode can beat: the row's `max_levels`, or the game's number of levels if that is smaller.
  - `witness_agent.evaluate_api` also writes `completions`, which holds the `completions` of all episodes added up and
    `truncated_share`, the share of calls cut off at the output limit.
- `episodes/<game>_seed<seed>.json`: the record of one (game, seed) episode:
  - `game_id`, `seed`, `max_levels`, `max_reflection_calls`, `reflection_calls`, `steps`, `stop_reason` and
    `observation_version`;
  - `levels_beaten`: the levels completed, at most `max_levels`; `null` when the episode aborted;
  - `levels_completed`, also for an aborted episode;
  - `error`: `null`, or the error that aborted the episode.
- `traces/<game>/seed<seed>/`: the episode's trace (above), unless `--no-traces` is given (`witness_agent.evaluate`
  only).
- `journal/<game>_seed<seed>.jsonl` (`witness_agent.evaluate_api` only): every reply, as it arrived, for `--resume`.
