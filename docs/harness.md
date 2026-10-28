# The canonical harness

`witness_agent` is the canonical harness: the text-only agent loop that every model plays through. It calls the model
each time it needs a new plan; each such call is a *reflection call*.

A model is plugged in as a *policy*, a function `policy(system, user) -> str` that returns the model's reply, so any
model or agent can be compared under the same conditions.

This repository has harness v2. Its prompt format is observation version 2, which every trace and `results.json`
record as `observation_version`. The results in the paper were measured with v1 (see the
[note in the README](../README.md#-leaderboard)).

## How an episode runs

One episode plays one game from its first level. The model never sees pixels: the board is shown as text built from
the game's state (an ASCII lattice with a legend, then coordinate lists as `(row, col)`).

**Reflection calls.** Whenever no planned action is left, the harness makes a reflection call. It sends a system
prompt and a user message with the board, what each action has done so far, the last steps, the rule list, notices
from the harness (for example, that part of a plan was ignored) and the recent attempts
([the observation](#the-observation)). There is no conversation history.

When a level is solved, the rest of the plan is dropped, and the next reflection call is made on the new level.

**The reply.** It uses tags:

- `<add>rule</add>` and `<delete>n</delete>` edit a list of at most 15 rules, and `<keep/>` keeps it. At most two
  `<add>` tags are applied per reply; the harness drops the rest and says so in the next prompt.
- `<plan>ids</plan>` gives up to 16 actions: `0` RESET, `1` up, `2` down, `3` left, `4` right, `5` CONFIRM.
- `<meta>` is free analysis that the harness ignores.

The prompt explains RESET and CONFIRM but not the four moves. It shows a move's direction once the harness has seen
the move change the path. A reply can look like this:

```text
<meta>ACTION4 moved the path right; ACTION3 did nothing at the left edge.</meta>
<add>ACTION4 moves the path one node right.</add>
<delete>2</delete>
<plan>4,4,1,5</plan>
```

If a reply has no plan, or its plan holds no valid action id, the harness makes one random move (up, down, left or
right) and makes a new reflection call at the next step.

**RESET.** The engine treats a `RESET` with nothing to undo as a restart of the whole game. So when no action has
been sent since the level began or since the last `RESET`, the harness does not pass a `RESET` on to the engine. It
still counts as one action, leaves the board unchanged and is marked `noop_reset` in the trace. An episode therefore
never goes back to level 1.

**Rule checks.** The harness keeps a confidence for each rule and checks the rules against the moves recorded on the
level. Only two kinds of claim can be refuted: that an action always moves the path, and that an action does nothing.

A rule making such a claim is removed once at least 3 outcomes of the actions it names are recorded on the level and
at least 30% of them contradict it (an outcome is the latest result of an action from one board state).

When a level is solved, every rule first counts one more confirming check, and the rules then carry over to the next
level at reduced confidence. Rules that name board coordinates are dropped, and so are rules whose confidence is still
below 0.4, which can happen only to rules that the recorded moves contradicted.

**Budgets and stops.** Every level allows 300 actions; if the level is not solved within them, the episode ends (stop
reason `level_budget`). The episode also ends:

- once `max_levels` levels are completed (`max_levels`);
- when the last level of the game is solved before that (`game_won`);
- when its reflection calls are used up (`call_budget`; the last plan still runs to its end).

The validation task file (`data/val.jsonl`) allows 250 reflection calls per episode, the training task files 300, and
the leaderboard's test protocol 1,000 for 20 levels.

**Determinism.** Given the same replies, an episode is deterministic. Its only randomness is in the random moves
after replies without a valid plan, which are seeded from the episode seed and the game id. The seed changes nothing
else: every seed plays the same levels.

## The observation

Each reflection call sends two messages, both built afresh from the harness state.

The system prompt holds general advice on discovering rules, hard limits (at most 2 new rules per reply, 15 rules in
the list and 16 actions per plan), the same `Known Action Semantics` section as the user message, and what CONFIRM
and RESET do. Only that section changes from call to call.

The user message has these sections, in this order:

- `## Current Board State`: the board size, the agent's position, the level, the budget (actions used on this level
  and reflection calls used), the legend and the board drawn as text.

  Below the board come the start and end nodes and the lists the level needs, with coordinates as `(row, col)`:
  `Mandatory dots` (each marked visited or not visited), `Cell content`, `Color groups` and `Value groups` (the cells
  that share a color or a value) and broken edges. Then come the path so far, what each action has done on this level
  and the last change.

  The lists are sorted (the path keeps its order), and `&` on the board marks a cell that holds two or more kinds of
  object (each is listed under `Cell content`).
- `## Known Action Semantics`: what the harness has seen each move do (its direction, how often it changed nothing,
  the move that undoes it) and how many CONFIRMs were rejected on this level.
- `## Recent Observations`: one example movement for each move that has extended the path, then up to the last 8
  actions of the level with their effects.
- `## Current Knowledge`: the numbered rule list. The harness tags a rule `✓` when the moves recorded on this level
  confirm it, `⚠DISPUTED` when some of them contradict it, and `[carried]` when it comes from the previous level.
- `## Harness notices`, when there are any: what the harness changed or ignored since the last reply, such as ignored
  plan text, a dropped `<add>` or a rule it removed.
- `## Recent Attempts (this level)`, once a plan has run to its end: the last two plans of the level and how each
  ended.
- `## Instructions`: what to look for and the reply format.

A rejected CONFIRM sends the path back to the start. The prompt then repeats the submitted path in that action's line
under `Recent Observations` and under `Recent Attempts`.

The harness tells game states apart by what the prompt shows of the board, not by pixels. It uses that to spot
actions that change nothing, to check the rules and to compute the per-action statistics.

The first user message of an episode on `ft12`, for example, contains this block:

```text
## Current Board State
Grid: 4×4 dot intersections (cursor moves on dots, not in cells) | Agent at row=0, col=3
Level: 1/10
Budget: actions on this level 0 of 300 | reflection calls used 0 of 300
Coordinates: (row, col), 0-indexed, (0,0) at top-left.
Legend: @=agent  E=end  Q=colored_square  g=gargoyles

Board:
   0 1 2 3 
 0       @
      Q   
 1        
    g Q Q 
 2        
    Q     
 3 E      

Start: (0,3)
End:   (3,0)

Cell content:
  (0,1): square color=6
  (1,1): square color=6
  (1,2): square color=10
  (2,0): square color=6
  (1,0): gargoyles color=10

Color groups (cells sharing the same color):
  Squares of color 6: (0,1), (1,1), (2,0)
  Squares of color 10: (1,2)
  Gargoyles of color 10: (1,0)
```

`write_trace` saves the complete system prompt and user message of every reflection call in `calls.jsonl`
([Policies](#policies)).

## Policies

```python
from pathlib import Path

from witness_agent import EpisodeLimits, run_episode, write_trace
from witness_env import make_game


def policy(system: str, user: str) -> str:
    return "<keep/>\n<plan>4,4,1,3,3,1,4,4,5</plan>"   # a fixed plan instead of a model


result = run_episode(make_game("tw01"), seed=0, policy=policy,
                     limits=EpisodeLimits(max_levels=1, max_reflection_calls=3))
print(result.levels_completed, result.stop_reason, len(result.calls))   # 1 max_levels 1
write_trace(result, Path("trace_demo"))
```

A policy returns one of three things:

- the reply as a string;
- a `witness_agent.episode.Completion`: the reply together with how it was produced (the finish reason, token usage
  and cost, and the model and provider that served it);
- `None`, which ends the episode at once (stop reason `policy_stop`).

A `Completion` with `finish_reason="length"` was cut off at the output limit: the call counts, but nothing in the
reply is applied, no action follows, and the next prompt tells the model why. Because such a reply takes no action, a
policy that can return one needs a `max_reflection_calls` limit; without it, an episode whose replies keep being cut
off never ends.

Any exception inside the episode, including one raised by the policy, makes `run_episode` raise `EpisodeAborted`; its
`partial` attribute holds everything recorded up to then.

`write_trace` writes the episode's `calls.jsonl`, `steps.jsonl` and `summary.json` ([data formats](data-formats.md)),
for an aborted episode as well.

### Against an OpenAI-compatible endpoint

`HttpPolicy` sends each reflection call as one chat-completion request (temperature 1.0, `max_tokens` 8192, thinking
disabled through `chat_template_kwargs`). It sends no API key, so the server must accept requests without one, as the
local SGLang server of `scripts/serve_policy.sh` does.

```python
from witness_agent import EpisodeLimits, run_episode
from witness_agent.http_policy import HttpPolicy
from witness_env import make_game

policy = HttpPolicy("http://127.0.0.1:30000/v1", "witness-policy")
result = run_episode(make_game("tw08"), seed=0, policy=policy,
                     limits=EpisodeLimits(max_levels=5, max_reflection_calls=250))
print(result.levels_completed, result.stop_reason)
```

A request that fails with a connection error, a timeout, HTTP 429 or 5xx, or a response without reply text is
repeated with growing pauses (1 s up to 60 s) for at most one hour per call. Any other HTTP error ends the episode at
once.

Models on OpenRouter are evaluated with `witness_agent.evaluate_api` ([evaluation](evaluation.md)).

## Playing a game in another process

The game can run in a separate process or on another machine, so the harness side needs neither the game code nor
its levels.

`run_episode` accepts either an in-process game or an engine-free `witness_game.GameClient`. The client exposes
`game_id`, `reset()` and `step(action)`; each action returns a typed `GameFrame` with the level counts, whether the
game is won and the visible board (`BoardTruth`).

The harness builds the prompts and keeps the rule list, the plans, the random fallback move, the budgets and the
stopping rules. The game process keeps the engine state, the levels and the rendering.

A server wraps its game in `witness_env.client.LocalGameClient`, then `witness_game.service.GameService`, and exposes
`describe()` as `game_info` and `action(session_id, sequence, action)` as `game_action` through a transport of its
choice.

A harness constructs `RemoteGameClient(call, info)`, where `call(name, arguments)` sends one request over that
transport and `info` is the reply to `game_info`, and passes the client to `run_episode`. Here a function stands in
for the transport:

```python
import json

from witness_agent import EpisodeLimits, run_episode
from witness_env import make_game
from witness_env.client import LocalGameClient
from witness_game import RemoteGameClient
from witness_game.service import GameService

# Game side: one service per episode.
service = GameService(LocalGameClient(make_game("tw01")))
handlers = {"game_info": service.describe, "game_action": service.action}


# Harness side: requests and replies cross the transport as plain JSON.
def call(name, arguments):
    reply = handlers[name](**json.loads(json.dumps(arguments)))
    return json.loads(json.dumps(reply))


client = RemoteGameClient(call, call("game_info", {}))
result = run_episode(client, seed=0, policy=lambda system, user: "<plan>4,4,1,3,3,1,4,4,5</plan>",
                     limits=EpisodeLimits(max_levels=1, max_reflection_calls=3))
print(result.levels_completed, result.stop_reason)   # 1 max_levels
```

The harness side needs only `witness_agent` and `witness_game`. There, `run_episode`, `write_trace` and
`RemoteGameClient` run without `arcengine`, `witness_env` or NumPy, and so does
`witness_agent.episode.load_tasks(Path(path), known_game_ids=ids)`: given the game ids the server admits, it reads a
task file without loading any game code. The evaluators and `witness_agent.catalog` import `witness_env`, and
`HttpPolicy` needs httpx.

The wire protocol `witness.game.v1` (`witness_game/wire.py`) turns each `GameFrame` into plain JSON values and back
without changing it: coordinate order, tuple keys, marker order and immutable types survive the round trip. It
carries the level counts and the visible board, never an engine object.

A session is one ordered stream of actions, each with a sequence number:

- A repeated delivery of the last request (same number, same action) gets the saved result and is not played again.
- A different action under that number, or any number other than the next one, is rejected.
- After a failed action, a lost reply or a reply that does not match its request, no further action is sent or
  accepted.
- A restarted game process has a new session id, so requests of the old session are rejected.

The library has no authentication and starts no server, so run it behind an authenticated transport.

To check a transport or a change to the harness, run:

```bash
python -m pytest tests/test_game_client.py tests/test_agent_episodes.py tests/test_agent_obs_v2_golden.py
```

These tests play recorded episodes both in one process and across a real JSON/HTTP connection. They check that the
results and trace files are identical and that the prompts match the stored ones byte for byte. They also check
cut-off replies, the random fallback and RESETs on every game of `witness_env`, repeated and failed requests, and that
the harness side runs without the game code. No model is called.

## Adding games

To play games that are not in this repository, such as held-out games of your own, put them in an importable module
and name it in the environment variable `WITNESS_GAME_CATALOG`. Both evaluators and the task-file loader
(`load_tasks`) then accept that module's game ids.

The module must define `PRIVATE_GAME_IDS`, a tuple of its game ids (none of them an id of `witness_env`), and
`make_game(game_id)` with the contract of `witness_env.make_game`; see `witness_agent/catalog.py`.

The game that `make_game` returns must report `game_id` as its own id (`WitnessGame.GAME_ID`, which also names its
level file, `levels/<GAME_ID>.json` in the package named by `LEVELS_PACKAGE`). Episodes are recorded under the game's
own id, so a mismatch shows up only after every episode has been played.

Training refuses such games: `scripts/train.py` trains on the games of `witness_env` only.

```bash
WITNESS_GAME_CATALOG=my_games python -m witness_agent.evaluate --base-url http://127.0.0.1:30000/v1 \
    --tasks my_tasks.jsonl --out runs/my_games
```
