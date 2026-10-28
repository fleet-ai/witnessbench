# Out-of-domain transfer evaluations

This directory reproduces the paper's transfer table. It compares Qwen3.8-27B with the RL checkpoint
[`WitnessGym-Qwen3.8-27B-RL-broadcast-step20`](https://huggingface.co/Guanghan-Ning/WitnessGym-Qwen3.8-27B-RL-broadcast-step20)
on four public benchmarks, using the same code, settings and repetitions that produced the paper's numbers.

## One command

```bash
GPUS=0,1,2,3,4 OPENROUTER_API_KEY=sk-or-... bash transfer/reproduce.sh
```

The command does the following:

1. It clones the four benchmarks at their pinned commits, applies our patches, and checks the md5 of every patched file.
2. It builds the Python environments and downloads both models plus the local judge.
3. It evaluates the base model, then the RL checkpoint.
4. It prints one summary per benchmark: base vs. RL, the paired difference, and a 95% bootstrap interval.

Results go to `transfer/work/results/` (or pass a directory as the first argument).

**Requirements**
- Linux with NVIDIA GPUs that can hold a 27B model in bf16 (we used B300s).
- SGLang `0.5.19.dev46+g97e43b4`. We ran inside `docker.io/radixark/miles@sha256:0c218363e4f4ffa8b98a7a9d357f88eb92780f0b2337099d436557dbbdb7279e`.
- Python 3.12, `git`, `curl`, and the `hf` CLI.
- Root access: WILT and Failing-to-Falsify run model-written code as user `nobody`.

The OpenRouter key pays for the two API judges the original benchmarks use.

## Settings

| Benchmark | Upstream (pinned commit) | Thinking | Output cap | Reps | Judge |
|---|---|---|---|---|---|
| CipherBank | [Leey21/CipherBank](https://github.com/Leey21/CipherBank) `05b450c`; data [yu0226/CipherBank](https://huggingface.co/datasets/yu0226/CipherBank) rev `d2f8f7c` | on | 32,768 tokens | 2 | — (exact match) |
| PhysGym (L4) | [principia-ai/PhysGym](https://github.com/principia-ai/PhysGym) `fe68079` | on | 32,000 tokens, context 65,536 | 3 | `google/gemini-2.5-flash`<br>(upstream's in-loop judge) |
| WILT | [riotgames/wilt](https://github.com/riotgames/wilt) `6344f04` | off | 4,096 tokens | 5 | — |
| Failing-to-Falsify | [ayushjhaveri/llm-confirmation-bias](https://github.com/ayushjhaveri/llm-confirmation-bias) `97f6433` | off | 1,024 new tokens | 5 | `meta-llama/llama-3.3-70b-instruct`<br>(OpenRouter, CoreWeave fp16)<br>+ local `Qwen2.5-Coder-32B-Instruct` |

Every model is served by one SGLang server per GPU (tensor parallelism 1). Every request uses
`--sampling-defaults openai`: temperature 1.0, top-p 1.0, no top-k. The server flags are in `run_all.sh` (`serve`).

## What the patches change, and why

The patches (`patches/*.diff`) do three things:
- point each harness at a self-hosted OpenAI-compatible endpoint;
- make the runs safe and scorable at scale;
- leave each benchmark's tasks, prompts and scoring rules unchanged.

Each changed line carries a `witness eval patch` comment (or `patched` for PhysGym).

**CipherBank**
- Accepts any served model name, where upstream had a hard-coded whitelist.
- Forwards `max_tokens`. Upstream never sent it, so SGLang's default of 128 tokens truncated every answer.
- Forwards a per-repetition seed.
- Logs each call.

**WILT**
- Adds an SGLang model class.
- Scores the final `lambda` in a sandboxed child process instead of with in-process `eval`.
- Separates endpoint failures, which are re-run, from empty model replies, which count as model behavior.
- Fixes the `repeats` key typo, which made upstream always report 0.
- Caps reminders after the test budget so episodes cannot loop forever.
- Makes optional imports lazy.

**Failing-to-Falsify**
- Lets a served model use the harness's OpenAI path.
- Sets thinking-aware output caps and logs each call.
- Lets an OpenAI-compatible judge stand in for the original local vLLM judges.
- Matches judge verdicts by step number rather than list position (a fix for an upstream misalignment).
- Blocks dunder access in the compatibility sandbox.
- Writes judge outputs atomically, so re-runs can resume.

**PhysGym**
- Reads the agent's temperature and output cap from the environment. Upstream hard-coded 0.2 and 4,000 tokens.
- Clamps experiments to the remaining quota. An over-budget request used to crash the problem.
- Retries transport errors and shrinks `max_tokens` on context-length errors.
- Bounds `sympy.simplify` with a timeout.
- Treats complex or non-numeric predictions and malformed replies as parse failures instead of crashing.

## Files

- `reproduce.sh`: the one command above.
- `setup.sh`: clone, pin, patch, md5 check, environments, CipherBank data.
- `run_all.sh run ARM MODEL OUT`: evaluates one model.
  - `ARM=base` serves the model as `qwen3.8-27b-base`; `ARM=step20` serves it as `witnessgym-qwen3.8-27b-rl-broadcast-step20`.
  - Any other ARM name is used as the served name as-is, so you can evaluate your own checkpoint.
- `run_all.sh summarize OUT`: prints the summaries.
- `bin/`: the per-benchmark launchers, the CipherBank driver and the summarizers used for the paper.
  - Only paths, defaults and model names were generalized.
  - Each launcher documents its options in its header comment. For example, `--reps 1 --split lite` on `bin/launch_wilt.sh play` gives a quick smoke run.
- `patches/`: one diff per benchmark against the pinned commit. `patched_md5.txt` holds the md5 of every patched file.
- `requirements-physgym.txt`: the frozen PhysGym environment.

## Cost

`run_all.sh` splits each model's work into five lanes:
- WILT + Failing-to-Falsify, about 50 minutes;
- CipherBank, about 9 hours for its 2 repetitions, which run one after another;
- one lane per PhysGym repetition, about 11 hours each.

With 5 GPUs, all lanes run in parallel, so one model takes about 12 hours (PhysGym is the slowest lane) and both models about a day. With fewer GPUs, the lanes queue up on the GPUs you give it.
