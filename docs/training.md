# Training

`witness_rl`, `configs/` and `scripts/train.py` train a policy on the training games with *broadcast GRPO*: plain
GRPO with one advantage per episode, applied to every [reflection call](harness.md) of the episode. The episode reward
counts the completed levels, minus a small cost per reflection call.

This code is a plug-in for the open-source [miles](https://github.com/radixark/miles) trainer, with configurations for
a 27B model on 8 B300 GPUs and a 9B model on 4 GPUs.

The usual protocol is to train on a training set, select checkpoints on the validation games
([evaluation](evaluation.md)), and have the maintainers run the test games for the leaderboard.

## Setup

Training uses the unmodified miles trainer at commit
[`cd464a1c4`](https://github.com/radixark/miles/commit/cd464a1c4c91546ea43abb2233b97bc1bc90eb97): the 27B
configuration on one node with 8 B300 GPUs, the 9B configuration on 4 GPUs (a 4-GPU node, or 4 of the 8 GPUs of a
node that runs nothing else).

Measured with this release on one node of 8 B300 GPUs, a 27B training step took 1.4 to 1.7 hours for task files of 21
to 26 games (20 steps on r1 took 28 hours), so a full 30-step run on `data/train.jsonl` takes about 45 hours. The
paper reports about 4 hours per step (about 119 hours per run) for its runs, which used harness v1.

The recipe was run in the miles Docker image below, which holds miles at commit `cd464a1c4`. Pull it by its digest,
then start it with the `docker run` command of the [miles installation
guide](https://miles.radixark.com/docs/getting-started/installation#method-1-docker-recommended), with this image in
place of `radixark/miles:latest`:

```bash
docker pull radixark/miles@sha256:0c218363e4f4ffa8b98a7a9d357f88eb92780f0b2337099d436557dbbdb7279e
```

If the image is ever unavailable, build one from miles'
[`docker/Dockerfile`](https://github.com/radixark/miles/blob/cd464a1c4c91546ea43abb2233b97bc1bc90eb97/docker/Dockerfile)
at that commit. Such a build takes miles, Megatron-LM and SGLang from the current heads of their branches, so it can
differ from the image above; the check below puts miles itself back at `cd464a1c4`.

In the container, miles is installed from its checkout in `/root/miles` and Megatron-LM from `/root/Megatron-LM`.
Check the checkout with `git -C /root/miles rev-parse --short=9 HEAD`; if it does not print `cd464a1c4`, run
`git -C /root/miles checkout cd464a1c4`.

Put the Hugging Face checkpoints `Qwen3.8-27B` and `Qwen3.5-9B` under `$MODEL_DIR`, then run, inside the container
and from this directory:

```bash
pip install --no-deps -e .
pip install --no-deps arcengine==0.9.3
export MODEL_DIR="$PWD/models" CKPT_DIR="$PWD/checkpoints"   # or other directories

# Convert the base model to torch_dist format.
bash scripts/convert_checkpoint.sh to-torch-dist 27b "$MODEL_DIR/Qwen3.8-27B" "$MODEL_DIR/Qwen3.8-27B_torch_dist"

# Print the model type, the GPU count and the miles arguments, and launch nothing.
python scripts/train.py --model 27b --run-name demo-27b --dry-run

# Train.
python scripts/train.py --model 27b --run-name demo-27b

# Convert the last checkpoint back to Hugging Face format.
bash scripts/convert_checkpoint.sh to-hf 27b "$CKPT_DIR/demo-27b/iter_0000029" "$MODEL_DIR/Qwen3.8-27B" runs/hf-27b
```

- `--no-deps` leaves the container's own packages untouched; the container already provides what the code needs
  besides arcengine (numpy, pydantic, PyYAML, httpx).
- `MODEL_DIR`, `CKPT_DIR` and `DATA_DIR` locate the base models, the checkpoints and the task files.
  `scripts/train.py` uses `models/`, `checkpoints/` and `data/` of this directory for any of them that is unset, but
  the conversion commands need `MODEL_DIR` and `CKPT_DIR` set, as in the `export` line.
- Both conversions take the miles model type from `configs/rl_<model>.yaml`. Conversion to torch_dist uses as many
  GPUs as training. Conversion back to Hugging Face format runs on the CPU.
- Use `scripts/convert_checkpoint.sh` for the conversion back rather than miles' converter alone: the saved
  checkpoints lack the `common.pt` file that miles' converter reads, and the script supplies the model structure in
  its place.
- `--model 9b` trains the 9B model. Convert it with `to-torch-dist 9b "$MODEL_DIR/Qwen3.5-9B"
  "$MODEL_DIR/Qwen3.5-9B_torch_dist"` and its checkpoints with `to-hf 9b ITER_DIR "$MODEL_DIR/Qwen3.5-9B" OUT_DIR`.
- `--num-rollout N` stops the run after N training steps instead of 30. As a smoke run, `--num-rollout 1` plays one
  full step (1.4 to 1.7 hours for the 27B model on 8 B300 GPUs) and writes one checkpoint. Give such a run its own
  `--run-name`, because re-running a command resumes from the checkpoints of its run name.
- `--seed N` sets miles' random seed to N (1 or more) instead of the recipe's 42, for a second training seed of the
  same setup.
- A checkpoint `$CKPT_DIR/NAME/iter_XXXXXXX` is saved after every step; `iter_k` holds the policy after k+1 steps
  (`iter_0000029` ends a full run).
- Each checkpoint holds the model and its optimizer state, hundreds of GB for the 27B model, and a full run writes 30
  of them: give `$CKPT_DIR` room for them or delete older iterations you do not need.
- `--trace-dir DIR` writes the trace of every training episode to `DIR/<game>/seed<seed>/`.
- Weights & Biases logging is off; `--wandb-project NAME` turns it on and reads the key from the `WANDB_API_KEY`
  environment variable.
- miles stops any SGLang and Ray processes already running on the node when a run starts, so use a dedicated node.
- Sampling on SGLang is not deterministic, so runs are not bit-reproducible.

| | 27B | 9B |
|---|---|---|
| GPUs | 8 | 4 |
| training parallelism | tensor 4 (sequence parallel), data 2 | tensor 2 (sequence parallel), data 2 |
| rollout engines | one SGLang engine, tensor 8 | four SGLang engines, tensor 1 |
| engine memory fraction | 0.8 | 0.6 |
| max tokens per GPU | 8192 | 9216 |
| optimizer state | offloaded to CPU | on GPU |

## Recipe

Both models use the same recipe (`configs/rl_common.yaml`):

- 30 training steps. Each step plays every training game 8 times (176 episodes on `data/train.jsonl`, each of up to
  10 levels and 300 reflection calls) and makes one optimizer update.
- Adam (`--optimizer adam`) with a constant learning rate of 1e-6, betas 0.9 and 0.98, and weight decay 0.1.
- The policy ratio is clipped to [0.8, 1.28] (`eps_clip` 0.2, `eps_clip_high` 0.28). There is no KL or entropy term.
- Sampling at temperature 1.0, with at most 8192 new tokens per call and thinking disabled.
- Seed 42, and a checkpoint after every step.

Rewards and advantages:

- An episode's reward is 1 for each completed level, plus 0.5 if it completes the row's `max_levels` (10) levels,
  minus 0.005 for each reflection call.
- In the code, each call's reward is its cost plus what the episode gains before the next call; the episode reward is
  the sum of these call rewards.
- Advantages use only the step's episodes that ended without an error. Each reward is first centered on the mean
  reward of its game's episodes, without dividing by their spread.
- The centered values of all games are then standardized together: minus their mean, divided by their standard
  deviation. Every call of an episode is trained with the episode's advantage.
- A call is trained on only if its reply ended with a stop token. Every other call (cut off at the token limit,
  aborted by the engine, or any other finish reason) keeps its cost but is not trained on.
- An episode that ends with an error gets advantage 0 and receives no update. The step still trains on the others,
  also when every episode of a game failed (the log names such games).
- An episode that fails before its first reflection call stops the run with an error naming the game, the seed and
  the error type. Nothing is saved for that step; re-running the command resumes from the last saved step.

## Training sets

| task file | games | episodes per step | what it is |
|---|---|---|---|
| `data/train.jsonl` (default) | 22 | 176 | the paper's 22 training games; tw09 and tw10 now have 10 levels, where the paper's runs had 5 |
| `data/train_r1.jsonl` | 21 | 168 | revised set r1: six bad levels replaced in four games, one rendering fix, ft13 left out |
| `data/train_r2.jsonl` | 26 | 208 | r1 plus five training-only games with region rules |

`--tasks PATH` trains on another task file; each step then plays every row of that file 8 times.

In r1, revised copies with the same rules replace tw02, tw03, tw10, ft03 and ft06 (`witness_env.TRAIN_R1_IDS`). The
copies replace six levels on which every path won or that copied another level, and ft06r1 draws the value-3 buoy
inside its cell on 4x4 boards. r1 also leaves out ft13, which `data/train.jsonl` includes, because its rule is close
to the rule of the validation game fv03.

r2 adds rg01, rg02, rg03, rg04 and rg06 (`witness_env.TRAIN_R2_IDS`, 10 levels each). Their rules read a property of
a marker's region: how many other regions it borders (rg01), its half-turn symmetry (rg02), a weighted balance about
a fulcrum (rg03), the parity of its size (rg04), and how many board corners it holds (rg06).

```bash
python scripts/train.py --model 27b --run-name r1-27b --tasks data/train_r1.jsonl

python scripts/train.py --model 27b --run-name r2-27b --tasks data/train_r2.jsonl
```

Training plays the games of `witness_env` only; games added through `WITNESS_GAME_CATALOG`
([adding games](harness.md#adding-games)) are refused.
