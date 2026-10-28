# Contributing

Bug reports, fixes and documentation improvements are welcome. To have a model or an agent evaluated on the private
test games, open a
[leaderboard submission](https://github.com/fleet-ai/witnessbench/issues/new?template=leaderboard-submission.yml)
instead. Report security problems privately, as described in [SECURITY.md](SECURITY.md).

## Setup

Python 3.12 or later is required.

```bash
git clone https://github.com/fleet-ai/witnessbench.git   # or your fork
cd witnessbench
python3.12 -m venv .venv                                 # or any Python 3.12 or later
source .venv/bin/activate
pip install -e ".[test]"
```

## Tests

```bash
python -m pytest -q                        # about 3 minutes
WITNESS_SLOW_TESTS=1 python -m pytest -q   # also runs the slow tests; about 11 minutes
```

The tests need no GPU, network access or API key. Many of them compare the harness's prompts, the games' frames and
board states, and the training code's outputs with stored copies under `tests/golden/`. When one of these comparisons
fails, your change has altered that output.

## Issues

Search the existing issues first. A good bug report gives:

- what you ran, what you expected and what happened instead;
- the shortest command or code that shows the problem, and the full error message;
- the commit (`git rev-parse --short HEAD`), the Python version and the operating system.

## Pull requests

- Make one change per pull request, and say what it changes and how you tested it.
- Run the tests. CI runs them on Python 3.12, 3.13 and 3.14 and checks that the package builds and installs.
- Update the guides in `docs/` when you change documented behavior.
- Update a stored copy under `tests/golden/` only when the new output is intended, and say in the pull request why
  it is correct.
  The prompt digests in `tests/golden/agent/obs_v2_prompt_hashes.json.gz` are recorded again with
  `python tests/test_agent_obs_v2_golden.py --record`.
