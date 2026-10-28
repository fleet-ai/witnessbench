# Third-party notices

This repository contains no third-party source code apart from the short excerpts in the patch files
of `transfer/patches/` (see the last section). The software listed below is installed separately and
is used under its own license.

## arcengine 0.9.3

The game engine. The games in `witness_env/` subclass its `ARCBaseGame`, and `pip install -e .`
installs it from PyPI as a dependency. It is distributed under the MIT License:

```text
MIT License

Copyright (c) 2026 ARC Prize Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## miles

The reinforcement-learning trainer, used unmodified at commit
[`cd464a1c4`](https://github.com/radixark/miles/commit/cd464a1c4c91546ea43abb2233b97bc1bc90eb97) of
https://github.com/radixark/miles. `witness_rl/` and `scripts/train.py` call it; it is not included
here. It is distributed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0).

`tests/stubs/miles/` is a small stand-in for the parts of the miles interface that `witness_rl` uses,
so that the tests run without miles. It reproduces the names, signatures and observable behavior of
those parts and is written independently of the miles source.

## Other dependencies

numpy (BSD-3-Clause, with parts under 0BSD, MIT, Zlib and CC0-1.0), pydantic (MIT, installed with
arcengine), httpx (BSD-3-Clause), PyYAML (MIT) and pytest (MIT) are installed from PyPI. Training
also uses the software provided by the miles container, including PyTorch, Megatron-LM, SGLang, Ray
and transformers, each under its own license.

## Transfer benchmarks

`transfer/setup.sh` clones four benchmarks from GitHub at pinned commits and applies the patches in
`transfer/patches/`. The benchmarks are not included in this repository. Each patch file holds the
lines it changes and a few unchanged lines around them, so it contains short excerpts of the
benchmark's source.

- **PhysGym**, https://github.com/principia-ai/PhysGym at commit
  `fe68079c0921029dde679ed44bc3192dd3b270ab`. MIT License (below).
- **CipherBank**, https://github.com/Leey21/CipherBank at commit
  `05b450ca8f11b966fb73d06278dea213e87540f9`. The repository has no license file. Its dataset,
  https://huggingface.co/datasets/yu0226/CipherBank at revision
  `d2f8f7cc7ca6a0c1b8aa25898747a9b80ed67729`, is distributed under the Apache License, Version 2.0, and
  is downloaded, not included.
- **WILT**, https://github.com/riotgames/wilt at commit `6344f04b3eb9119664c87c4d1b6cd2a227f70e8d`.
  The repository has no license file.
- **Failing-to-Falsify**, https://github.com/ayushjhaveri/llm-confirmation-bias at commit
  `97f64338f88e75cdf3f5572ee6cd899f8b95c2b9`. The repository has no license file.

The models that `transfer/` downloads or calls (Qwen/Qwen3.8-27B, Qwen/Qwen2.5-Coder-32B-Instruct and
the API judges `google/gemini-2.5-flash` and `meta-llama/llama-3.3-70b-instruct`) are used under their
own licenses and terms.

PhysGym's license:

```text
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
