<p align="center">
  <img src="assets/banner.png" alt="polaris" width="80%">
</p>

<h3 align="center">Polaris</h3>

<p align="center">
  <strong>Tiny LLM pretraining on dual RX 480s.</strong><br/>
  Inspired by <a href="https://github.com/karpathy/nanoGPT"><code>karpathy/nanogpt</code></a>.
</p>

> [!CAUTION]
> This repo is a work in progress! Feel free to check in on it frequently to see updates.

A from-scratch experiment harness to study **Mixture-of-Experts (MoE) vs.
dense feed-forward layers** in a small GPT-style transformer, on consumer
hardware. Named after the GCN4 "Polaris" GPU architecture this runs on
(RX 480, gfx803).

> **New here / picking this back up?** Read [`LAB.md`](LAB.md) — the
> guide to what you're building, the rules, the build order, and the
> failure modes. It's the entry point.

## What this is — and isn't

**Is:** a learning project. The headline experiment holds *active* parameters
per token roughly fixed and sweeps total parameters by varying the number of
experts and their granularity, watching both
- validation loss vs. total parameters, and
- the systems cost the literature understates — FLOPs-vs-params decoupling,
  the naive-dispatch penalty, and per-expert load imbalance.

A **dense FFN sweep runs alongside as the comparison baseline.** Dense is a
first-class arm of the study.

**Isn't:** a reproduction of "Slicing and Dicing: Configuring Optimal
Mixtures of Experts." It's inspired by that paper, scoped *way* down to fit
on this hardware. The goal is to understand the mechanism and measure the
systems behavior, not to chase the paper's numbers.

## Hardware & environment

- 2× AMD Radeon RX 480, 4 GB VRAM each, **gfx803 (Polaris)**.
- All PyTorch work runs **inside** the
  `robertrosenbusch/rocm6_gfx803_comfyui:5.7` container
  (ROCm 5.7 + PyTorch 2.3.0a0 + Python 3.10 in a conda env). See
  `env/container-run.md` for why this specific image — short version:
  ROCm 6.x has a gfx803 fp32 GEMM bug that 5.7 doesn't.
- **fp32 is the configured default** and verified-correct on this stack
  by `env/check-matmul.py`. bf16/fp16 also work if you ever want to add
  precision as a study axis.
- **No `torch.compile`** — no gfx803 backend in inductor.
- **Two-card strategy:** one independent run per card via
  `HIP_VISIBLE_DEVICES=0` / `=1`. No RCCL, no DDP.

torch is **never** installed via this repo's manifest — it is whatever the
container ships. See `env/container-run.md` for the launch recipe and
`env/run-container.sh` for the host-side wrapper.

## Layout

- `polaris/` — the package. `model.py`, `config.py`, `ffn/{dense,moe}.py`,
  `train.py`, `sample.py`, `instrumentation.py`. All stubs right now.
- `data/` — dataset preparation (TinyStories primary, Shakespeare smoke
  test). Prepared `.bin` files are gitignored.
- `experiments/` — config files and the sweep design.
- `env/` — container launch docs + host-side wrapper script.
- `tests/` — smoke tests (none yet).

## How to run

```
./env/run-container.sh           # start (or attach to) the persistent
                                 # container "polaris-dev"; drops you in a
                                 # shell at /workspace.
# Inside the container, the first time:
pip install -e .                 # installs pure-Python deps + makes the
                                 # `polaris` package importable from anywhere.
# Day to day:
HIP_VISIBLE_DEVICES=0 python -m polaris.train --config experiments/configs/<x>.yaml
```

For two concurrent runs (one per card), open a second host shell and run
`./env/run-container.sh` again — it `docker exec`s into the same container.

## Dependency manifest choice

This project uses `pyproject.toml` (not `requirements.txt`): `pip install
-e .` makes `polaris` importable from anywhere inside the container, which
keeps `python -m polaris.train` clean from any working directory. Pure-
Python deps only — see the torch caveat under *Hardware & environment*.

## Borrowing convention

Some plumbing is adapted from **karpathy/nanoGPT** (MIT) by hand —
`get_batch`-style data loading, the training-loop skeleton, the cosine LR
schedule, checkpoint save/load, the sampler. Each borrowed function carries
a comment like:

    # adapted from nanoGPT (karpathy/nanoGPT, MIT)

Log it in `CREDITS.md` when you borrow. Everything else — the model, the
MoE, the instrumentation — is hand-written on purpose.

## Before you write model code

**Read `KNOWN_GOTCHAS.md` first**, then confirm you're on the validated
stack:

```
./env/run-container.sh
python env/check-matmul.py --dtype fp32       # expect ALL OK
```

If that diagnostic ever fails in fp32, you've drifted off the stack —
stop and check `KNOWN_GOTCHAS.md` before training. (Why it matters is in
*Hardware & environment* above.)

## See also

- `KNOWN_GOTCHAS.md` — toolchain surprises to read before writing model
  code.
- `CREDITS.md` — attribution table.
- `env/container-run.md` — full container launch details, two-card pattern,
  and gotchas.
- `data/README.md` — dataset choices and the expected on-disk layout.
- `experiments/README.md` — the sweep design.

