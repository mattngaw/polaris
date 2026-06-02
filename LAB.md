# LAB.md — The Polaris Lab: MoE vs. Dense on gfx803

> A from-scratch experiment harness to study Mixture-of-Experts against dense
> feed-forward layers in a small GPT-style transformer, running on two consumer
> AMD RX 480 (gfx803) cards. The scaffolding gives you prose contracts and
> `# TODO(human):` markers; the model, MoE, training loop, and instrumentation
> are yours to write. This is a scaffold I'm using on myself; there is
> nothing to hand in. The value is in getting something working and then
> iterating on it.

---

## 1. Introduction

You are building a single transformer that can run its feed-forward sub-layer in
two modes — a **dense MLP** or a **Mixture-of-Experts** layer — selected by one
config field, `ffn_type: dense | moe`. Both modes go through the same block, the
same optimizer, the same schedule, the same data path. A paired run differs in
*only* the FFN. That constraint is the whole experiment: it is what makes a
dense-vs-MoE comparison mean something.

The headline question, scoped down from "Slicing and Dicing: Configuring Optimal
Mixtures of Experts," is: **hold active parameters per token roughly fixed, sweep
total parameters by varying expert count and granularity, and watch two things** —
(a) whether validation loss improves as total parameters grow, and (b) the systems
costs the literature tends to understate: the FLOPs-vs-params decoupling, the
naive-dispatch wall-clock penalty, and per-expert load imbalance.

The systems angle is the point. At this scale (1–30M params on 4 GB cards), small
validation-loss differences between paired runs will sit inside the noise floor;
load imbalance and the dispatch penalty will not. `experiments/README.md` is
explicit that **the systems story is the primary result, not the loss curves.**
It's a systems-measurement project at heart: the interesting result is in the
measurement, not the headline metric. The mechanism is legible at this scale — a
router, a pool
of experts, a dispatch step, and a load-balancing loss are each small enough to
write by hand and instrument directly.

---

## 2. Logistics

**Hardware.** 2× AMD Radeon RX 480, 4 GB VRAM each, gfx803 (Polaris — the project
is named for the architecture, not the star). No NVLink-equivalent, no working
collectives.

**Compute environment.** All PyTorch work runs *inside* the
`robertrosenbusch/rocm6_gfx803_comfyui:5.7` container (ROCm 5.7, PyTorch 2.3.0a0,
HIP 5.7, Python 3.10 in a conda env at `/opt/conda/envs/py_3.10/`). The host has
no ROCm and no project-local torch. The image is large (~38 GB extracted) and
ships ComfyUI, which is dormant — the entrypoint is overridden with `bash`. The
`env/run-container.sh` wrapper handles create / start / exec / stop; read
`env/container-run.md` for the manual `docker run` equivalent and the rationale
behind each flag.

**Why this specific image.** Path-validation (2026-05-20) found that every ROCm
6.x image tested produces structured-wrong fp32 GEMM on gfx803. ROCm 5.7 is the
community-validated path where fp32 is correct. This is not optional trivia; it is
the reason the stack is pinned. See Section 9 and `KNOWN_GOTCHAS.md`.

**Two-card pattern.** No multi-GPU training. RCCL on gfx803 is unreliable, so there
is no DDP. Instead you run **one independent process per card**, selecting the card
with `HIP_VISIBLE_DEVICES=0` or `=1` *inside* the container (never at container
launch — setting it at launch masks a card). Open two host shells, run
`./env/run-container.sh` in each (the second `docker exec`s into the same
persistent container), and launch one run per card. This maps cleanly onto the
sweep: each (dense, moe) pair runs together, one per card, sharing nothing but the
dataset on disk.

**Expected effort.** This is open-ended. Expect the bulk of the time to go into
(a) getting a correct, instrumented training loop running on Shakespeare, and (b)
restructuring the MoE dispatch so it survives `n_experts ≥ 8` (see Section 7 and
the gotchas index). The model and dense FFN are the easy part.

---

## 3. Required Artifacts

Each artifact below has a prose contract in its stub docstring. The contracts are
authoritative; what follows is a summary of *what each must do*, not how.

**Config (`polaris/config.py`).** One schema for both arms. `ffn_type` is a
first-class field. The training loop branches on nothing except `ffn_type`, and
even that hides behind the FFN interface. fp32 and no-compile are defaults that
cannot be silently overridden. Include a `validate()` that rejects incoherent
combinations (e.g. `ffn_type == "dense"` with `n_experts > 1`). Fields cover model,
ffn, optim, schedule, train, and data groups — the docstring enumerates them.

**Model (`polaris/model.py`).** A GPT-style transformer whose block calls its
feed-forward sub-layer through a **common FFN interface** (an `nn.Module` taking
`(B, T, C)` and returning `(B, T, C)`, plus an auxiliary channel for MoE signals —
you pick the discipline and apply it to both arms). Write the causal self-attention,
the norm choice, and the residual wiring yourself. Token/position embeddings and
the LM head are fair game to borrow from nanoGPT (Section 4).

**Dense FFN (`polaris/ffn/dense.py`).** The baseline arm: `Linear(C, H)` →
activation → `Linear(H, C)`. Match the active-parameter budget of the MoE arm you
intend to compare it against. Return `aux=None` (or your interface's equivalent) so
the training loop stays uniform.

**MoE FFN (`polaris/ffn/moe.py`).** The focal artifact. A router (`Linear(C,
n_experts)` → softmax → top-k), a pool of `n_experts` dense FFNs at granularity
`ffn_hidden_per_expert`, a **naive dense-masked dispatch/combine** (no block-sparse
kernels exist on gfx803 — every expert sees a `(B*T, C)` input under a selection
mask, which is intentionally inefficient and is the thing you are measuring), the
Switch/GShard load-balancing aux loss exposed via the aux channel, and a per-expert
token-count hook that `instrumentation.py` reads. Read the stub's heads-up comments
before writing: the `n_experts = 8` backward crash and the aux-weight tuning issue
are load-bearing.

**Training loop (`polaris/train.py`).** Invoked as `python -m polaris.train
--config <path>`, one process per card. Loads config, builds the model via the
FFN-type dispatch, samples batches from memory-mapped `.bin` files, runs AdamW with
cosine-warmup LR, adds the aux loss when present, clips gradients, evaluates at
intervals, logs metrics, and checkpoints. The `get_batch` loader, LR schedule,
checkpoint save/load, and eval loop are borrowed from nanoGPT (Section 4); the
aux-loss combination, the per-expert load hook, and the step/dispatch timing
breakdown are yours.

**Instrumentation (`polaris/instrumentation.py`).** Per-expert load recorder
(counts per (layer, expert), accumulated over training), a `flops_per_token(config)`
accountant returning **(active, total)** FLOPs/token, a step/dispatch timer built on
`torch.cuda.Event(enable_timing=True)` (confirmed working on HIP), and a metrics
writer (CSV/JSONL under `runs/<run_id>/`) with at least iter, train/val/aux loss,
step_ms, dispatch_ms, and per-expert counts. Must be cheap, and a no-op when
`ffn_type == "dense"`.

**Data prep (`data/prepare_shakespeare.py`, `data/prepare_tinystories.py`).**
Download, tokenize, split, and write `train.bin` / `val.bin` / `meta.pkl` under
`data/<name>/` in the nanoGPT flat-token convention (uint16, `np.memmap`-friendly).
Shakespeare is char-level; TinyStories uses a small custom BPE (~4–8K vocab, *not*
the 50K GPT-2 vocab — a big embedding table would soak up the parameter budget that
needs to land in the experts). Both are stubs now. See `data/README.md`.

**Sweep configs and the write-up.** Config files live in `experiments/configs/`,
one per run, human-readable and diff-able (suggested:
`<dataset>_<ffn>_<size>_<seed>.yaml`). The sweep driver that schedules pairs across
the two cards is intentionally not scaffolded — it is yours. The write-up is the
thing you are building toward; Section 6 describes what a complete one shows.

---

## 4. Programming Rules

These are the constraints that keep you on the validated stack and keep the
comparison honest. Tables first, because they are crisper than prose.

**Precision / dtype rules:**

| Action | Status | Why |
| --- | --- | --- |
| Train in **fp32** | OK | Default; verified correct on ROCm 5.7 by `env/check-matmul.py`. |
| Train in bf16 / fp16 | OK *if verified* | Both pass the diagnostic on this stack, but treat as a non-default study axis, not a convenience. gfx803 bf16 may be software-emulated (community issue #53); long-run behavior is unpressured. |
| Add AMP / autocast scaffolding "to be safe" | Not OK | No reason to on 5.7, where fp32 works. Only add if you make precision an explicit axis and verify it first. |
| Assume fp32 is broken (the old framing) | Not OK | That was a ROCm 6.x bug. On 5.7, fp32 is the correct default. |

**Borrowing rules (nanoGPT, MIT):**

| Piece | Status | Note |
| --- | --- | --- |
| `get_batch` / memmap sampling | OK to borrow | Mark origin; log in `CREDITS.md`. |
| Cosine-with-warmup LR schedule | OK to borrow | Same. |
| Checkpoint save/load, eval loop | OK to borrow | Same. |
| `generate()` for `sample.py` | OK to borrow | Same. |
| Token/position embeddings, LM head | OK to borrow | Pure plumbing; mark in `CREDITS.md`. |
| Attention, block wiring, norm choice | Not OK to borrow | Intellectually load-bearing — write by hand. |
| The MoE layer, router, dispatch, aux loss | Not OK to borrow | The point of the project. |
| The instrumentation | Not OK to borrow | nanoGPT has none; it is the primary result. |
| Cloning / vendoring / pasting any nanoGPT file | Not OK | Borrow specific functions by hand, with the origin comment, logged in `CREDITS.md`. |

Borrowed functions carry the comment `# adapted from nanoGPT (karpathy/nanoGPT,
MIT)` and a row in `CREDITS.md`.

**Other rules:**

- **No `torch.compile`.** No inductor/triton backend for gfx803. `compile = False`
  is a config default; do not flip it.
- **Never install torch.** The container's torch is the only build that works on
  this hardware. `pyproject.toml` lists pure-Python deps only; a `pip install torch`
  would replace the working build and break everything.
- **4 GB VRAM budget.** Every byte matters. The naive masked MoE multiplies
  activation memory by `n_experts` (every expert receives the full `(B*T, C)`
  tensor). Path-validation ran comfortably under budget only at toy shapes
  (`B=32, T=64, C=128`, `ffn_hidden=512` dense / `128 per-expert × 4` MoE); real
  model shapes will press the limit. Memory has not been stress-tested — budget
  deliberately.
- **The diagnostic must pass first (non-negotiable).** Before any training, run
  `python env/check-matmul.py --dtype fp32` inside the container and confirm exit 0
  / "ALL OK". If it fails, you have drifted off the validated stack — stop and
  diagnose, do not train. Wire the same check into `train.py` as an early bail so a
  future image drift cannot silently pin your loss curves at `ln(vocab_size)`.

---

## 5. Driver Programs

**`env/check-matmul.py` — the stack oracle.** Run it inside the container, before
anything else, every time you change image / ROCm / torch versions:

```
python env/check-matmul.py --dtype fp32      # expect "ALL OK", exit 0
```

It runs four GEMM correctness checks — three against a CPU reference (`F.linear`,
`nn.Linear`, a transposed-view `a @ W.T`) plus a `ones @ ones.T` with a hand-known
exact answer of 128.0. Exit 0 means GEMM is correct in the chosen dtype and you can
train in it; exit 1 means it is broken and points you at `KNOWN_GOTCHAS.md`. **This
must pass in fp32 before any training run** — see Section 4. You can also check
`--dtype bf16` / `fp16` / `fp64` if you ever explore precision as an axis.

**First-run sanity sequence (inside the container).** From `env/container-run.md`:

```
rocminfo | grep -c 'Name: *gfx803'                          # expect 2
python -c "import torch; print(torch.cuda.device_count())"  # expect 2
python env/check-matmul.py --dtype fp32                     # expect "ALL OK"
pip install -e .                                            # deps + the polaris package
```

If `device_count()` is 1, you set `HIP_VISIBLE_DEVICES` at launch (don't) or a
`/dev/dri/renderD12*` node is missing. If the matmul check fails, you may have
landed on a ROCm 6.x image by accident.

**Smoke pattern — the one authoritative definition.** Before trusting any sweep
numbers, reproduce the path-validation smoke: a small transformer + AdamW + cosine
LR + checkpoint, a few hundred iters on **char-level Shakespeare**, both arms.
You are looking for monotonic loss decrease, no NaN, and a coherent-ish sample —
confirmation that the loop *runs*, not science. Do this on Shakespeare, not
TinyStories, and do not read into the Shakespeare loss curves. (Stage 3 in
Section 8 and the Section 6 checklist both refer back to this.)

---

## 6. Done-When Checklist

This is how you know you actually built and measured the machine, not just got
something to run. You're done with a line when you can show yourself the evidence —
each box is self-verifiable — you confirm it for yourself:

- [ ] A working transformer with a swappable FFN, selectable by `ffn_type` alone,
      with paired (dense, moe) runs that differ *only* in the FFN.
- [ ] `env/check-matmul.py` passing in fp32 on your stack. Keep the torch/HIP
      version line it prints in your run notes, so future-you can tell instantly
      if the stack has drifted.
- [ ] An end-to-end Shakespeare smoke for both arms: loss decreases, no NaN, a
      sample is produced.
- [ ] At least one MoE training run that survives to a useful iteration count,
      with the dispatch restructured if you went past `n_experts = 4`.
- [ ] A **FLOPs-per-token accountant** reporting active vs. total, and a short
      explanation of the gap for your configs.
- [ ] **Per-expert load plots** over training — the distribution across experts
      and how it evolves. Note the aux-loss weight you used and whether routing
      collapsed or balanced.
- [ ] A **step-time / dispatch-time breakdown** comparing MoE against the
      matched-active-params dense baseline — the naive-dispatch penalty, measured.
- [ ] The sweep results you actually collected: for each (dataset, ffn_type,
      total-params) point, final val loss/perplexity plus the systems metrics.
- [ ] An honest account of what the loss curves do and do not show at this scale,
      and what the systems measurements show (which is where the legible result
      lives).
- [ ] `CREDITS.md` filled in for everything borrowed.

Where loss differences are inside the noise, say so. The systems numbers are the
result; the loss curves are context.

**This checklist is the first complete loop, not the finish line.** Once it's all
checked you have a working machine to improve. Pick the weakest measurement — the
dispatch penalty, the routing balance, the FLOPs gap — and iterate on it. The point
of the project is the second and third pass; the first one just gets you a machine
worth iterating on (see the closing note in Section 8).

---

## 7. Useful Tips

Debugging recipes calibrated to this project's actual failure modes (all of these
were observed during path-validation — see `KNOWN_GOTCHAS.md`):

- **Loss pinned at `ln(vocab_size)` forever.** This is the fp32 GEMM signature, not
  a learning-rate problem. Adam happily shuffles params around the uniform-logits
  attractor. Run `env/check-matmul.py --dtype fp32`; if it fails you are on a
  ROCm 6.x image. Do not tune hyperparameters against this — fix the stack.

- **MoE backward crashes around `n_experts = 8`.** The naive boolean-mask-per-expert
  pattern (`flat[mask]` gather → expert → `out[mask] = ...`) fails in the *backward*
  pass on this stack — either a "Memory access fault by GPU node" (B=32) or a "numel:
  integer multiplication overflow" (B=16). It reproduces with `aux_loss_weight = 0`,
  so the dispatch graph is the cause, not the aux loss. Forward and routing are
  healthy at 8; `n_experts = 4` is fine end-to-end; the crash is reliably in
  backward. The autograd graph from many chained boolean-mask gather/scatter ops is
  the suspect — restructuring the dispatch to shrink that graph is the work the
  headline sweep depends on, and figuring out *how* is yours to do.

- **Router collapses (one expert eats everything; aux loss climbs).** At this scale,
  the Switch paper's default `aux_loss_weight = 0.01` was too weak — full collapse
  within ~100 iters at `n_experts = 4` on a ~1 MB dataset. An order of magnitude
  larger restored balanced routing at `n_experts = 4`; you will have to find your
  own working value. Expect to tune the aux weight per `(n_experts, dataset_size)`,
  or add a capacity-factor + token-drop policy that constrains balance independently
  of the soft aux.

- **First-iter step time is a ~0.5 s outlier; measure with HIP events.** HIP kernels
  compile/autotune on first use, so steady state arrives within a few iters —
  exclude the first one or two iters from any reported step-time statistic, or your
  average is dominated by warmup. Time with `torch.cuda.Event(enable_timing=True)`
  (HIP events under the hood, confirmed working), not `time.time()`, and keep the
  timer no-op on the dense arm so your numbers describe the model, not the
  instrumentation.

- **If you add DataLoader workers.** The path-validated loader was plain `np.memmap`
  + random offsets, no workers. Going to `num_workers > 0` re-enters the untested
  `--ipc host` + `--shm-size 16G` interaction zone — verify it deliberately.

When stuck, arrive at a rubber-duck session (LLM or otherwise) with the exact
`env/check-matmul.py` output, the `torch.__version__` / `hip` version line, the
config and full traceback (especially whether the failure is forward or backward),
and the `n_experts` / `B` / dtype involved — most sharp edges here are
stack-and-shape specific, and that context is what makes them diagnosable.

---

## 8. Strategic Advice

Build in this order. Each stage gives you a checkpoint where something works before
you add the next source of complexity.

1. **Config.** Define the schema with `ffn_type` first-class and fp32 / no-compile
   locked as defaults. *Outcome:* a config you can load and validate; incoherent
   combos fail loud.

2. **Model + dense FFN.** Write the transformer, the block, the attention, and the
   dense FFN behind the common interface. *Outcome:* a model you can instantiate and
   forward on random data without touching MoE.

3. **Shakespeare smoke.** Run the smoke defined in Section 5 on the dense arm —
   borrow `get_batch`, the LR schedule, checkpointing, and the eval loop.
   *Outcome:* loss decreases monotonically, no NaN, a sample prints. The loop is
   proven; the science has not started.

4. **MoE.** Add the router, expert pool, naive masked dispatch, and aux loss. Keep
   `n_experts = 4` and `top_k = 1` first. *Outcome:* MoE trains end-to-end at
   `n_experts ≤ 4`; you will likely see a wall-clock penalty over the matched dense
   baseline (path-validation saw step time ~2.4× dense at `n_experts = 4` — a
   ballpark that confirms the timing is wired right, not a target), and you may see
   router collapse if the aux weight is too low — tune it.

5. **Instrumentation.** Wire per-expert load, the FLOPs accountant, and the
   step/dispatch timer; write metrics to disk. *Outcome:* the plots that are the
   actual result. This is where the project pays off.

6. **TinyStories prep.** Implement the BPE prep and build the `.bin` files. *Outcome:*
   a dataset with enough structure for routers to specialize on — named characters,
   dialogue vs. narration — which Shakespeare lacks.

7. **The sweep.** Write the two-card driver, then sweep expert count and granularity
   at matched active params, dense baseline alongside. *Outcome:* the headline data.
   Expect loss differences inside the noise floor and a legible systems story — the
   FLOPs gap, the dispatch penalty, and load imbalance evolving over training. The
   open question is what the penalty *tracks*: active-FLOPs, total-FLOPs, or
   something in between — that's for your timer and accountant to answer, not for
   this doc to assert. To sweep past `n_experts = 4` you must have restructured the
   dispatch (Stage 4 / Section 7) first.

Do not jump to the sweep before instrumentation works — a sweep without trustworthy
measurements produces numbers you cannot defend. And the staircase is a re-entry
point, not a one-way trip: once the sweep runs, the natural next move is to go back
and improve the weakest stage — restructure the dispatch further, retune the aux
weight, tighten the instrumentation — rather than stopping at the first set of
numbers.

---

## 9. Appendix A: The Validated Stack

The combination confirmed working on this hardware (`KNOWN_GOTCHAS.md`,
`env/container-run.md`):

| Component | Value |
| --- | --- |
| GPUs | 2× AMD RX 480, 4 GB each, gfx803 (Polaris) |
| Image | `robertrosenbusch/rocm6_gfx803_comfyui:5.7` |
| ROCm / HIP | 5.7 |
| PyTorch | 2.3.0a0 |
| Python | 3.10 (conda env `/opt/conda/envs/py_3.10/`) |
| fp32 GEMM | correct (verified by `env/check-matmul.py`) |
| bf16 / fp16 / fp64 GEMM | correct (bf16 possibly emulated; unpressured) |
| Required env at launch | `HSA_OVERRIDE_GFX_VERSION=8.0.3`, `PYTORCH_ROCM_ARCH=gfx803` |
| Card selection | `HIP_VISIBLE_DEVICES=0|1`, set per-process *inside* the container |
| `torch.compile` | unavailable (no gfx803 inductor backend) |
| Multi-GPU / RCCL / DDP | not used (unreliable on gfx803) |

**Why 5.7 and not 6.x — the detail Section 2 omits.** On every ROCm 6.x image tested
(6.1.2 and 6.4.3), fp32 GEMM on gfx803 produced structured-wrong output for
rectangular / transposed-operand matmuls. The two 6.x images gave **bit-identical**
broken output (`max_diff` agreeing to 7 sig figs), indicating one shared upstream
kernel — a ROCm-6.x regression specific to gfx803, not a per-build quirk.
bf16/fp16/fp64 were correct even on the broken images. The documented
`F.linear((32,128),(65,128))` test showed a max diff of 56.8 vs. CPU on 6.x and
passed cleanly on 5.7; the `ones @ ones.T` test (exact answer 128.0) came back at a
GPU mean of 126.0 on 6.x — about 1–2% of entries silently wrong — and exact on 5.7.
The bug is filed but unanswered upstream (`robertrosenbusch/gfx803_rocm` #55). No
env-var workaround fixed it; the community-converged fix is to drop to 5.7. To
verify your own stack, run the oracle from Section 5.

---

## 10. Appendix B: Known Gotchas Index

Full detail and the cross-image verification matrix live in `KNOWN_GOTCHAS.md` —
that file is the source of truth. The **HEADLINE** section covers the fp32 GEMM bug
(loss stuck at `ln(V)`, the cross-image matrix, the env-var toggles that did not
fix it, and upstream issues #55 / #43 / #53); the **Secondary** section covers
everything path-validated on ROCm 5.7 (`n_experts ≥ 8` backward crash, aux-weight
tuning at small scale, the ~2.4× naive-dispatch penalty at `n_experts = 4`, the
first-iter step-time outlier, `torch.cuda.Event` working on HIP, both BPE libs
installing cleanly, the `np.memmap` loader, and the un-stress-tested memory
budget). Section 7 already carries the working fingerprints and directions for the
ones you will actually hit.
