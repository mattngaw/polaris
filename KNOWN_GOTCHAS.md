# Known gotchas

Surprises uncovered while validating the toolchain.

---

## HEADLINE — ROCm 6.x has an fp32 GEMM regression on gfx803; we use ROCm 5.7

**Discovered:** 2026-05-20 during path-validation smoke runs.
**Resolved:** by switching the project image to
`robertrosenbusch/rocm6_gfx803_comfyui:5.7`. fp32 GEMM on this image is
correct on gfx803, confirmed end-to-end via `env/check-matmul.py`.

### The bug (kept here because the diagnostic still matters)

On every **ROCm 6.x** image we tested (6.1.2 and 6.4.3), fp32 GEMM on
gfx803 produces structured-wrong output for rectangular matmul shapes —
specifically `F.linear`, `nn.Linear`, or any `a @ W.T` with a transposed
non-contiguous operand. Output entries collapse toward a few constants;
in a language model this pins loss at exactly `ln(vocab_size)` and the
network never learns. **bf16, fp16, fp64** all work correctly even on
the broken images.

On **ROCm 5.7** all three dtypes are correct. The bug is therefore a
ROCm-6.x regression specific to gfx803, not a property of any individual
build or our specific image. Two different ROCm 6.x images we tested
gave **bit-identical** broken output (`max_diff` agrees to 7 sig figs)
for the same matmul — same upstream kernel.

### Diagnostic — always run before training

`env/check-matmul.py` is the canonical "is your stack ok?" oracle.
Exit 0 means GEMM is correct in the chosen dtype on your stack. Run it
any time you change image/ROCm/torch versions.

```
python env/check-matmul.py --dtype fp32
python env/check-matmul.py --dtype bf16
python env/check-matmul.py --dtype fp16
```

### What was tested

Same shape, `F.linear(x:(32,128), W:(65,128))`, CPU vs GPU max diff:

| dtype  | max diff vs CPU | verdict |
| ------ | ---------------- | ------- |
| fp32   | **56.8**         | **BROKEN** |
| bf16   | ~0               | OK |
| fp16   | ~1e-4 (noise)    | OK |
| fp64   | ~4e-14           | OK |

Direct evidence:
- `ones() @ ones().T` in fp32: GPU mean = 126.0 (expected 128.0). About 1-2%
  of entries are silently wrong. In fp16/bf16 the same test is exact.
- An end-to-end bf16 training of a small transformer block (causal attn +
  LayerNorm + GELU FFN + residuals) trained for 30 iters with loss
  decreasing monotonically and no NaN.
- An end-to-end fp32 training of the same shape sits at exactly `ln(V)`
  forever — Adam happily moves params around the "uniform logits"
  attractor.

### Cross-image verification matrix (this hardware, 2026-05-20)

The diagnostic was run against three images on the same RX 480, same host:

| Image | torch | ROCm | fp32 | bf16 | fp16 |
| --- | --- | --- | --- | --- | --- |
| `woodrex/rocm612-torch24-gfx803` (current project image) | 2.4.0a0 | 6.1.2 | **BAD** (Δ=56.82) | OK | OK |
| `robertrosenbusch/rocm6_gfx803_pytorch:6.4.3_0.2.8` | 2.8.0 | 6.4.3 | **BAD** (Δ=56.82) | OK | OK |
| `robertrosenbusch/rocm6_gfx803_comfyui:5.7` | 2.3.0a0 | 5.7 | **OK** | OK | OK |

Observations:
1. The 6.1.2 and 6.4.3 images give **bit-identical broken output** for fp32
   (`max_diff` agrees to 7 sig figs). Same upstream broken kernel, two
   different builds.
2. ROCm 5.7 is correct in fp32. The bug is therefore a **ROCm 6.x
   regression specific to gfx803**, not a problem with any individual
   build.
3. The project runs on the 5.7 image
   (`robertrosenbusch/rocm6_gfx803_comfyui:5.7`) and trains in fp32, which
   is correct on this stack.

### Upstream context (community has hit this, no fix in this image)

This bug is **filed but unanswered** on the related community repo:

- [`robertrosenbusch/gfx803_rocm` #55 — "nn.Linear returns incorrect
  numbers"](https://github.com/robertrosenbusch/gfx803_rocm/issues/55).
  Same RX 580 hardware, same `nn.Linear` symptom (CPU returns `[3.3, 5.5]`,
  GPU returns `[5.5, 5.5]`). Zero comments. Same bug, different image.

- [`robertrosenbusch/gfx803_rocm` #43 — "Artifacts/corruption on RX 570
  with ComfyUI"](https://github.com/robertrosenbusch/gfx803_rocm/issues/43).
  Long thread: multiple users report image corruption from ComfyUI on
  ROCm ≥ 6.x for gfx803. **Established community workaround: drop to
  ROCm 5.7.** Specifically `robertrosenbusch/rocm6_gfx803_comfyui:5.7`
  is the pre-built image users confirm works (model-agnostic — same
  corruption with multiple SD models on 6.x, none on 5.7).

- [`robertrosenbusch/gfx803_rocm` #53 — "unsupport bf16 on
  rx580"](https://github.com/robertrosenbusch/gfx803_rocm/issues/53).
  Community claim that gfx803 lacks native bf16 support — "the hardware
  lacks efficient implementation." This contradicts the in-isolation
  bf16 GEMM correctness we measured below; the reconciliation is
  probably that PyTorch's bf16 path on gfx803 is software-emulated and
  *can* produce correct results but its long-run / large-shape behavior
  hasn't been pressure-tested by us.

**Filing upstream** on `woodrex83/ROCm-For-RX580` referencing #55 is
worth doing if you confirm the bug on your machine via
`env/check-matmul.py` — that repo's maintainer has not seen this report
yet.

### What we ended up doing

Switched the project image from `woodrex/rocm612-torch24-gfx803`
(ROCm 6.1.2, fp32 broken) to `robertrosenbusch/rocm6_gfx803_comfyui:5.7`
(ROCm 5.7, fp32 correct). The container recipe in
`env/container-run.md` and the host-side wrapper at
`env/run-container.sh` both target the 5.7 image now.

Costs of the switch: older PyTorch (2.3 vs 2.4), Python is now in a
conda env at `/opt/conda/envs/py_3.10/`, image is ~38 GB extracted, and
ComfyUI is bundled (dormant — we override the entrypoint).

Benefit: all three dtypes work correctly on this hardware, so the
project is not constrained to bf16/fp16, and the fp32 default holds up
empirically.

### About bf16/fp16 (still useful information)

Even on the broken ROCm 6.x images, bf16 and fp16 GEMM are correct in
isolated tests, and a small bf16 transformer-block forward/backward
trained without NaN. So if you ever end up on a ROCm 6.x image, bf16 is
a viable fallback — but treat it as a fallback, not a primary path, due
to the community claim (issue #53 above) that gfx803 has no native bf16
support and these operations may be software-emulated.

### A note on what we couldn't reproduce

When I first hit the bug I described it broadly as "matmul is broken." That
was wrong — fp32 is, the other dtypes aren't. The original framing also
suggested "no env-var workaround"; the actual workaround at the *image*
level is to drop to ROCm 5.7, which the community has converged on.

### What was tried (in case the bug ever reappears)

Env-var toggles that did **not** fix the fp32 case:
- `DISABLE_ADDMM_HIP_LT=1`
- `ROCBLAS_USE_HIPBLASLT=0`
- `TORCH_BLAS_PREFER_HIPBLASLT=0`
- `TORCH_USE_HIPBLASLT=0`
- `PYTORCH_TUNABLEOP_ENABLED=0`
- `MIOPEN_FIND_MODE=NORMAL`
- `torch.use_deterministic_algorithms(True)`
- Materializing transposes via `.contiguous()` before matmul
- Storing W as `(K, N)` and doing `x @ W` instead of `x @ W.T`

If a bf16 bug ever appears under specific patterns, start with the env vars
above before assuming the workaround stops working.

---

## Secondary

### Both BPE tokenizer libraries install cleanly

Inside the container, both `pip install sentencepiece` (0.2.1) and
`pip install tokenizers` (HF) install and import without error. The
`# TODO(human): choose` in `data/prepare_tinystories.py` and
`pyproject.toml` is still open, but pick on API ergonomics — install is
not a discriminator.

### `torch.cuda.Event(enable_timing=True)` works on ROCm/HIP

Confirmed: `cuda.Event` start/stop/elapsed_time returns sensible
milliseconds on gfx803, mapped to HIP events under the hood. Use it for
per-step / per-dispatch timing in `polaris/instrumentation.py`.

### Stable Diffusion as reference

The same container image runs Stable Diffusion WebUI (`SD/` subdir of
`~/ROCm-For-RX580/`). SD runs in fp16. If anything in the polaris stack
ever seems hardware-unreliable, comparing against a known-working SD config
on the same hardware is a useful sanity anchor.

### Path-validated on ROCm 5.7 (dense + MoE smoke on Shakespeare)

After the image switch, a throwaway end-to-end smoke ran in scratch:
small transformer + AdamW + cosine LR + checkpoint, 400 iters on
char-level Shakespeare. Both dense and MoE arms completed; observations:

- **fp32 training works end-to-end.** Both arms had monotonic loss
  decrease and no NaN. AdamW state in fp32 is fine.
- **Naive masked-MoE dispatch's wall-clock cost at small scale.** With
  `n_experts=4`, `top_k=1`, and the project's documented naive masked
  dispatch (every expert sees `(B*T, C)` masked by a per-expert
  selection), step time was ~2.4× the matched-dense baseline. This is
  the "naive-dispatch penalty" the project is set up to measure; the
  ratio will grow with `n_experts`.
- **Aux loss is numerically stable in fp32** at this scale; no NaN, no
  blowup. Behaves as theory predicts (hovers near the balanced
  equilibrium when balance is maintained).
- **First-iter step time is an outlier** (~0.5 s) on both cards from
  HIP kernel compile/cache warmup; steady state hits within a few
  iters. Exclude the first iter or two from any reported step-time
  statistics. Worth wiring into `polaris/instrumentation.py`.
- **`np.memmap` + a `get_batch` that just samples random offsets and
  stacks `int64` slices works fine** under the container's default
  settings (no DataLoader workers in the smoke, so the `--ipc host` +
  `--shm-size 16G` interaction is still untested — flag it once you add
  workers).
- **Memory was not stress-tested.** Smoke ran at `B=32, T=64, C=128,
  ffn_hidden=512 (dense) / 128 per-expert × 4 (MoE)` — comfortably
  under 4 GB. Real model shapes (more layers, wider C, longer T) will
  hit the budget; the naive-masked MoE multiplies activation memory by
  `n_experts` since every expert receives the full `(B*T, C)` tensor.

### Heads-up — n_experts ≥ 8 crashes the naive masked dispatch

The naive masked-dispatch MoE pattern documented in `polaris/ffn/moe.py`
(boolean mask per expert; gather → expert forward → scatter) fails at
`n_experts = 8` on this stack (gfx803 + ROCm 5.7 + PyTorch 2.3). The
failure is in the **backward** pass and reproduces across batch sizes
(B=16 and B=32) and with the aux loss disabled (`aux_loss_weight = 0`),
so the dispatch+gather+scatter pattern in backward is the proximate
cause — not the aux loss or memory pressure.

Symptoms observed:
- B=32: `Memory access fault by GPU node-1 ... Page not present or
  supervisor privilege` during backward, somewhere between iter 0 and
  iter 100.
- B=16: `RuntimeError: numel: integer multiplication overflow` during
  `total.backward()`, same training stage.

Forward and routing at n=8 look healthy (the iter-0 eval printed
sensible per-expert fractions, no NaN). The crash is reliably in
backward.

**Constraint for the project:** with the naive dispatch as written, the
expert-count sweep is bounded by `n_experts ≤ 4` on this hardware/image.
To exercise higher `n_experts` (the whole point of the headline
experiment), the dispatch needs to be **restructured** — vectorized
across experts with a single matmul, or rewritten on top of
`torch.scatter`/`torch.index_select` so the autograd graph has many
fewer chained boolean-mask operations. The choice of restructuring is
part of the implementation, not the scaffolding; flagged in
`polaris/ffn/moe.py`.

### Heads-up — aux loss weight at small scale

The smoke initially used `AUX_W = 0.01` (the Switch transformer paper's
default). At this small scale (`n_experts=4`, ~1 MB dataset), that was
not enough to prevent **full router collapse** (a single expert
absorbing all tokens per layer; aux loss climbing rather than
decreasing). Bumping `AUX_W` to `0.1` produced balanced routing in
under 100 iters and aux loss stable near the theoretical equilibrium.

Takeaway: at the polaris experiment scale, the literature's `0.01` is
likely too weak. Plan to tune `aux_loss_weight` per `n_experts` /
dataset-size combination, or add a capacity-factor + token-drop policy
to provide a hard balance constraint independent of the soft aux.
