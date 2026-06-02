"""Measurement hooks — the primary deliverable of this project.

At this scale the validation-loss curves between (dense, moe) pairs will
be close together and noisy. The **systems measurements** are what we are
actually here for and what will produce legible, citable plots:

1. **Per-expert load** — for each MoE layer, the number of tokens routed
   to each expert per forward, accumulated over training. Plot the
   distribution over experts and how it evolves over iterations. This is
   the load imbalance the literature understates.

2. **FLOPs-per-token accountant** — a closed-form formula given the model
   config. Should report both:
     - **active** FLOPs/token (what the network actually computes per
       token, which is what wall-clock should track), and
     - **total parameter** FLOPs/token (what you would compute if every
       expert fired).
   The gap between these is the headline systems argument.

3. **Step timing** — wall-clock per step, with a sub-breakdown of how
   much of the step is spent in dispatch/combine masking. Compare against
   the dense baseline at matched active params; the overhead is the
   "naive-dispatch penalty."

This module exposes the hooks the training loop calls (e.g. a step timer,
an expert-load recorder, a FLOPs accountant, a metrics writer), plus
utilities to dump the collected metrics in a format your plotting
scripts can consume (JSON or CSV under `runs/<run_id>/metrics.*`).

Keep this module *cheap*. The measurement overhead must be small relative
to the step itself, or the timing numbers are useless. CUDA event timing
on ROCm works (HIP events under the hood) — prefer
`torch.cuda.Event(enable_timing=True)` over Python-side `time.time()`.
"""

# TODO(human): heads-up — the first 1-2 training iters have an outlier
#              step time (~0.5 s on these cards) because HIP kernels
#              compile / autotune on first use. Exclude the first few
#              iters from any reported step-time statistics, or your
#              "average step time" will be dominated by the warmup.
# TODO(human): a small `Timer`-style helper using
#              `torch.cuda.Event(enable_timing=True)` for start/stop around
#              dispatch/combine and the full step.
# TODO(human): an expert-load recorder that the MoE module updates each
#              forward; accumulates counts per (layer, expert) and
#              supports a `flush()` to a metrics file.
# TODO(human): `flops_per_token(config)` returning (active_flops,
#              total_flops). Derive from n_layer, n_embd, n_head,
#              block_size, ffn_hidden, n_experts, top_k, vocab_size.
# TODO(human): a metrics writer that writes one row per logged step to a
#              CSV or JSONL under `runs/<run_id>/`, with at least:
#              iter, train_loss, val_loss, aux_loss, step_ms, dispatch_ms,
#              per-expert token counts.
# TODO(human): keep all of the above optional / no-op when
#              ffn_type == "dense", so the dense baseline doesn't pay for
#              instrumentation it doesn't need.
