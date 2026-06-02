"""Mixture-of-Experts feed-forward layer — the focal arm of the study.

This is the heart of `polaris`. Implements the same FFN interface as
`dense.py`, but instead of a single MLP it contains:

1. **Router** — a small linear projection from (B, T, C) to
   (B, T, n_experts), followed by softmax and top-k selection. Outputs
   per-token expert assignments and gate weights.

2. **Expert pool** — `n_experts` independent dense FFN sub-modules. Each
   expert is structurally the same as the dense baseline but with hidden
   dimension `ffn_hidden_per_expert` (the *granularity* axis of the
   sweep).

3. **Dispatch / combine** — route each token to its top-k experts, run
   each expert on its assigned tokens, and combine the outputs weighted by
   the gates. **On gfx803 there are no block-sparse / grouped-GEMM kernels
   we can use**, so the implementation is the naive dense-masked version:
   every expert sees a (B*T, C) input multiplied by a (B*T,) selection
   mask. This is intentionally inefficient — the gap between this and
   "ideal" sparse MoE compute is exactly one of the things we want to
   measure.

4. **Load-balancing auxiliary loss** — the standard
   `n_experts * mean(routing_prob) . mean(top1_assignment)` term
   (Switch / GShard formulation). Returned via the FFN interface's aux
   channel so the training loop can add `aux_loss_weight * aux` to the
   main cross-entropy loss.

5. **Instrumentation hook** — record per-expert token counts each forward.
   This is read by `instrumentation.py` to plot load imbalance over
   training.

The "capacity factor" and dropped-token handling are yours to decide —
Switch picks one expert with capacity, GShard picks two; both are fair
game, just be consistent across the sweep.
"""

# TODO(human): heads-up — masked dispatch was not exercised in
#              path-validation (the fp32 GEMM bug on the 6.x image
#              surfaced first). Now that the project is on ROCm 5.7
#              with fp32 GEMM working, verify the masked-MoE pattern
#              (index_put-style writes + per-expert aux loss
#              accumulation) explicitly when you implement it. See
#              KNOWN_GOTCHAS.md.
# TODO(human): implement the router: Linear(C, n_experts) -> softmax
#              -> top-k.
# TODO(human): implement the expert pool as nn.ModuleList of `n_experts`
#              dense FFNs (you can reuse the dense module from `dense.py`).
# TODO(human): implement the naive masked dispatch/combine. Document the
#              shape of the mask and the cost in a comment — it is
#              O(n_experts * B*T*C*H); that is not a bug, that is the thing
#              we are measuring.
# TODO(human): heads-up — the naive boolean-mask-per-expert pattern
#              (`flat[mask]` gather → expert(...) → `out[mask] = ...`)
#              **crashes the GPU backward at n_experts=8** on this stack
#              (gfx803 + ROCm 5.7 + PyTorch 2.3). Reproduced at B=16 and
#              B=32, with aux_loss_weight=0 (so the dispatch backward is
#              the cause, not the aux). n_experts=4 is fine. The headline
#              expert-count sweep will require restructuring this — likely
#              a single vectorized matmul across experts or a
#              torch.scatter / torch.index_select formulation with a much
#              smaller autograd graph. See KNOWN_GOTCHAS.md.
# TODO(human): implement the load-balancing aux loss; expose it via the
#              FFN interface's auxiliary channel.
# TODO(human): record per-expert token counts in a way that
#              `instrumentation.py` can pick up (a module attribute
#              updated each forward is fine).
# TODO(human): decide on capacity_factor / drop policy and document it.
# TODO(human): heads-up — at small scale, the Switch paper's default
#              `aux_loss_weight = 0.01` is too weak: a path-validation
#              smoke (n_experts=4, ~1 MB dataset) saw full router collapse
#              within ~100 iters at 0.01, and balanced routing at 0.1.
#              Plan to either tune `aux_loss_weight` per (n_experts,
#              dataset_size) combination or add a capacity-factor + drop
#              policy that constrains balance independently of the soft
#              aux. See KNOWN_GOTCHAS.md.
# TODO(human): heads-up — naive masked dispatch's wall-clock cost
#              scales linearly with n_experts at fixed top_k=1. In the
#              smoke (n_experts=4) step time was ~2.4× the matched-dense
#              baseline. Sweep ratio: roughly (n_experts / top_k) × dense.
#              That gap is one of the things `instrumentation.py` should
#              measure formally.
