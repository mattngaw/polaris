"""GPT-style transformer for `polaris`.

Defines the model and its block. The block is intentionally structured so
that its feed-forward sub-layer is an **FFN object satisfying a common
interface**, not a hard-coded MLP. Both `polaris.ffn.dense` and
`polaris.ffn.moe` implement that interface, so swapping arms of the study
is a config change (`ffn_type: dense | moe`) — not a code change — and a
paired (dense, moe) run differs *only* in the FFN module.

The FFN interface, in prose (you write the actual signature in `ffn/`):

- It is an `nn.Module`.
- Forward takes a token-stream tensor of shape (B, T, C) and returns a
  tensor of the same shape.
- For MoE it additionally exposes whatever auxiliary signals the training
  loop needs to add the load-balancing loss and to record per-expert load
  (e.g. as a second return value, or as an attribute set during forward).
  Decide which discipline you want and apply it consistently across both
  FFN types so the training loop is uniform.

Borrowing note: the attention (causal self-attention with a triangular
mask), the LayerNorm/RMSNorm choice, and the residual block wiring are the
intellectually load-bearing parts of the model — write them yourself. The
token + position embeddings and the LM head are pure plumbing and are fair
game for direct borrowing from nanoGPT (mark them as such in CREDITS.md).
"""

# TODO(human): heads-up — the project is on ROCm 5.7 specifically
#              because ROCm 6.x has a gfx803 fp32 GEMM bug. Always run
#              `python env/check-matmul.py --dtype fp32` before training
#              to confirm your stack hasn't drifted. See KNOWN_GOTCHAS.md.
# TODO(human): implement the block:
#              norm -> attention -> residual -> norm -> FFN(via interface)
#              -> residual.
# TODO(human): implement causal self-attention (multi-head, fp32, no flash).
# TODO(human): wire token + position embeddings and the LM head
#              (may be borrowed verbatim from nanoGPT — note in CREDITS).
# TODO(human): instantiate the FFN in each block by looking up
#              `config.ffn_type` and dispatching to the dense or MoE class.
# TODO(human): expose a `from_config(cfg)` (or similar) that builds the whole
#              model given a Config and returns it ready to `.to(device)`.
