"""Dense feed-forward layer — the baseline arm of the study.

The FFN-shaped comparison point for the MoE arm. Same input/output
contract, same parameter accounting story (you write the
FLOPs-per-token formula in `instrumentation.py` and reuse it here),
no routing, no experts.

Implements the FFN interface defined in prose in `model.py`. Match its
auxiliary-signal discipline (return shape, side-channels) so that the
training loop treats dense and MoE FFNs identically — i.e. no `if
ffn_type` branches in `train.py`.
"""

# TODO(human): heads-up — ROCm version matters here. The project is on
#              5.7 because 6.x has a gfx803 fp32 GEMM bug that would
#              silently break this layer. See KNOWN_GOTCHAS.md.
# TODO(human): implement the dense FFN module:
#              Linear(C, H) -> activation -> Linear(H, C), where H is set
#              by the config field `ffn_hidden`. Match the active-param
#              budget of the MoE arm you want to compare against.
# TODO(human): decide activation (GELU is the obvious default; matching it
#              between the dense baseline and the MoE experts keeps the
#              comparison clean).
# TODO(human): if your FFN interface returns a tuple (out, aux), return
#              aux=None here so the training loop is uniform.
