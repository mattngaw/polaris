"""Central configuration for `polaris` runs.

A single place to define a run's hyperparameters, with **FFN type
(`dense` | `moe`) as a first-class field** so the dense baseline and the
MoE arm share one code path and one config schema. The training loop never
branches on dataset, optimizer, or schedule — it branches *only* on
`ffn_type`, and even that branch lives behind a common FFN interface (see
`model.py`).

Defaults that belong here, given the gfx803 constraints:

- `dtype = torch.float32`     (no AMP / no bf16 on gfx803)
- `compile = False`           (no inductor backend for gfx803)
- `device = "cuda"`           (one device per process; pair with
                              `HIP_VISIBLE_DEVICES=0|1` at the shell)

Config fields to define:

- **model**: vocab_size, n_layer, n_head, n_embd, block_size, dropout
- **ffn**:   ffn_type, ffn_hidden, and (for moe) n_experts, top_k,
             capacity_factor, aux_loss_weight
- **optim**: learning_rate, weight_decay, betas, grad_clip
- **schedule**: warmup_iters, lr_decay_iters, min_lr, max_iters
- **train**: batch_size, gradient_accumulation_steps, eval_interval,
             eval_iters, seed, out_dir
- **data**:  dataset name and path to the prepared `.bin` files
"""

# TODO(human): decide the config representation (dataclasses vs YAML loader
#              vs Pydantic vs argparse). nanoGPT uses exec()-of-a-Python-file;
#              you can do better than that.
# TODO(human): define the schema with `ffn_type` as a first-class field.
# TODO(human): set fp32 + no-compile as defaults that can't be silently
#              overridden.
# TODO(human): add a `validate()` step that fails loud on incoherent combos
#              (e.g. ffn_type == "dense" but n_experts > 1).

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class Config:
    # model config
    vocab_size: int
    n_layer: int
    d_model: int
    max_seq_len: int
    dropout: float

    # attn config
    n_head: int
    d_head: int
    qkv_bias: bool

    # ffn config
    ffn_type: Literal["dense", "moe"]
    ffn_hidden: int
    # moe specific
    n_experts: int | None
    top_k: int | None
    aux_loss_weight: float

    # optimizer config
    learning_rate: float

    # schedule config
    warmup_iters: int
    min_learning_rate: float

    # train config
    seed: int
    batch_size: int
    max_iters: int

    # data config

    # gfx803 constraints
    dtype: str = "float32"
    compile: bool = False
    device: str = "cuda"

    def validate(self) -> None:
        if self.dtype != "float32":
            raise ValueError("gfx803 requires dtype='float32'")

        if self.compile:
            raise ValueError("gfx803 requires compile=False")

        if self.ffn_type == "dense":
            if self.n_experts is not None or self.top_k is not None:
                raise ValueError("dense FFN config should not set n_experts or top_k")

        if self.ffn_type == "moe":
            if self.n_experts is None:
                raise ValueError("MoE FFN config requires n_experts")
            if self.top_k is None:
                raise ValueError("MoE FFN config requires top_k")
            if self.n_experts < 1:
                raise ValueError("n_experts must be >= 1")
            if self.top_k < 1:
                raise ValueError("top_k must be >= 1")
            if self.top_k > self.n_experts:
                raise ValueError("top_k cannot exceed n_experts")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    raw.update(overrides or {})

    config = Config(**raw)
    config.validate()
    return config
