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
