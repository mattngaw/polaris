"""Training loop for `polaris`.

End-to-end runner: load a config, build the model (dense or MoE,
transparently to this file), load `train.bin` / `val.bin` via memory-mapped
random-offset sampling, run AdamW with a cosine LR schedule and warmup,
log loss + per-expert load + step timing, save checkpoints, evaluate at
intervals.

Designed to be invoked as `python -m polaris.train --config <path>`, one
process per card, with `HIP_VISIBLE_DEVICES=0` or `=1` to select the card.
There is no multi-GPU distributed setup — RCCL on gfx803 is unreliable.

Several pieces are intended to be **borrowed by hand from nanoGPT** rather
than designed from scratch:

- `get_batch(split)` — np.memmap + random-offset sampling.
- The cosine-with-warmup LR schedule (or the linear-warmup wrapper).
- Checkpoint save/load (`torch.save` of state_dict + optimizer + iter_num).
- The eval loop (`estimate_loss` over a fixed number of batches per split).

Each borrowed function should carry the convention from `CREDITS.md`. The
pieces you write yourself are:

- The MoE aux-loss combination (this is project-specific; nanoGPT has no
  MoE).
- The per-expert load logging hook into `instrumentation.py`.
- The step-time / dispatch-time breakdown.
"""

# TODO(human): heads-up — `torch.cuda.Event(enable_timing=True)` works
#              on HIP/ROCm; use it for step timing. Also add an early
#              sanity check that calls env/check-matmul.py's main against
#              the configured dtype and bails loudly if it fails — that
#              way a future image drift can't silently break the loss
#              curves. See KNOWN_GOTCHAS.md for the underlying bug story.
# TODO(human): argument parsing (--config and CLI overrides).
# TODO(human): load config and instantiate model via the FFN-type dispatch
#              in `model.py`.
# TODO(human): borrow `get_batch` from nanoGPT (note in CREDITS).
#              Heads-up: the nanoGPT pattern (np.memmap + random offset
#              + stack int64 slices, no DataLoader workers) worked fine
#              in path-validation under the 5.7 container's default
#              flags. If you ever add `num_workers > 0` to a DataLoader,
#              you re-enter the `--ipc host` + `--shm-size 16G`
#              interaction zone that is currently untested. See
#              KNOWN_GOTCHAS.md.
# TODO(human): build the AdamW optimizer with weight-decay only on 2D
#              params (the nanoGPT param-group pattern is fine to borrow).
# TODO(human): borrow the cosine LR schedule from nanoGPT.
# TODO(human): main loop — forward, add aux loss if present, backward,
#              grad clip, step, optimizer.zero_grad(set_to_none=True).
# TODO(human): per-step instrumentation hooks (timing, per-expert load)
#              via `polaris.instrumentation`.
# TODO(human): eval interval — `estimate_loss` on a fresh sample of val
#              batches.
# TODO(human): checkpoint save/load.

if __name__ == "__main__":
    raise NotImplementedError("TODO(human): wire up the entry point.")
