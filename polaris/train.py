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

import os
import pickle
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, load_config
from .model import GPTModel

DATA_DIR = "data/shakespeare"
PICKLE_PATH = DATA_DIR + "/meta.pkl"
CONFIG_PATH = "experiments/configs/dense.yaml"


def get_batch(config: Config, split: Literal["train", "val"]):
    if split == "train":
        data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype="uint16", mode="r")
    elif split == "val":
        data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype="uint16", mode="r")
    idxs = torch.randint(0, len(data) - config.max_seq_len, (config.batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(data[idx : idx + config.max_seq_len].astype(np.int64))
            for idx in idxs
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                data[idx + 1 : idx + 1 + config.max_seq_len].astype(np.int64)
            )
            for idx in idxs
        ]
    )
    x = x.pin_memory().to(config.device, non_blocking=True)
    y = y.pin_memory().to(config.device, non_blocking=True)
    return x, y


if __name__ == "__main__":
    with open(PICKLE_PATH, "rb") as f:
        meta = pickle.load(f)

    overrides = {}
    if meta.get("vocab_size"):
        overrides["vocab_size"] = meta.get("vocab_size")

    config = load_config(CONFIG_PATH, overrides)

    torch.manual_seed(config.seed)

    model = GPTModel(config).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()

    for it in range(config.max_iters):
        x, y = get_batch(config, "train")
        logits, aux = model(x)
        # todo fold aux
        aux = [a for a in aux if a is not None]

        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        loss = loss + config.aux_loss_weight * sum(aux)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % 20 == 0:
            print(f"iter {it} loss: {loss.item()}")
