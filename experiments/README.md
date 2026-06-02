# Experiments

The sweep design and where configs live.

## Headline experiment

Hold **active** parameters per token roughly fixed; vary total parameters
by sweeping the **number of experts** and **expert granularity**
(FFN hidden dim per expert). A dense FFN is run alongside as the
matched-active-params baseline.

For each (dataset, FFN type, total-params point) we want:

1. **Quality** — final validation loss / perplexity, learning curves.
2. **Systems** — FLOPs-per-token (active vs. total), wall-clock per step,
   per-expert token counts and load imbalance over training, dispatch
   time as a fraction of step time.

At this scale the **systems story is the primary deliverable**, not the
loss curves. Small loss differences across runs will sit inside the noise
floor at this size; load imbalance and the naive-dispatch penalty will
not.

## FFN-type axis

Configs distinguish runs by `ffn_type: dense | moe`. The model is the
same otherwise — same depth, attention, embedding, optimizer, schedule,
seed — so a paired run differs *only* in the FFN. That is what makes the
comparison meaningful.

## Two-card execution pattern

We have two RX 480s. Multi-GPU training is **not** used (RCCL on gfx803 is
unreliable). Instead we run **one experiment per card**, two cards at
once:

```
# host shell A
./env/run-container.sh
HIP_VISIBLE_DEVICES=0 python -m polaris.train --config experiments/configs/<dense>.yaml

# host shell B  (second shell into the same container)
./env/run-container.sh
HIP_VISIBLE_DEVICES=1 python -m polaris.train --config experiments/configs/<moe>.yaml
```

This pattern fits naturally with the headline sweep: each (dense, moe)
pair runs together, one per card, sharing nothing but the dataset on
disk.

The sweep driver that schedules pairs across the cards is `# TODO(human)`
— yours to write.

## `configs/`

Config files live here, one per run. Format is `# TODO(human): choose`
(YAML is the conventional pick; JSON or a Python dict are also fine).
Suggested naming: `<dataset>_<ffn>_<size>_<seed>.yaml`, e.g.
`tinystories_moe_8e_top2_s0.yaml`. Keep them human-readable so an old run
is diff-able against a new one.
