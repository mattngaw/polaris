# Data

Two datasets are used in `polaris`. Neither is checked into the repo — the
prepared `.bin` files and tokenizer artifacts are all gitignored.

## Primary: TinyStories

- Small vocabulary; coherent at the 1-30 M-parameter scale.
- Rich enough that added expert capacity has *somewhere to go*: routers have
  structure to specialize on (named characters, dialogue vs. narration,
  simple scene transitions).
- This is the dataset the headline MoE-vs-dense sweep runs on.

## Smoke test: char-level Shakespeare

- Tiny. Sole purpose is to confirm the training loop runs end-to-end
  (loss decreases, a sample is produced).
- Not used for any of the science. Don't read into Shakespeare loss curves.

## Tokenizer

- For TinyStories: a **small custom BPE, ~4-8 K vocab.** *Not* the 50 K
  GPT-2 vocab — a big embedding table would soak up the parameter budget
  that needs to land in the experts for the sweep to be meaningful.
- For Shakespeare: **character-level** (no BPE training).

The BPE library is `# TODO(human): choose` — `sentencepiece` and HF
`tokenizers` are both candidates. Pin it in `pyproject.toml` once chosen.

## Expected layout

After running the prepare scripts (which are stubs right now), each dataset
should produce, under `data/<name>/`:

```
data/
├── tinystories/
│   ├── train.bin       # uint16 token stream, gitignored
│   ├── val.bin         # uint16 token stream, gitignored
│   ├── tokenizer.json  # or .model — gitignored
│   └── meta.pkl        # vocab size, tokenizer info — gitignored
└── shakespeare/
    ├── train.bin
    ├── val.bin
    └── meta.pkl        # char-to-int map, vocab size
```

The `.bin` files use the nanoGPT convention: a flat sequence of token IDs
as `uint16` (or `uint32` if vocab > 65 535) suitable for `np.memmap`. The
training loop reads them by sampling random offsets.

## Why `.bin` files (not HF `datasets`)

Memory-mapping a flat token array is the cheapest possible data path. No
tokenizer at train time, no PyArrow, no streaming. On 4 GB cards every byte
of memory matters and the data-loading overhead must be near zero. nanoGPT
does the same thing for the same reason.

## Preparation

Both `prepare_*.py` files are **stubs**. They should each:

1. Download the raw dataset.
2. Train the tokenizer (for TinyStories) or build the char map
   (for Shakespeare).
3. Tokenize the full corpus into a flat token stream.
4. Split into train/val.
5. Write `train.bin` / `val.bin` / `meta.pkl` under `data/<name>/`.

Implementing them is part of the project — see `LAB.md` §3 (Required
Artifacts) and the contract in each stub's docstring.
