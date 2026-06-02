"""Prepare the TinyStories dataset for training.

End state: writes `data/tinystories/train.bin`, `data/tinystories/val.bin`,
the tokenizer artifact, and `meta.pkl` describing vocab size and tokenizer.

The pipeline this should run end-to-end:

1. Download the TinyStories corpus (HuggingFace `roneneldan/TinyStories` is
   the conventional source; pick a copy whose license you've verified).
2. Train a **small custom BPE tokenizer** with vocab size in the 4-8 K
   range. Do *not* use the GPT-2 50 K vocab — a large embedding table
   would absorb the parameter budget that needs to land in the experts for
   the sweep to be meaningful.
3. Tokenize the entire corpus and concatenate into one stream.
4. Random-split into train/val (or use any splits the dataset provides).
5. Write the streams as `uint16` to flat `.bin` files (`np.memmap`-friendly,
   nanoGPT convention).
6. Pickle a small `meta.pkl` with at least `{vocab_size, tokenizer_path}`.

This is plumbing, but writing the data pipeline yourself is part of the
learning project.
"""

# TODO(human): pick the tokenizer library (sentencepiece vs HF tokenizers)
#              and pin it in pyproject.toml.
# TODO(human): download the dataset.
# TODO(human): train the BPE tokenizer; save its artifact under
#              data/tinystories/.
# TODO(human): tokenize, split, write the .bin files and meta.pkl.

if __name__ == "__main__":
    raise NotImplementedError("TODO(human): implement preparation pipeline.")
