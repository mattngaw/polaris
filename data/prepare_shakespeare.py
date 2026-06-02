"""Prepare the char-level Shakespeare dataset for smoke testing.

This dataset exists **only** to confirm the training loop runs end-to-end.
None of the science depends on it.

The pipeline:

1. Download Karpathy's `tinyshakespeare/input.txt` (~1 MB).
2. Build a character-level vocabulary (set of unique characters, sorted).
3. Encode the full text as a stream of `uint16` token ids.
4. Random or 90/10 split into train/val.
5. Write `data/shakespeare/train.bin` and `val.bin`.
6. Pickle `meta.pkl` with `{vocab_size, stoi, itos}`.

No BPE here — character-level keeps the smoke test trivial and removes the
tokenizer as a moving part.
"""

# TODO(human): download `tinyshakespeare/input.txt`.
# TODO(human): build the character vocabulary (stoi / itos).
# TODO(human): encode the text, split, and write the .bin files.
# TODO(human): pickle meta.pkl.

if __name__ == "__main__":
    raise NotImplementedError("TODO(human): implement preparation pipeline.")
