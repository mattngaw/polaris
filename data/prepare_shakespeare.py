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

import pickle
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path("data/shakespeare")


if __name__ == "__main__":
    # download tinyshakespeare
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "input.txt"
    path, _ = urlretrieve(
        TINY_SHAKESPEARE_URL, path
    )  # maybe should fix with real error handling

    # read it and break into chars
    with open(path) as f:
        text = f.read()
    chars = sorted(set(text))  # sorted so vocab has deterministic ids
    vocab_size = len(chars)

    # build encoder/decoder
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[i] for i in ids])

    # split into train and validation
    n = len(text)
    train_text = text[: int(0.9 * n)]
    val_text = text[int(0.9 * n) :]

    train_ids = encode(train_text)
    val_ids = encode(val_text)

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(DATA_DIR / "train.bin")
    val_ids.tofile(DATA_DIR / "val.bin")

    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
    }

    with open(DATA_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
