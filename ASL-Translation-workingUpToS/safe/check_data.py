""" Quick sanity check of the pickled dataset produced by pickle_data.py. """
import pickle
import sys
from collections import Counter

import numpy as np

from landmarks import ALPHABET, DEFAULT_DATA_PATH, INPUT_SIZE


def to_index(y: object) -> int:
    """ Supports both int labels and legacy one-hot labels. """
    return int(np.argmax(y)) if isinstance(y, (np.ndarray, list)) else int(y)


def main(path: str = str(DEFAULT_DATA_PATH)) -> int:
    try:
        with open(path, 'rb') as f:
            splits = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: '{path}' not found. Run pickle_data.py first!")
        return 1

    for name, data in zip(("train", "val", "test"), splits, strict=True):
        counts = Counter(ALPHABET[to_index(y)] for _, y in data)
        bad_shape = sum(1 for x, _ in data if np.asarray(x).size < INPUT_SIZE)
        print(f"{name:5s}: {len(data):6d} samples | {len(counts)} classes | bad shapes: {bad_shape}")
        print(f"       {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
