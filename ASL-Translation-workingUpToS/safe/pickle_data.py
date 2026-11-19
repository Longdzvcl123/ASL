"""
CSV file(s) of raw MediaPipe landmarks -> normalized, split dataset in a pickle file.

Usage:  python pickle_data.py [--csv FILE [FILE ...]] [--out New_asl_data.pkl]
Output: pickle of (train, val, test), each a list of (features[42], label_index).
"""
import argparse
import csv
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from landmarks import ALPHABET, BASE_DIR, DEFAULT_DATA_PATH, NUM_LANDMARKS, label_to_index, normalize_csv_row

# NOTE: the legacy '../asl_dataset_flagged.csv' is NOT used: trained alone it only reaches ~35%
# test accuracy (bad/mislabeled recordings) and mixing it in drops the model to ~68%.
DEFAULT_CSVS = [BASE_DIR / 'AB_asl_dataset.csv']
EXPECTED_COLUMNS = 1 + NUM_LANDMARKS * 3   # label + (x, y, z) * 21
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_BLOCKS_FOR_SPLIT = 3
# Consecutive webcam frames are almost identical. Splitting frame-by-frame puts
# near-duplicates in train AND val/test, which inflates accuracy. We split by
# blocks of consecutive frames instead, so validation measures real generalization.
BLOCK_SIZE = 100
SEED = 42

Sample = tuple[np.ndarray, int]
Recordings = dict[int, list[np.ndarray]]    # label index -> features, in recording order


def parse_row(row: list[str]) -> Sample | None:
    """ One CSV row -> (features, label index), or None if the row is invalid. """
    if len(row) != EXPECTED_COLUMNS:
        return None
    y_idx = label_to_index(row[0])
    if y_idx is None:
        return None
    try:
        coords = [float(v) for v in row[1:]]
    except ValueError:
        return None
    if not np.all(np.isfinite(coords)):
        return None
    return normalize_csv_row(coords), y_idx


def load_csv(path: Path | str) -> Recordings:
    """ Reads one CSV, keeps the recording order, skips (and reports) bad rows. """
    per_class: Recordings = defaultdict(list)
    skipped = 0
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)                       # header
        for row in reader:
            sample = parse_row(row)
            if sample is None:
                skipped += 1
                continue
            per_class[sample[1]].append(sample[0])
    if skipped:
        print(f"[!] Skipped {skipped} invalid row(s) in '{path}'")
    return dict(per_class)


def split_blocks(recordings: list[Recordings], rng: random.Random) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """ Per file and per class: cut into blocks of consecutive frames, assign whole blocks to train/val/test. """
    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []
    for per_class in recordings:
        for y_idx, samples in sorted(per_class.items()):
            blocks = [samples[i:i + BLOCK_SIZE] for i in range(0, len(samples), BLOCK_SIZE)]
            rng.shuffle(blocks)
            enough = len(blocks) >= MIN_BLOCKS_FOR_SPLIT
            n_val = max(1, round(len(blocks) * VAL_RATIO)) if enough else 0
            n_test = max(1, round(len(blocks) * TEST_RATIO)) if enough else 0
            for i, block in enumerate(blocks):
                target = val if i < n_val else test if i < n_val + n_test else train
                target.extend((x, y_idx) for x in block)
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def missing_letters(recordings: list[Recordings]) -> list[str]:
    present = {k for per_class in recordings for k in per_class}
    return [letter for i, letter in enumerate(ALPHABET) if i not in present]


def report(recordings: list[Recordings]) -> None:
    counts: Counter[str] = Counter()
    for per_class in recordings:
        counts.update({ALPHABET[k]: len(v) for k, v in per_class.items()})
    print("Samples per letter:", dict(sorted(counts.items())))
    missing = missing_letters(recordings)
    if missing:
        print(f"[!] No data for: {' '.join(missing)} -> the model can NEVER predict these letters.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', nargs='+', default=[str(p) for p in DEFAULT_CSVS])
    parser.add_argument('--out', default=str(DEFAULT_DATA_PATH))
    args = parser.parse_args(argv)

    recordings = []
    for path in args.csv:
        try:
            recordings.append(load_csv(path))
        except FileNotFoundError:
            print(f"ERROR: '{path}' not found.")
            return 1
    if not any(recordings):
        print("ERROR: no valid rows found.")
        return 1

    report(recordings)
    train, val, test = split_blocks(recordings, random.Random(SEED))

    with open(args.out, 'wb') as f:
        pickle.dump((train, val, test), f)
    print(f"[+] Saved '{args.out}' -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
