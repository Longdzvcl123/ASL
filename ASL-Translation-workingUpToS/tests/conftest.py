import csv
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from landmarks import ALPHABET, NUM_LANDMARKS

RNG = np.random.default_rng(0)


def make_hand(seed: int = 0) -> np.ndarray:
    """ A plausible (21, 3) hand: random points with wrist and middle MCP well separated. """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.3, 0.7, size=(NUM_LANDMARKS, 3))
    pts[0, :2] = (0.5, 0.8)
    pts[9, :2] = (0.5, 0.5)
    return pts


def as_landmarks(points: np.ndarray) -> list[SimpleNamespace]:
    """ Mimics MediaPipe landmark objects. """
    return [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2]) if len(p) > 2 else 0.0) for p in points]


def write_csv(path, rows: list[list]) -> None:
    header = ['label'] + [f'{a}{i}' for i in range(NUM_LANDMARKS) for a in 'xyz']
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def class_prototype(label_idx: int) -> np.ndarray:
    """ Each class gets a distinct, learnable hand shape. """
    return make_hand(seed=1000 + label_idx)


@pytest.fixture
def small_csv(tmp_path):
    """ 3 letters x 300 frames (jittered prototypes) -> enough blocks for a train/val/test split. """
    rows = []
    for label_idx in range(3):
        base = class_prototype(label_idx)
        for _ in range(300):
            pts = base + RNG.normal(0, 0.003, base.shape)
            rows.append([ALPHABET[label_idx], *pts.flatten().tolist()])
    path = tmp_path / 'data.csv'
    write_csv(path, rows)
    return path


@pytest.fixture
def small_pickle(tmp_path, small_csv):
    import pickle_data
    out = tmp_path / 'data.pkl'
    assert pickle_data.main(['--csv', str(small_csv), '--out', str(out)]) == 0
    return out


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
