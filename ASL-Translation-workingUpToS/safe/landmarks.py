"""
Shared constants and landmark preprocessing.

Every script (pickle_data, train, demo) MUST use the functions in this file so the
features seen during training are exactly the features seen during inference.
"""
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Folder containing the scripts: default file paths are resolved from here, so the
# scripts work no matter which directory they are launched from.
BASE_DIR = Path(__file__).resolve().parent
HAND_MODEL_PATH = BASE_DIR / 'hand_landmarker.task'
DEFAULT_MODEL_PATH = BASE_DIR / 'best_asl_model.pt'
DEFAULT_DATA_PATH = BASE_DIR / 'New_asl_data.pkl'

NUM_LANDMARKS = 21
INPUT_SIZE = NUM_LANDMARKS * 2          # X and Y only (Z is too noisy from a webcam)
ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
NUM_CLASSES = len(ALPHABET)
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9                      # base of the middle finger, used as scale reference
EPS = 1e-8

# Standard MediaPipe hand skeleton (index pairs)
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),   # Palm
    (1, 2), (2, 3), (3, 4),                               # Thumb
    (5, 6), (6, 7), (7, 8),                               # Index
    (9, 10), (10, 11), (11, 12),                          # Middle
    (13, 14), (14, 15), (15, 16),                         # Ring
    (17, 18), (18, 19), (19, 20),                         # Pinky
]


def normalize_points(points_xy: Any) -> np.ndarray:
    """
    Translation + scale invariant features.
    points_xy: array-like of shape (21, 2).
    Returns a float32 vector of length 42.
    """
    points = np.asarray(points_xy, dtype=np.float64).reshape(NUM_LANDMARKS, 2)
    anchored = points - points[WRIST_IDX]                       # wrist becomes the origin
    scale = np.linalg.norm(anchored[MIDDLE_MCP_IDX]) + EPS      # distance-to-camera invariant
    return (anchored / scale).flatten().astype(np.float32)


def normalize_csv_row(coords_raw: Sequence[float]) -> np.ndarray:
    """ 63 raw floats (x, y, z per landmark) from the CSV -> 42 normalized features. """
    return normalize_points(np.asarray(coords_raw, dtype=np.float64).reshape(NUM_LANDMARKS, 3)[:, :2])


def normalize_landmarks(hand_landmarks: Sequence[Any]) -> np.ndarray:
    """ MediaPipe landmark list (objects with .x and .y) -> 42 normalized features. """
    return normalize_points([[lm.x, lm.y] for lm in hand_landmarks])


def label_to_index(label: str) -> int | None:
    """ 'A'..'Z' -> 0..25, or None if the label is invalid. """
    label = label.strip().upper()
    if len(label) == 1 and label in ALPHABET:
        return ALPHABET.index(label)
    return None
