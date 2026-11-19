import numpy as np
import pytest

from conftest import as_landmarks, make_hand
from landmarks import (
    ALPHABET,
    INPUT_SIZE,
    MIDDLE_MCP_IDX,
    NUM_CLASSES,
    label_to_index,
    normalize_csv_row,
    normalize_landmarks,
    normalize_points,
)


def test_output_shape_and_dtype():
    out = normalize_points(make_hand()[:, :2])
    assert out.shape == (INPUT_SIZE,)
    assert out.dtype == np.float32


def test_wrist_is_origin_and_middle_mcp_has_unit_length():
    out = normalize_points(make_hand()[:, :2]).reshape(-1, 2)
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-7)
    assert np.linalg.norm(out[MIDDLE_MCP_IDX]) == pytest.approx(1.0, abs=1e-5)


def test_translation_and_scale_invariance():
    pts = make_hand()[:, :2]
    moved = pts * 2.5 + np.array([0.3, -0.1])
    np.testing.assert_allclose(normalize_points(pts), normalize_points(moved), atol=1e-5)


def test_degenerate_hand_does_not_divide_by_zero():
    out = normalize_points(np.zeros((21, 2)))
    assert np.all(np.isfinite(out))


def test_csv_row_and_mediapipe_paths_give_identical_features():
    pts = make_hand(3)
    np.testing.assert_array_equal(normalize_csv_row(pts.flatten().tolist()), normalize_landmarks(as_landmarks(pts)))


def test_csv_row_ignores_z():
    pts = make_hand(4)
    other_z = pts.copy()
    other_z[:, 2] += 5
    np.testing.assert_array_equal(normalize_csv_row(pts.flatten()), normalize_csv_row(other_z.flatten()))


@pytest.mark.parametrize("label, expected", [("A", 0), ("z", 25), (" m ", 12), ("\tQ", 16)])
def test_label_to_index_valid(label, expected):
    assert label_to_index(label) == expected


@pytest.mark.parametrize("label", ["", "AB", "1", "é", "label", "quit"])
def test_label_to_index_invalid(label):
    assert label_to_index(label) is None


def test_alphabet():
    assert NUM_CLASSES == 26 and ALPHABET[0] == "A" and ALPHABET[-1] == "Z"
