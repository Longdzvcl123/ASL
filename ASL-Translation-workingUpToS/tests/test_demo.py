from collections import deque

import mediapipe as mp
import numpy as np
import pytest
import torch

import demo
from conftest import as_landmarks, make_hand
from network import ASLNet


def probs(winner: int, p: float) -> np.ndarray:
    out = np.full(26, (1 - p) / 25)
    out[winner] = p
    return out


def test_stabilize_majority_vote_and_confidence():
    buffer = deque([probs(0, 0.9), probs(0, 0.7), probs(1, 0.99)])
    letter, confidence = demo.stabilize(buffer)
    assert letter == "A"
    assert confidence == pytest.approx((0.9 + 0.7 + (0.01 / 25)) / 3 * 100)


def test_predict_returns_probabilities():
    out = demo.predict(ASLNet().eval(), as_landmarks(make_hand()), torch.device("cpu"))
    assert out.shape == (26,)
    assert out.sum() == pytest.approx(1.0, abs=1e-5)


def test_get_best_device():
    assert demo.get_best_device().type in {"cpu", "cuda", "mps"}


def test_drawing_does_not_crash_and_draws_something():
    frame = np.zeros((480, 640, 3), np.uint8)
    demo.draw_hand(frame, as_landmarks(make_hand()))
    demo.draw_label(frame, "A", 95.0)
    demo.draw_label(frame, "B", 10.0)
    assert frame.any()


def test_detector_runs_in_video_mode():
    detector = demo.create_detector()
    assert detector is not None
    try:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((240, 320, 3), np.uint8))
        for ts in (1, 2, 3):
            result = detector.detect_for_video(img, ts)
        assert result.hand_landmarks == []
    finally:
        detector.close()


def test_create_detector_missing_task_file(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "HAND_MODEL_PATH", tmp_path / "missing.task")
    assert demo.create_detector() is None


def test_main_fails_cleanly_with_missing_model(tmp_path):
    assert demo.main(["--model", str(tmp_path / "missing.pt")]) == 1


class FakeCapture:
    """ Simulates a camera that delivers `n` black frames then stops. """
    def __init__(self, n: int, opened: bool = True):
        self.n, self.opened, self.released = n, opened, False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        return True, np.zeros((240, 320, 3), np.uint8)

    def set(self, *_):
        return True

    def release(self):
        self.released = True


@pytest.fixture
def no_gui(monkeypatch):
    monkeypatch.setattr(demo.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(demo.cv2, "waitKey", lambda *a: -1)
    monkeypatch.setattr(demo.cv2, "destroyAllWindows", lambda: None)


def test_run_asl_demo_full_loop_with_fake_camera(monkeypatch, no_gui, capsys):
    cap = FakeCapture(3)
    monkeypatch.setattr(demo.cv2, "VideoCapture", lambda *_: cap)
    assert demo.main([]) == 0
    assert cap.released
    assert "Camera stopped sending frames" in capsys.readouterr().out


def test_run_asl_demo_camera_not_available(monkeypatch, no_gui, capsys):
    cap = FakeCapture(0, opened=False)
    monkeypatch.setattr(demo.cv2, "VideoCapture", lambda *_: cap)
    assert demo.main(["--camera", "7"]) == 1
    assert cap.released
    assert "could not open camera 7" in capsys.readouterr().out


def test_run_loop_with_detected_hand(monkeypatch, no_gui):
    """ Fake detector always returns a hand: prediction + drawing path is exercised. """
    hand = as_landmarks(make_hand())
    timestamps = []

    class FakeDetector:
        def detect_for_video(self, _img, ts):
            timestamps.append(ts)
            return type("R", (), {"hand_landmarks": [hand]})()

    drawn = []
    monkeypatch.setattr(demo, "draw_label", lambda frame, letter, conf: drawn.append((letter, conf)))
    demo.run_loop(FakeCapture(4), FakeDetector(), ASLNet().eval(), torch.device("cpu"))
    assert len(drawn) == 4
    assert timestamps == sorted(set(timestamps))       # strictly increasing for VIDEO mode
