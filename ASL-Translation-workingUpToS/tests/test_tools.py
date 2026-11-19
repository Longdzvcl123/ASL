""" Tests for check_data.py and the non-camera parts of dataset_fill_fixed.py """
import csv
import pickle

import numpy as np
import pytest

import check_data
import dataset_fill_fixed as fill


def test_check_data_reports_splits(small_pickle, capsys):
    assert check_data.main(str(small_pickle)) == 0
    out = capsys.readouterr().out
    assert "train:" in out and "3 classes" in out and "bad shapes: 0" in out


def test_check_data_legacy_one_hot(tmp_path, capsys):
    path = tmp_path / 'legacy.pkl'
    sample = (np.zeros(42), np.eye(26)[4].reshape(26, 1))
    path.write_bytes(pickle.dumps(([sample], [(np.zeros(42), 4)], [])))
    assert check_data.main(str(path)) == 0
    assert "'E': 1" in capsys.readouterr().out


def test_check_data_missing_file(tmp_path, capsys):
    assert check_data.main(str(tmp_path / 'nope.pkl')) == 1
    assert "Run pickle_data.py first" in capsys.readouterr().out


@pytest.mark.parametrize("frame, expected", [(0, 0), (199, 0), (200, 1), (1000, 5), (1200, 0)])
def test_variation_phase_cycles(frame, expected):
    assert fill.get_variation_phase(frame) == fill.PHASES[expected]


def test_ensure_header_creates_once(tmp_path):
    path = tmp_path / 'rec.csv'
    fill.ensure_header(path)
    fill.ensure_header(path)            # must not duplicate / overwrite
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1
    assert rows[0][:4] == ['label', 'x0', 'y0', 'z0'] and len(rows[0]) == 64


def test_ask_label_retries_until_valid(monkeypatch, capsys):
    answers = iter(["", "AB", "7", " k "])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert fill.ask_label() == "K"
    assert capsys.readouterr().out.count("single letter A-Z") == 3


def test_ask_label_quit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "Quit")
    assert fill.ask_label() is None


def test_main_rejects_invalid_frames(tmp_path):
    with pytest.raises(SystemExit):
        fill.main(['--frames', '0', '--csv', str(tmp_path / 'x.csv')])


def test_main_missing_task_file(monkeypatch, tmp_path):
    monkeypatch.setattr(fill, "HAND_MODEL_PATH", tmp_path / "missing.task")
    with pytest.raises(SystemExit, match="not found"):
        fill.main(['--csv', str(tmp_path / 'x.csv')])


class FakeCapture:
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


class FakeDetector:
    """ Returns a hand on every other frame. """
    def __init__(self):
        from conftest import as_landmarks, make_hand
        self.hand, self.calls, self.timestamps, self.closed = as_landmarks(make_hand()), 0, [], False

    def detect_for_video(self, _img, ts):
        self.calls += 1
        self.timestamps.append(ts)
        hands = [self.hand] if self.calls % 2 else []
        return type("R", (), {"hand_landmarks": hands})()

    def close(self):
        self.closed = True


@pytest.fixture
def no_gui(monkeypatch):
    monkeypatch.setattr(fill.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(fill.cv2, "waitKey", lambda *a: -1)
    monkeypatch.setattr(fill.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(fill.time, "sleep", lambda *_: None)


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.reader(f))


def test_record_letter_writes_only_frames_with_a_hand(tmp_path, no_gui):
    path = tmp_path / 'rec.csv'
    fill.ensure_header(path)
    detector = FakeDetector()
    count = fill.record_letter(FakeCapture(100), detector, "B", str(path), frames_per_letter=5)
    rows = read_rows(path)
    assert count == 5 and len(rows) == 6
    assert all(r[0] == "B" and len(r) == 64 for r in rows[1:])
    assert detector.calls == 9                      # hand on every other frame
    assert detector.timestamps == sorted(set(detector.timestamps))


def test_record_letter_stops_when_camera_fails(tmp_path, no_gui, capsys):
    count = fill.record_letter(FakeCapture(3), FakeDetector(), "C", str(tmp_path / 'r.csv'), 50)
    assert count == 2
    assert "Failed to grab frame" in capsys.readouterr().out


def test_record_letter_esc_interrupts(tmp_path, no_gui, monkeypatch):
    monkeypatch.setattr(fill.cv2, "waitKey", lambda *a: fill.ESC_KEY)
    assert fill.record_letter(FakeCapture(10), FakeDetector(), "D", str(tmp_path / 'r.csv'), 50) == 1


def test_main_full_session(tmp_path, no_gui, monkeypatch):
    cap, detector = FakeCapture(100), FakeDetector()
    answers = iter(["a", "quit"])
    monkeypatch.setattr("builtins.input", lambda _p: next(answers))
    monkeypatch.setattr(fill.cv2, "VideoCapture", lambda *_: cap)
    monkeypatch.setattr(fill, "create_detector", lambda: detector)
    path = tmp_path / 'session.csv'
    assert fill.main(['--csv', str(path), '--frames', '3']) == 0
    assert len(read_rows(path)) == 4
    assert cap.released and detector.closed


def test_main_camera_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(fill.cv2, "VideoCapture", lambda *_: FakeCapture(0, opened=False))
    with pytest.raises(SystemExit, match="Could not open webcam"):
        fill.main(['--csv', str(tmp_path / 'x.csv')])


def test_create_detector_real():
    detector = fill.create_detector()
    detector.close()
