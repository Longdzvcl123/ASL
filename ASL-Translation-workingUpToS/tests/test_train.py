import numpy as np
import pytest
import torch

import train
from landmarks import INPUT_SIZE, NUM_LANDMARKS
from network import ASLNet


def test_augment_shape_and_device():
    x = torch.randn(16, INPUT_SIZE)
    out = train.augment(x)
    assert out.shape == x.shape and out.device == x.device


def test_augment_rotation_preserves_distances_up_to_scale():
    torch.manual_seed(0)
    x = torch.randn(64, INPUT_SIZE)
    out = train.augment(x)
    ratio = out.view(64, NUM_LANDMARKS, 2).norm(dim=2) / x.view(64, NUM_LANDMARKS, 2).norm(dim=2)
    # only scale jitter (+/-5%) and tiny noise may change lengths
    max_change = train.SCALE_JITTER / 2 + 0.01
    assert torch.all((ratio - 1).abs() < max_change + 0.05)


def test_augment_uses_a_different_angle_per_sample():
    torch.manual_seed(0)
    x = torch.tensor([[1.0, 0.0] * NUM_LANDMARKS] * 32)
    angles = torch.atan2(train.augment(x)[:, 1], train.augment(x)[:, 0])
    assert angles.std() > 0.01
    assert angles.abs().max() <= np.deg2rad(train.MAX_ROTATION_DEG) + 0.05


@pytest.mark.parametrize("label", [3, np.int64(3), np.eye(26)[3], np.eye(26)[3].reshape(26, 1)])
def test_to_tensors_supports_int_and_one_hot_labels(label):
    x, y = train.to_tensors([(np.zeros(INPUT_SIZE), label)])
    assert x.shape == (1, INPUT_SIZE) and x.dtype == torch.float32
    assert y.tolist() == [3] and y.dtype == torch.long


def test_accuracy():
    model = ASLNet().eval()
    x = torch.randn(10, INPUT_SIZE)
    y = model(x).argmax(1)
    assert train.accuracy(model, x, y) == 1.0


def test_main_trains_and_reports_test_accuracy(small_pickle, tmp_path, capsys):
    out_dir = tmp_path / 'out'
    assert train.main(['--data', str(small_pickle), '--epochs', '10', '--out-dir', str(out_dir)]) == 0
    assert (out_dir / 'best_asl_model.pt').exists()
    assert (out_dir / 'last_asl_model.pt').exists()
    text = capsys.readouterr().out
    assert "Test accuracy (best model)" in text
    test_acc = float(text.split("Test accuracy (best model): ")[1].split("%")[0])
    assert test_acc > 90      # 3 well-separated synthetic classes must be learned


def test_early_stopping(small_pickle, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(train, "EARLY_STOP_PATIENCE", 1)
    monkeypatch.setattr(train, "EVAL_EVERY", 1)
    assert train.main(['--data', str(small_pickle), '--epochs', '50', '--out-dir', str(tmp_path)]) == 0
    assert "Early stopping" in capsys.readouterr().out


def test_main_missing_data(tmp_path, capsys):
    assert train.main(['--data', str(tmp_path / 'nope.pkl'), '--out-dir', str(tmp_path)]) == 1
    assert "Run pickle_data.py first" in capsys.readouterr().out


def test_main_empty_split(tmp_path):
    import pickle
    path = tmp_path / 'empty.pkl'
    path.write_bytes(pickle.dumps(([], [], [])))
    assert train.main(['--data', str(path), '--out-dir', str(tmp_path)]) == 1


def test_main_rejects_zero_epochs(tmp_path):
    with pytest.raises(SystemExit):
        train.main(['--epochs', '0', '--out-dir', str(tmp_path)])
