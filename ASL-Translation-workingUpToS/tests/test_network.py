import torch

from landmarks import DEFAULT_MODEL_PATH, INPUT_SIZE, NUM_CLASSES
from network import ASLNet, load_model, save_model


def test_forward_shape():
    model = ASLNet().eval()
    assert model(torch.zeros(4, INPUT_SIZE)).shape == (4, NUM_CLASSES)


def test_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    src = ASLNet().eval()
    path = tmp_path / 'm.pt'
    assert save_model(src, path)

    dst = ASLNet().eval()
    assert load_model(dst, path, 'cpu')
    x = torch.randn(3, INPUT_SIZE)
    torch.testing.assert_close(src(x), dst(x))


def test_load_missing_file(tmp_path, capsys):
    assert not load_model(ASLNet(), tmp_path / 'missing.pt', 'cpu')
    assert "not found" in capsys.readouterr().out


def test_load_corrupted_file(tmp_path):
    path = tmp_path / 'bad.pt'
    path.write_bytes(b'not a checkpoint')
    assert not load_model(ASLNet(), path, 'cpu')


def test_load_wrong_architecture(tmp_path):
    path = tmp_path / 'other.pt'
    torch.save(torch.nn.Linear(2, 2).state_dict(), path)
    assert not load_model(ASLNet(), path, 'cpu')


def test_save_to_invalid_location(tmp_path):
    assert not save_model(ASLNet(), tmp_path / 'no_such_dir' / 'm.pt')


def test_shipped_model_is_compatible():
    """ The committed best_asl_model.pt must load into the current architecture. """
    assert load_model(ASLNet(), DEFAULT_MODEL_PATH, 'cpu')
