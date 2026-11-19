import random

import numpy as np

import pickle_data
from conftest import load_pickle, make_hand, write_csv
from landmarks import INPUT_SIZE


def valid_row(label="A", seed=0):
    return [label, *make_hand(seed).flatten().tolist()]


def test_parse_row_valid():
    sample = pickle_data.parse_row(valid_row("C"))
    assert sample is not None
    features, y = sample
    assert y == 2 and features.shape == (INPUT_SIZE,)


def test_parse_row_rejects_bad_rows():
    good = valid_row()
    assert pickle_data.parse_row([]) is None
    assert pickle_data.parse_row(good[:-1]) is None                     # missing column
    assert pickle_data.parse_row(["AB", *good[1:]]) is None             # bad label
    assert pickle_data.parse_row(["A", "oops", *good[2:]]) is None      # not a number
    assert pickle_data.parse_row(["A", "nan", *good[2:]]) is None       # not finite


def test_load_csv_skips_invalid_rows(tmp_path, capsys):
    path = tmp_path / 'd.csv'
    write_csv(path, [valid_row("A", 1), valid_row("B", 2), ["ZZ", *valid_row()[1:]], valid_row()[:10]])
    per_class = pickle_data.load_csv(path)
    assert {k: len(v) for k, v in per_class.items()} == {0: 1, 1: 1}
    assert "Skipped 2 invalid row(s)" in capsys.readouterr().out


def test_split_blocks_keeps_whole_blocks_together():
    # 1000 frames of one class, each frame tagged with its position in feature[0]
    samples = [np.full(INPUT_SIZE, i, dtype=np.float32) for i in range(1000)]
    train, val, test = pickle_data.split_blocks([{0: samples}], random.Random(0))

    assert len(train) + len(val) + len(test) == 1000
    assert len(val) == len(test) == pickle_data.BLOCK_SIZE

    def blocks(part):
        return {int(x[0]) // pickle_data.BLOCK_SIZE for x, _ in part}

    assert blocks(train).isdisjoint(blocks(val))
    assert blocks(train).isdisjoint(blocks(test))
    assert blocks(val).isdisjoint(blocks(test))


def test_split_blocks_small_class_goes_to_train_only():
    samples = [np.zeros(INPUT_SIZE, dtype=np.float32)] * 150     # only 2 blocks
    train, val, test = pickle_data.split_blocks([{5: samples}], random.Random(0))
    assert len(train) == 150 and not val and not test


def test_split_is_reproducible():
    samples = [np.full(INPUT_SIZE, i, dtype=np.float32) for i in range(500)]
    a = pickle_data.split_blocks([{0: samples}], random.Random(42))
    b = pickle_data.split_blocks([{0: samples}], random.Random(42))
    assert [float(x[0]) for x, _ in a[1]] == [float(x[0]) for x, _ in b[1]]


def test_missing_letters_and_report(capsys):
    recordings = [{0: [np.zeros(INPUT_SIZE)]}, {1: [np.zeros(INPUT_SIZE)] * 2}]
    assert pickle_data.missing_letters(recordings)[:2] == ["C", "D"]
    pickle_data.report(recordings)
    out = capsys.readouterr().out
    assert "'A': 1" in out and "'B': 2" in out and "No data for: C D" in out


def test_main_writes_pickle(small_pickle):
    train, val, test = load_pickle(small_pickle)
    assert len(train) + len(val) + len(test) == 900
    assert val and test
    x, y = train[0]
    assert x.shape == (INPUT_SIZE,) and isinstance(y, int)


def test_main_accepts_several_csv(tmp_path, small_csv):
    out = tmp_path / 'two.pkl'
    assert pickle_data.main(['--csv', str(small_csv), str(small_csv), '--out', str(out)]) == 0
    assert sum(len(part) for part in load_pickle(out)) == 1800


def test_main_missing_file(tmp_path, capsys):
    assert pickle_data.main(['--csv', str(tmp_path / 'nope.csv'), '--out', str(tmp_path / 'o.pkl')]) == 1
    assert "not found" in capsys.readouterr().out


def test_main_no_valid_rows(tmp_path):
    path = tmp_path / 'empty.csv'
    write_csv(path, [])
    assert pickle_data.main(['--csv', str(path), '--out', str(tmp_path / 'o.pkl')]) == 1
