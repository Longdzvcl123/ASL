# ASL Translation — real-time sign language alphabet recognition

Recognizes American Sign Language (ASL) alphabet letters from a webcam in real time.
[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) extracts 21 hand landmarks per
frame, and a small PyTorch network classifies their normalized coordinates.

```
webcam frame ──► MediaPipe Hand Landmarker ──► 21 (x, y) points ──► normalize ──► ASLNet (MLP) ──► letter + confidence
                                                                   (wrist = origin,
                                                                    scale = wrist→middle-finger base)
```

## Project structure

```
.
├── safe/
│   ├── demo.py                 # real-time webcam demo
│   ├── dataset_fill_fixed.py   # record your own landmarks into a CSV
│   ├── pickle_data.py          # CSV -> normalized train/val/test split (.pkl)
│   ├── check_data.py           # sanity check of the .pkl
│   ├── train.py                # train the model, prints test accuracy
│   ├── network.py              # ASLNet architecture + save/load helpers
│   ├── landmarks.py            # shared constants & normalization (single source of truth)
│   ├── best_asl_model.pt       # trained weights used by the demo
│   ├── hand_landmarker.task    # MediaPipe hand model
│   └── AB_asl_dataset.csv      # recorded dataset (16 801 frames)
├── tests/                      # pytest suite (no webcam needed)
├── requirements.txt
└── requirements-dev.txt
```

## Installation

Requires **64-bit Python 3.10+** (PyTorch has no 32-bit build).

```bash
python -m venv venv_asl
# Windows:  .\venv_asl\Scripts\activate      (if blocked: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser)
# Linux/macOS: source venv_asl/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

NVIDIA GPU (optional): reinstall torch with the CUDA command from <https://pytorch.org/get-started/locally/>.
Training also runs fine on CPU (~1–2 min).

## Usage

The scripts locate their default files relative to `safe/`, so they can be launched from any folder.

```bash
python safe/demo.py                          # press 'q' to quit
python safe/demo.py --camera 1               # use another webcam
```

### Train with your own data

```bash
python safe/dataset_fill_fixed.py --csv safe/my_data.csv        # 1. record (type a letter, sign it, ESC to stop early)
python safe/pickle_data.py --csv safe/AB_asl_dataset.csv safe/my_data.csv   # 2. normalize + split
python safe/check_data.py                                       # 3. optional: per-letter counts
python safe/train.py                                            # 4. -> safe/best_asl_model.pt
python safe/demo.py                                             # 5. try it
```

`train.py` options: `--data`, `--epochs` (default 300, with early stopping), `--out-dir`.

> ⚠️ Steps 2 and 4 overwrite `safe/New_asl_data.pkl` and `safe/best_asl_model.pt`. Use `--out` / `--out-dir` to keep the old ones.

## Results & limitations

- **Supported letters:** the dataset only contains **A B C D E F G H I J K L M O R S**. N, P, Q, T, U, V, W, X, Y, Z
  have no data, so the model can never predict them. Record them with `dataset_fill_fixed.py` to add them.
- **Honest evaluation:** consecutive webcam frames are near-duplicates, so `pickle_data.py` splits by blocks of 100
  consecutive frames (not frame by frame). With that split the network reaches **~99% validation / ~93% test** accuracy.
  The most confused letters are L→S, I→L/K, M→A, E→S (worth re-recording).
- J and Z are motion signs; a single static frame cannot fully capture them.
- The data was recorded with one hand; signing with the other hand gives mirrored landmarks and lower accuracy.
- Camera resolution is fixed to 640×480 in both the recorder and the demo (x/y are relative to the frame aspect ratio).

## Development

```bash
pip install -r requirements-dev.txt
pytest --cov                 # 77 tests, ~98% coverage, no webcam required
ruff check safe tests
```

CI (GitHub Actions) runs lint + tests on Python 3.10 and 3.12 for every push / pull request.
