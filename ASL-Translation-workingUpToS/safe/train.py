"""
Trains ASLNet on the pickle produced by pickle_data.py.

Usage: python train.py [--data New_asl_data.pkl] [--epochs 300] [--out-dir <script folder>]
"""
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from landmarks import BASE_DIR, DEFAULT_DATA_PATH, INPUT_SIZE, NUM_LANDMARKS
from network import ASLNet, load_model, save_model

# --- HYPERPARAMETERS ---
BATCH_SIZE = 512
LEARNING_RATE = 5e-4        # lower = slower but more stable learning
EVAL_EVERY = 5              # epochs between validations
LR_PATIENCE = 6             # validations without improvement before halving the LR
EARLY_STOP_PATIENCE = 20    # validations without improvement before stopping
MAX_ROTATION_DEG = 20
NOISE_AMPLITUDE = 0.002
SCALE_JITTER = 0.1          # +/- 5%
SEED = 42


def augment(batch_x: torch.Tensor) -> torch.Tensor:
    """ Random rotation (one angle PER SAMPLE), light noise and scale jitter, all on-device. """
    n = batch_x.size(0)
    angles = torch.deg2rad((torch.rand(n, 1, device=batch_x.device) * 2 - 1) * MAX_ROTATION_DEG)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)

    pts = batch_x.view(n, NUM_LANDMARKS, 2)
    x, y = pts[:, :, 0], pts[:, :, 1]
    rotated = torch.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a], dim=2).view(n, INPUT_SIZE)

    noise = (torch.rand_like(rotated) - 0.5) * NOISE_AMPLITUDE
    scale = 1.0 + (torch.rand(n, 1, device=batch_x.device) - 0.5) * SCALE_JITTER
    return (rotated + noise) * scale


def to_tensors(data: list) -> tuple[torch.Tensor, torch.Tensor]:
    """ list of (features, label) -> (X, y). Supports int labels and legacy one-hot labels. """
    x = torch.tensor(np.stack([np.asarray(f, dtype=np.float32).ravel()[:INPUT_SIZE] for f, _ in data]))
    y = torch.tensor([int(np.argmax(lbl)) if isinstance(lbl, (np.ndarray, list)) else int(lbl) for _, lbl in data],
                     dtype=torch.long)
    return x, y


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return (model(x).argmax(1) == y).float().mean().item()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(augment(batch_x)), batch_y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def fit(model: nn.Module, loader: DataLoader, val: tuple[torch.Tensor, torch.Tensor],
        epochs: int, device: torch.device, best_path: str) -> float:
    """ Training loop with LR scheduling, best-model checkpointing and early stopping. """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=LR_PATIENCE, factor=0.5)
    best_acc, stale = -1.0, 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        if epoch % EVAL_EVERY and epoch != epochs:
            continue
        acc = accuracy(model, *val)
        scheduler.step(acc)
        status = ""
        if acc > best_acc:
            best_acc, stale = acc, 0
            save_model(model, best_path)
            status = "NEW BEST!"
        else:
            stale += 1
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | Val Acc: {acc:.2%} | LR: {lr:.1e} {status}")
        if stale >= EARLY_STOP_PATIENCE:
            print(f"[!] Early stopping: no improvement for {stale * EVAL_EVERY} epochs.")
            break
    return best_acc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=str(DEFAULT_DATA_PATH))
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--out-dir', default=str(BASE_DIR))
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error('--epochs must be >= 1')

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training on: {device} ({gpu_name})")

    try:
        with open(args.data, 'rb') as f:
            train_raw, val_raw, test_raw = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: '{args.data}' not found. Run pickle_data.py first!")
        return 1
    if not train_raw or not val_raw:
        print("ERROR: train or validation split is empty.")
        return 1

    tx, ty = to_tensors(train_raw)
    val = tuple(t.to(device) for t in to_tensors(val_raw))
    loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best_asl_model.pt")
    model = ASLNet().to(device)
    print(f"Training for up to {args.epochs} epochs (Batch Size: {BATCH_SIZE}, train: {len(tx)}, val: {len(val[1])})")
    best_acc = fit(model, loader, val, args.epochs, device, best_path)

    save_model(model, os.path.join(args.out_dir, "last_asl_model.pt"))
    print(f"\nTraining Complete! Best validation accuracy: {best_acc:.2%}")

    # Final unbiased score: the test split was never used for any decision above
    if test_raw and load_model(model, best_path, device):
        test = tuple(t.to(device) for t in to_tensors(test_raw))
        print(f"Test accuracy (best model): {accuracy(model, *test):.2%}")
    print(f"Use '{best_path}' for your real-time demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
