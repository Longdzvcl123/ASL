import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn

from landmarks import INPUT_SIZE, NUM_CLASSES


# --- MODEL ARCHITECTURE ---
# WARNING: changing layer names/sizes breaks compatibility with existing .pt checkpoints.
class ASLNet(nn.Module):
    """
    ASL letter classifier.
    Input: 42 normalized coordinates (21 landmarks * X,Y).
    Output: 26 logits (Alphabet A-Z).
    """
    def __init__(self, input_size: int = INPUT_SIZE, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_size, 1024)   # Input (42) --> Hidden 1 (1024)
        self.fc2 = nn.Linear(1024, 512)          # Hidden 1 --> Hidden 2 (512)
        self.fc3 = nn.Linear(512, 256)           # Hidden 2 --> Hidden 3 (256)
        self.fc_out = nn.Linear(256, num_classes)  # Hidden 3 --> Output (26)

        # ReLU: negative activations are set to 0, positive ones pass through
        self.relu = nn.ReLU()
        # Dropout: randomly disables 20% of neurons during training so the network
        # cannot rely on a few specific neurons (reduces over-fitting / memorization)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc_out(x)


# --- HELPER FUNCTIONS ---

def save_model(model: nn.Module, path: str | Path) -> bool:
    """ Saves the model weights. Returns True if successful. """
    try:
        torch.save(model.state_dict(), path)
        print(f"[+] Model successfully saved to: {path}")
        return True
    except (OSError, RuntimeError) as e:
        print(f"[-] Error saving model to '{path}': {e}")
        return False


def load_model(model: nn.Module, path: str | Path, device: torch.device | str) -> bool:
    """
    Loads weights into `model` (GPU/CPU mapping handled automatically).
    Returns True if successful, False otherwise.
    """
    if not os.path.exists(path):
        print(f"[-] Error: Model file '{path}' not found.")
        return False

    try:
        # weights_only=True: never execute arbitrary pickled code from a .pt file
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[+] Model weights loaded successfully on: {device}")
        return True
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as e:
        print(f"[-] Failed to load model weights from '{path}': {e}")
        return False
