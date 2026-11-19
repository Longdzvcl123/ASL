import torch
import torch.nn as nn
import os

# --- MODEL ARCHITECTURE ---
class ASLNet(nn.Module):
    """
    Universal ASL Neural Network Architecture.
    Optimized for high-performance inference on RTX GPUs.
    Input: 42 normalized coordinates (21 landmarks * X,Y).
    Output: 26 classes (Alphabet A-Z).
    """
    def __init__(self, input_size=42, num_classes=26):
        super(ASLNet, self).__init__()
        
        # Matches the architecture found in your training checkpoint
        self.fc1 = nn.Linear(input_size, 1024) #Input Layer of 42 inputs goes into the first hidden layer of 1024 nodes
        self.fc2 = nn.Linear(1024, 512) # First Layer (1024) --> Second Layer (512)
        self.fc3 = nn.Linear(512, 256) # Second layer (512) --> Third layer (256)
        self.fc_out = nn.Linear(256, num_classes)# Third layer (256) --> Output layer (26)
        
        # Activation and Regularization
        self.relu = nn.ReLU() # Rectified Linear Unit ==> if the functions returns a 
                              # value < 0 it's set to 0 (we don't care about this value)
                              # else the information (hand coordinates) is usefull

        self.dropout = nn.Dropout(0.2) # Allows us to disable 20% of the previous nodes
                                       # If you study with an oppen book you will remember the position of 
                                       # the text you want to read and not the concept that you are reading
                                       # it's kind of the same thing happening here

    def forward(self, x):
        """ Defines the forward pass of the neural network. """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)
        return x

# --- HELPER FUNCTIONS ---

def save_model(model, path):
    """
    Saves the model weights to the specified path.
    Used by train.py after a successful training session.
    """
    try:
        torch.save(model.state_dict(), path)
        print(f"[+] Model successfully saved to: {path}")
    except Exception as e:
        print(f"[-] Error saving model: {e}")

def load_model(model, path, device):
    """
    Universal loader for PyTorch models.
    Handles hardware mapping (GPU to CPU or vice-versa) automatically.
    Returns: True if successful, False otherwise.
    """
    if not os.path.exists(path):
        print(f"[-] Error: Model file '{path}' not found.")
        return False
        
    try:
        # 'map_location' is crucial for cross-platform compatibility
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[+] Model weights loaded successfully on: {device}")
        return True
    except Exception as e:
        print(f"[-] Failed to load model weights: {e}")
        return False