import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import random
from network import ASLNet, save_model
from torch.utils.data import DataLoader, TensorDataset

# --- DATA AUGMENTATION: GPU ROTATION ---
def rotate_points(tensors, device, max_angle=20):
    """ Applies random rotation to landmarks on the GPU """
    angle = np.radians(random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    cos_a = torch.tensor(cos_a, dtype=torch.float32).to(device)
    sin_a = torch.tensor(sin_a, dtype=torch.float32).to(device)

    # Reshape to (Batch, 21 points, 2 coordinates X/Y)
    pts = tensors.view(-1, 21, 2)
    x = pts[:, :, 0]
    y = pts[:, :, 1]

    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a

    return torch.stack([new_x, new_y], dim=2).view(-1, 42) 

def train():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training on: {device} ({gpu_name})")

    # 2. Data Loading
    try:
        with open('New_asl_data.pkl', 'rb') as f:
            train_raw, val_raw, _ = pickle.load(f)
    except FileNotFoundError:
        print("ERROR: 'New_asl_data.pkl' not found. Run pickle_data.py first!")
        return

    def prepare(data):
        # Force 2D (42 values) to ensure consistency
        x_tensors = torch.stack([torch.tensor(np.array(x)[:42], dtype=torch.float32) for x, y in data])
        y_labels = torch.tensor([np.argmax(y) if isinstance(y, (np.ndarray, list)) else y for x, y in data], dtype=torch.long)
        return x_tensors, y_labels

    print("Preparing tensors...")
    tx, ty = prepare(train_raw)
    vx, vy = prepare(val_raw)

    # 3. Hyperparameters (Optimized for RTX 3070)
    batch_size = 512
    learning_rate = 0.0005 # if low --> takes more time ^ learns better
    train_loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True)

    # 4. Model, Loss & Optimizer
    model = ASLNet(input_size=42).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Reduce LR if accuracy plateaus for 50 evaluation steps
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50, factor=0.5)

    best_acc = 0.0
    print(f"Training started for 5000 epochs (Batch Size: {batch_size})")

    # Ne pas mettre d'epoch trop grand
    # si trop grand --> apprentissage par coeur et dans d'autres contextes ==> BUG
    # si trop petit --> pas assez d'apprentissage 
    for epoch in range(301):
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # --- LIVE DATA AUGMENTATION ---
            aug_x = rotate_points(batch_x, device, max_angle=20) # 20° rotation
            noise = (torch.rand_like(aug_x) - 0.5) * 0.002      # Light noise
            scale = 1.0 + (torch.rand(aug_x.size(0), 1).to(device) - 0.5) * 0.1 # Scale ±5%
            
            aug_x = (aug_x + noise) * scale
            
            # Backpropagation
            optimizer.zero_grad()
            outputs = model(aug_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 5. Periodic Evaluation & Best Model Saving
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(vx.to(device))
                acc = (val_outputs.argmax(1) == vy.to(device)).float().mean().item()
                
                scheduler.step(acc)
                
                current_loss = epoch_loss / len(train_loader)
                
                if acc > best_acc:
                    best_acc = acc
                    save_model(model, "best_asl_model.pt") # Save the best model
                    status = "NEW BEST!"
                else:
                    status = ""

                print(f"Epoch {epoch:04d} | Loss: {current_loss:.4f} | Val Acc: {acc:.2%} {status}")

    # Final Save
    save_model(model, "last_model_5000.pt")
    print(f"\nTraining Complete! Highest accuracy reached: {best_acc:.2%}")
    print("Use 'best_asl_model.pt' for your real-time demo.")

if __name__ == "__main__":
    train()