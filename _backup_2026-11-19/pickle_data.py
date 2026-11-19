import csv
import numpy as np
import pickle
import random

# --- LA MÊME FONCTION QUE DANS TRAIN ET DEMO ---
def normalize_geometric(coords_raw):
    # On passe de 63 à 42 (X et Y uniquement)
    points = np.array(coords_raw).reshape(21, 3)[:, :2]
    wrist = points[0].copy()
    anchored = points - wrist
    # Échelle stable sur le point 9 (base du majeur)
    scale = np.linalg.norm(anchored[9]) + 1e-8
    return (anchored / scale).flatten().astype(np.float32)

def vectorize_label(j):
    e = np.zeros((26, 1))
    e[j] = 1.0
    return e

with open('AB_asl_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    raw_data = list(reader)

dataset = []
for row in raw_data:
    label = row[0].strip() 
    # Lecture des 63 floats
    coords_raw = [float(x) for x in row[1 : ]]
    
    # --- CRUCIAL : On normalise AVANT de stocker ---
    x_norm = normalize_geometric(coords_raw)
    
    y_idx = ord(label) - ord('A')
    # x_norm fait déjà 42 de long maintenant
    dataset.append((x_norm, y_idx))

random.shuffle(dataset)

total_samples = len(dataset)
train_end = int(total_samples * 0.8)
val_end = int(total_samples * 0.9)

train_raw = dataset[:train_end]
val_raw = dataset[train_end:val_end]
test_data = dataset[val_end:]

# On garde le format que ton train.py attend
training_data = [(x, vectorize_label(y_idx)) for x, y_idx in train_raw]
validation_data = val_raw 

with open('New_asl_data.pkl', 'wb') as f:
    pickle.dump((training_data, validation_data, test_data), f)

print(f"✅ Dataset synchronisé : train: {len(training_data)}, val: {len(validation_data)}")