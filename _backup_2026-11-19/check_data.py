import pickle
import numpy as np

with open('New_asl_data.pkl', 'rb') as f:
    train, val, test = pickle.load(f)

# On regarde les 5 premiers labels uniques
labels = [y for x, y in train]
# Si tes labels sont des One-Hot, on prend l'argmax
if isinstance(labels[0], (np.ndarray, list)):
    unique_labels = sorted(list(set([np.argmax(y) for y in labels])))
else:
    unique_labels = sorted(list(set(labels)))

print(f"Index trouvés dans le dataset : {unique_labels}")
print(f"Nombre de classes : {len(unique_labels)}")