import numpy as np
import json

out_dir = 'data/adult/tabddpm/'

# Load arrays
y_train = np.load(f'{out_dir}/y_train.npy')
X_cat_train = np.load(f'{out_dir}/X_cat_train.npy')
X_num_train = np.load(f'{out_dir}/X_num_train.npy')

with open(f'{out_dir}/info.json') as f:
    info = json.load(f)

print("--- DIAGNOSTICS ---")

# 1. Check Target Bounds
y_unique = np.unique(y_train)
print(f"y_train unique values: {y_unique}")
if y_train.max() >= 2 or y_train.min() < 0:
    print("🚨 CRITICAL ERROR: y_train contains values outside [0, 1]. Your config.toml expects num_classes = 2!")

# 2. Check Categorical Bounds
cat_sizes = info['cat_sizes']
for i, size in enumerate(cat_sizes):
    col_max = X_cat_train[:, i].max()
    col_min = X_cat_train[:, i].min()
    print(f"Cat Col {i} (Max Allowed: {size-1}): min={col_min}, max={col_max}")
    if col_max >= size or col_min < 0:
        print(f"🚨 CRITICAL ERROR: Categorical Column {i} has out-of-bounds indices!")

# 3. Check Numerical Integrity
if np.isnan(X_num_train).any() or np.isinf(X_num_train).any():
    print("🚨 CRITICAL ERROR: X_num_train contains NaNs or Infs! This will crash the MLP.")
else:
    print("Numerical data is clean.")
