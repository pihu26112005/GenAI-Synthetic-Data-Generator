import os
import json
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# --- CONFIGURATION ---
DATASET_NAME = 'toy'
DATA_DIR = f"data/{DATASET_NAME}"
INFO_DIR = "data/Info"

maj_centers = [[10,10], [-30,-20], [40,-50]]
min_centers = [[-30,10], [20,-20]]
maj_cov = [[[100,0],[0,100]], [[64,0],[0,64]], [[16,0],[0,16]]]
min_cov = [[[100,0],[0,100]], [[16,0],[0,16]]]
maj_std = [10, 8, 4]
min_std = [10, 4]

maj_samples = 7500
min_samples = 200

def compute_gmm_probability(x, mean, cov):
    return multivariate_normal(mean=mean, cov=cov).pdf(x)

def get_gmm_probs(points, weights, means, covariances):
    prob = np.zeros(len(points))
    for w, m, c in zip(weights, means, covariances):
        prob += w * compute_gmm_probability(points, m, c)
    return prob

def main():
    print(f"Generating '{DATASET_NAME}' dataset...")
    
    # 1. Generate Blobs
    X_maj, _ = make_blobs(n_samples=maj_samples, centers=maj_centers, cluster_std=maj_std, random_state=42)
    X_min, _ = make_blobs(n_samples=min_samples, centers=min_centers, cluster_std=min_std, random_state=42)
    
    df_maj = pd.DataFrame(X_maj, columns=['f1', 'f2'])
    df_maj['target'] = 0
    
    df_min = pd.DataFrame(X_min, columns=['f1', 'f2'])
    df_min['target'] = 1
    
    df_all = pd.concat([df_maj, df_min], axis=0).reset_index(drop=True)

    # 2. Fit GMMs to define the True Mathematical Overlap
    gmm0 = GaussianMixture(n_components=3, random_state=42)
    gmm0.means_, gmm0.covariances_, gmm0.weights_ = np.array(maj_centers), np.array(maj_cov), np.array([1/3]*3)

    gmm1 = GaussianMixture(n_components=2, random_state=42)
    gmm1.means_, gmm1.covariances_, gmm1.weights_ = np.array(min_centers), np.array(min_cov), np.array([1/2]*2)

    prior0 = maj_samples / (maj_samples + min_samples)
    prior1 = min_samples / (maj_samples + min_samples)

    points = df_all[['f1', 'f2']].values
    prob0 = get_gmm_probs(points, gmm0.weights_, gmm0.means_, gmm0.covariances_) * prior0
    prob1 = get_gmm_probs(points, gmm1.weights_, gmm1.means_, gmm1.covariances_) * prior1

    # 3. Apply ORD Logic (0 = Majority, 1 = Overlap, 2 = Minority)
    predicted_majority = (prob0 >= prob1).astype(int)
    
    # Initialize cond with the base targets
    df_all['cond'] = df_all['target']
    
    # If it's truly a minority point (target=1) but math says it's in the majority zone, it's OVERLAP (1)
    overlap_mask = (df_all['target'] == 1) & (predicted_majority == 1)
    
    df_all.loc[df_all['target'] == 0, 'cond'] = 0 # Clear Majority
    df_all.loc[overlap_mask, 'cond'] = 1          # Overlap
    df_all.loc[(df_all['target'] == 1) & (~overlap_mask), 'cond'] = 2 # Clear Minority
    
    print(f"Class Distribution (cond):\n{df_all['cond'].value_counts().sort_index()}")

    # 4. Create Directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INFO_DIR, exist_ok=True)

    # 5. Split and Save
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['cond'])
    
    # Base files
    train_df.to_csv(f"{DATA_DIR}/train.csv", index=False)
    test_df.to_csv(f"{DATA_DIR}/test.csv", index=False)
    
    # ORD specific files expected by your preprocess scripts
    train_df.drop(columns=['target']).to_csv(f"{DATA_DIR}/imbalanced_ord.csv", index=False)
    train_df.drop(columns=['cond']).to_csv(f"{DATA_DIR}/imbalanced_noord.csv", index=False)

    # 6. Create info.json for Tabular Diffusion
    info_dict = {
        "name": DATASET_NAME,
        "task_type": "binclass",
        "header": "infer",
        "column_info": {
            "f1": {"type": "numerical", "subtype": "float"},
            "f2": {"type": "numerical", "subtype": "float"},
            "target": {"type": "categorical", "subtype": "integer"},
            "cond": {"type": "categorical", "subtype": "integer"}
        }
    }
    
    with open(f"{INFO_DIR}/{DATASET_NAME}.json", "w") as f:
        json.dump(info_dict, f, indent=4)

    print(f"\nSuccessfully generated {DATASET_NAME} dataset in {DATA_DIR}/")
    print(f"info.json saved to {INFO_DIR}/{DATASET_NAME}.json")

if __name__ == "__main__":
    main()