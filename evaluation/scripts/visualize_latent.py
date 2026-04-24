import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.simplefilter(action='ignore', category=Warning)

def main():
    parser = argparse.ArgumentParser(description="Visualize Latent Space (train_z.npy)")
    parser.add_argument('--dataset', type=str, default='adult', help="Dataset name (e.g., adult, toy)")
    parser.add_argument('--method', type=str, choices=['pca', 'tsne'], default='pca', 
                        help="Dimensionality reduction method: 'pca' (fast) or 'tsne' (better for clusters)")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Load the True Labels 
    labels_path = os.path.join(base_dir, f"../data/{args.dataset}/y_train.npy")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Could not find training labels at {labels_path}")
    
    y_train = np.load(labels_path).flatten()
    
    # Map labels to their meaning based on ORD formulation
    label_map = {0: "Majority (0)", 1: "Overlap (1)", 2: "Minority (2)"}
    y_mapped = [label_map.get(int(lbl), str(lbl)) for lbl in y_train]

    # The 4 models that actually have distinct latent spaces
    model_configs = {
        "1. Vanilla (No Regularization)": os.path.join(base_dir, f"ablation/ablation_1_vanilla/{args.dataset}/train_z.npy"),
        "2. MMD Only (Dense but Mixed)": os.path.join(base_dir, f"ablation/ablation_3_mmd_only/{args.dataset}/train_z.npy"),
        "3. Triplet Only (Separated but Shattered)": os.path.join(base_dir, f"ablation/ablation_4_triplet_only/{args.dataset}/train_z.npy"),
        "4. Champion (Separated & Dense)": os.path.join(base_dir, f"ablation/ablation_5_final/{args.dataset}/train_z.npy")
    }

    plot_data = []
    
    print(f"Loading Latent Spaces for '{args.dataset}' and applying {args.method.upper()}...")

    for name, path in model_configs.items():
        if not os.path.exists(path):
            print(f"  [!] Missing latent file: {path}")
            continue
            
        print(f"  Processing {name}...")
        z = np.load(path)
        
        # --- FIX 1: Strip the CLS token exactly like CTabSyn's latent_utils.py does ---
        if len(z.shape) == 3:
            z = z[:, 1:, :] 
            z = z.reshape(z.shape[0], -1)
            
        # --- FIX 2: Safely handle VAE batch padding length mismatches ---
        min_len = min(len(y_mapped), z.shape[0])
        if z.shape[0] != len(y_mapped):
            print(f"      -> Warning: VAE padding detected. Labels ({len(y_mapped)}) != Latents ({z.shape[0]}). Safely truncating to {min_len}.")
            
        z_clean = z[:min_len]
        current_labels = y_mapped[:min_len]
        
        # Apply Dimensionality Reduction
        if args.method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            # t-SNE takes a while, so we print a status update
            print("      -> Running t-SNE (This may take a few minutes)...")
            reducer = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=30)
            
        z_2d = reducer.fit_transform(z_clean)
        
        df_z = pd.DataFrame(z_2d, columns=['Component 1', 'Component 2'])
        df_z['Class'] = current_labels
        plot_data.append((name, df_z))

    if not plot_data:
        print("No valid latent files found to plot. Exiting.")
        return

    # Generate 2x2 Grid Plot
    print("\nGenerating Plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    palette = {"Majority (0)": "#3498db", "Overlap (1)": "#e74c3c", "Minority (2)": "#2ecc71"}
    
    for idx, (name, df_z) in enumerate(plot_data):
        ax = axes[idx]
        sns.scatterplot(data=df_z, x='Component 1', y='Component 2', 
                        hue='Class', palette=palette, alpha=0.5, s=10, ax=ax, edgecolor=None)
        
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        if idx == 0:
            ax.legend(title="ORD Class", loc='best')
        else:
            ax.get_legend().remove()

    plt.suptitle(f"Latent Space ({args.method.upper()}) Analysis - {args.dataset.upper()}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_file = os.path.join(base_dir, f"latent_space_{args.method}_{args.dataset}.png")
    plt.savefig(out_file, dpi=300)
    print(f"✅ Saved visualization to: {out_file}")

if __name__ == "__main__":
    main()