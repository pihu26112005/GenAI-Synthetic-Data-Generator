import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# --- THE TRUE MATHEMATICAL ORACLE ---
# We must redefine the exact math used to generate the dataset
maj_samples = 7500
min_samples = 200

maj_centers = [[10,10], [-30,-20], [40,-50]]
min_centers = [[-30,10], [20,-20]]
maj_cov = [[[100,0],[0,100]], [[64,0],[0,64]], [[16,0],[0,16]]]
min_cov = [[[100,0],[0,100]], [[16,0],[0,16]]]

# Recreate the exact GMMs
gmm0 = GaussianMixture(n_components=3, random_state=42)
gmm0.means_, gmm0.covariances_, gmm0.weights_ = np.array(maj_centers), np.array(maj_cov), np.array([1/3]*3)

gmm1 = GaussianMixture(n_components=2, random_state=42)
gmm1.means_, gmm1.covariances_, gmm1.weights_ = np.array(min_centers), np.array(min_cov), np.array([1/2]*2)

def compute_gmm_probability(points, weights, means, covariances):
    prob = np.zeros(len(points))
    for w, m, c in zip(weights, means, covariances):
        prob += w * multivariate_normal(mean=m, cov=c).pdf(points)
    return prob

def compute_bayesian_accuracy(df):
    """Calculates Oracle Accuracy for the generated points."""
    prior0 = maj_samples / (maj_samples + min_samples)
    prior1 = min_samples / (maj_samples + min_samples)
    
    points = df[['f1', 'f2']].values
    
    prob0 = compute_gmm_probability(points, gmm0.weights_, gmm0.means_, gmm0.covariances_) * prior0
    prob1 = compute_gmm_probability(points, gmm1.weights_, gmm1.means_, gmm1.covariances_) * prior1

    # Mathematical truth: which class SHOULD this point belong to?
    predicted_class = (prob1 > prob0).astype(int)
    
    # Identify errors
    cond_maj_wrong = (df['class'] == 0) & (predicted_class == 1)
    cond_min_wrong = (df['class'] == 1) & (predicted_class == 0)

    maj_count = len(df[df['class'] == 0])
    min_count = len(df[df['class'] == 1])

    # Avoid division by zero if a model totally fails to generate a class
    maj_acc = 1 - (cond_maj_wrong.sum() / maj_count) if maj_count > 0 else 0
    min_acc = 1 - (cond_min_wrong.sum() / min_count) if min_count > 0 else 0
    total_acc = 1 - ((cond_maj_wrong.sum() + cond_min_wrong.sum()) / len(df))

    return total_acc, maj_acc, min_acc

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    model_configs = {
        "1. Vanilla": os.path.join(base_dir, "ablation", "ablation_1_vanilla", "toy", "synthetic.csv"),
        "2. Post-Hoc": os.path.join(base_dir, "ablation", "ablation_2_band_aid", "toy", "synthetic.csv"),
        "3. MMD Only": os.path.join(base_dir, "ablation", "ablation_3_mmd_only", "toy", "synthetic.csv"),
        "4. Triplet Only": os.path.join(base_dir, "ablation", "ablation_4_triplet_only", "toy", "synthetic.csv"),
        "5. Champion": os.path.join(base_dir, "ablation", "ablation_5_final", "toy", "synthetic.csv"),
        # "TabDDPM Baseline": os.path.join(base_dir, "baselines", "tabddpm", "toy", "synthetic.csv")
    }

    results = []
    plot_data = []

    print("=== BAYESIAN ORACLE EVALUATION ===")
    
    for name, path in model_configs.items():
        if not os.path.exists(path):
            print(f"  [!] Missing file: {path}")
            continue
            
        syn_df = pd.read_csv(path)
        
        # Standardize target to 'class' (0 and 1)
        if 'cond' in syn_df.columns:
            syn_df = syn_df[syn_df['cond'] != 1].reset_index(drop=True) # Drop overlap
            syn_df['class'] = syn_df['cond'].replace({2: 1})            # Remap minority
        elif 'target' in syn_df.columns:
            syn_df = syn_df.rename(columns={'target': 'class'})
            if 2 in syn_df['class'].values:
                syn_df = syn_df[syn_df['class'] != 1].reset_index(drop=True)
                syn_df['class'] = syn_df['class'].replace({2: 1})
        
        # Store for plotting
        plot_data.append((name, syn_df))
        
        # Compute Math Accuracy
        total_acc, maj_acc, min_acc = compute_bayesian_accuracy(syn_df)
        
        results.append({
            "Model": name,
            "Total Math Acc": round(total_acc * 100, 2),
            "Majority Math Acc": round(maj_acc * 100, 2),
            "Minority Math Acc (Recall)": round(min_acc * 100, 2)
        })

    # Print Table
    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print(" BAYESIAN ORACLE METRICS (2D TOY DATASET)")
    print("="*70)
    print(results_df.to_markdown(index=False))
    
    # --- VISUALIZATION: THE 2D SCATTER PLOTS ---
    print("\nGenerating 2D Scatter Visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (name, df) in enumerate(plot_data):
        ax = axes[idx]
        # Plot majority (0) in blue, minority (1) in orange
        sns.scatterplot(data=df, x='f1', y='f2', hue='class', palette='Set1', alpha=0.6, ax=ax, s=15)
        ax.set_title(f"{name}\nMin Acc: {results_df.iloc[idx]['Minority Math Acc (Recall)']}%")
        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 50)
        
    plt.tight_layout()
    plot_path = os.path.join(base_dir, "toy_2d_distributions.png")
    plt.savefig(plot_path)
    print(f"Saved visualization to: {plot_path}")

if __name__ == "__main__":
    main()