import os
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

warnings.simplefilter(action='ignore', category=Warning)

def load_and_standardize_data(filepath, expected_target):
    """Loads CSV, standardizes target column, and removes ORD overlap class."""
    df = pd.read_csv(filepath)
    if 'cond' in df.columns and expected_target not in df.columns:
        df = df.rename(columns={'cond': expected_target})
        
    if expected_target in df.columns and 2 in df[expected_target].values:
        df = df[df[expected_target] != 1].reset_index(drop=True)
        df[expected_target] = df[expected_target].replace({2: 1})
            
    return df

def preprocess_for_distances(real_df, syn_df):
    """Encodes categoricals and scales all data to [0, 1] for fair Euclidean distance."""
    real_encoded = real_df.copy()
    syn_encoded = syn_df.copy()
    
    # 1. Encode Categorical Columns safely
    for col in real_df.columns:
        if real_df[col].dtype == 'object' or real_df[col].dtype.name == 'category':
            le = LabelEncoder()
            # Fit on real data, transform both
            real_encoded[col] = le.fit_transform(real_encoded[col].astype(str))
            
            # Handle any unseen categories in synthetic data safely
            syn_encoded[col] = syn_encoded[col].astype(str).map(
                lambda s: s if s in le.classes_ else '<unknown>'
            )
            # Add <unknown> to classes to prevent transform crash
            le.classes_ = np.append(le.classes_, '<unknown>')
            syn_encoded[col] = le.transform(syn_encoded[col])
            
    # 2. Scale all columns to [0, 1] so distances aren't dominated by large numbers
    scaler = MinMaxScaler()
    real_scaled = pd.DataFrame(scaler.fit_transform(real_encoded), columns=real_encoded.columns)
    syn_scaled = pd.DataFrame(scaler.transform(syn_encoded), columns=syn_encoded.columns)
    
    return real_scaled, syn_scaled

def compute_privacy_metrics(real_df, syn_df):
    """Calculates the 5th percentile of DCR and NNDR."""
    # Fit Nearest Neighbors on the Real Training Data
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1)
    nn.fit(real_df)
    
    # Query distances from Synthetic Data to Real Data
    distances, _ = nn.kneighbors(syn_df)
    
    # Distance to the 1st closest real record
    dcr = distances[:, 0]
    
    # Ratio: (Distance to 1st) / (Distance to 2nd)
    # Add a tiny epsilon to prevent division by zero if 2nd neighbor sits on the exact same spot
    nndr = distances[:, 0] / (distances[:, 1] + 1e-10)
    
    # We report the 5th percentile (the highest risk cases)
    dcr_5th = np.percentile(dcr, 5)
    nndr_5th = np.percentile(nndr, 5)
    
    return dcr_5th, nndr_5th

def main():
    parser = argparse.ArgumentParser(description="Evaluate Privacy and Memorization Metrics")
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--target', type=str, default='income')
    parser.add_argument('--train_path', type=str, default='../data/adult/train.csv')
    args = parser.parse_args()

    print(f"Loading Real Training Data ({args.train_path})...")
    real_df = load_and_standardize_data(args.train_path, args.target)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_configs = {
        "1. Vanilla": os.path.join(base_dir, "ablation", "ablation_1_vanilla", args.dataset, "synthetic.csv"),
        "2. Post-Hoc": os.path.join(base_dir, "ablation", "ablation_2_band_aid", args.dataset, "synthetic.csv"),
        "3. MMD Only": os.path.join(base_dir, "ablation", "ablation_3_mmd_only", args.dataset, "synthetic.csv"),
        "4. Triplet Only": os.path.join(base_dir, "ablation", "ablation_4_triplet_only", args.dataset, "synthetic.csv"),
        "5. Champion": os.path.join(base_dir, "ablation", "ablation_5_final", args.dataset, "synthetic.csv"),
        "TabDDPM Baseline": os.path.join(base_dir, "baselines", "tabddpm", args.dataset, "synthetic.csv")
    }

    results = []

    for model_name, syn_path in model_configs.items():
        if not os.path.exists(syn_path):
            print(f"  [!] Missing file for {model_name}: looked in {syn_path}")
            continue
        
        print(f"Evaluating: {model_name}")
        syn_df = load_and_standardize_data(syn_path, args.target)
        
        # Align columns to guarantee exact match
        syn_df = syn_df[real_df.columns]
        
        # Scale and encode data
        real_scaled, syn_scaled = preprocess_for_distances(real_df, syn_df)

        # Compute DCR and NNDR
        dcr_5th, nndr_5th = compute_privacy_metrics(real_scaled, syn_scaled)
        
        results.append({
            "Model": model_name,
            "5th% DCR (↑)": round(dcr_5th, 4),
            "5th% NNDR (↑)": round(nndr_5th, 4)
        })

    if results:
        print("\n" + "="*65)
        print(f" PRIVACY & MEMORIZATION REPORT - Dataset: {args.dataset.upper()}")
        print("="*65)
        results_df = pd.DataFrame(results)
        print(results_df.to_markdown(index=False))
        
        # Save results
        out_csv = os.path.join(base_dir, f"privacy_results_{args.dataset}.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"\nSaved results to {out_csv}")
    else:
        print("\n[!] No models were evaluated. Check the file paths.")

if __name__ == "__main__":
    main()