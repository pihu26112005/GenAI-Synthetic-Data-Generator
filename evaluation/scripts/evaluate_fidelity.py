import os
import argparse
import warnings
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=Warning)

from sdmetrics.reports.single_table import QualityReport

def load_and_standardize_data(filepath, expected_target):
    df = pd.read_csv(filepath)
    if 'cond' in df.columns and expected_target not in df.columns:
        df = df.rename(columns={'cond': expected_target})
        
    if expected_target in df.columns and 2 in df[expected_target].values:
        df = df[df[expected_target] != 1].reset_index(drop=True)
        df[expected_target] = df[expected_target].replace({2: 1})
            
    return df

def match_class_distribution(real_df, syn_df, target_col):
    """Subsamples synthetic data to perfectly match the real training data's class ratio.
       This prevents false-positive fidelity errors caused by class imbalance shifts."""
    if target_col not in syn_df.columns:
        return syn_df
        
    matched_syn = []
    for class_label in real_df[target_col].unique():
        real_count = len(real_df[real_df[target_col] == class_label])
        syn_class_data = syn_df[syn_df[target_col] == class_label]
        
        if len(syn_class_data) > 0:
            # Sample with replacement if we need more synthetic points than generated
            sampled = syn_class_data.sample(n=real_count, replace=True, random_state=42)
            matched_syn.append(sampled)
            
    if matched_syn:
        return pd.concat(matched_syn).sample(frac=1, random_state=42).reset_index(drop=True)
    return syn_df

def find_meta_data(df, target_col, UNIQ_THRESHOLD=20):
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
    int_cols = [col for col in df.columns if df[col].dtype == 'int64']
    float_cols = [col for col in df.columns if df[col].dtype == 'float64']
    
    disc_int_cols = [col for col in int_cols if df[col].nunique() < UNIQ_THRESHOLD]
    disc_float_cols = [col for col in float_cols if df[col].nunique() < UNIQ_THRESHOLD]
    
    discrete_cols = cat_cols + disc_int_cols + disc_float_cols
    conti_cols = [col for col in int_cols + float_cols if col not in discrete_cols]

    metadata = {'columns': {}}
    for col in conti_cols:
        metadata['columns'][col] = {"sdtype": "numerical"}
    for col in discrete_cols:
        metadata['columns'][col] = {"sdtype": "categorical"}
        
    if target_col in metadata['columns']:
        metadata['columns'][target_col] = {"sdtype": "categorical"}
        
    return metadata, conti_cols, discrete_cols

def compute_math_fidelity(real_df, syn_df, conti_cols):
    # 1. Wasserstein Distance (FIXED: Scaled to [0,1] so large numbers don't dominate)
    scaler = MinMaxScaler()
    real_scaled = pd.DataFrame(scaler.fit_transform(real_df[conti_cols]), columns=conti_cols)
    syn_scaled = pd.DataFrame(scaler.transform(syn_df[conti_cols]), columns=conti_cols)
    
    wd_scores = []
    for col in conti_cols:
        wd = wasserstein_distance(real_scaled[col], syn_scaled[col])
        wd_scores.append(wd)
    avg_wd = np.mean(wd_scores) if wd_scores else 0

    # 2. Jensen-Shannon Divergence (FIXED: True probabilities that sum to 1)
    jsd_scores = []
    for col in conti_cols:
        min_val = min(real_df[col].min(), syn_df[col].min())
        max_val = max(real_df[col].max(), syn_df[col].max())
        bins = np.linspace(min_val, max_val, 21)
        
        p, _ = np.histogram(real_df[col], bins=bins)
        q, _ = np.histogram(syn_df[col], bins=bins)
        
        # Normalize to probability mass function
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        
        # Add epsilon to prevent log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        jsd_scores.append(jensenshannon(p, q))
    avg_jsd = np.mean(jsd_scores) if jsd_scores else 0

    # 3. Pairwise Correlation Error (FIXED: Only applied to mathematically continuous columns)
    real_corr = real_df[conti_cols].corr().fillna(0).values
    syn_corr = syn_df[conti_cols].corr().fillna(0).values
    corr_error = np.linalg.norm(real_corr - syn_corr)

    return avg_wd, avg_jsd, corr_error

def main():
    parser = argparse.ArgumentParser(description="Evaluate Statistical Fidelity")
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--target', type=str, default='income')
    parser.add_argument('--train_path', type=str, default='../data/adult/train.csv')
    args = parser.parse_args()

    print(f"Loading Real Data ({args.train_path})...")
    real_df = load_and_standardize_data(args.train_path, args.target)
    metadata_dict, conti_cols, _ = find_meta_data(real_df, args.target)

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
        
        # Align Columns
        syn_df = syn_df[real_df.columns]
        
        # PERFECT CLASS DISTRIBUTION MATCHING (Crucial for SDMetrics)
        syn_df = match_class_distribution(real_df, syn_df, args.target)

        # 1. Compute Math Metrics
        avg_wd, avg_jsd, corr_error = compute_math_fidelity(real_df, syn_df, conti_cols)
        
        # 2. Compute SDMetrics
        report = QualityReport()
        report.generate(real_df, syn_df, metadata_dict, verbose=False)
        properties = report.get_properties()
        
        col_shape_score = properties[properties['Property'] == 'Column Shapes']['Score'].values[0]
        col_pair_score = properties[properties['Property'] == 'Column Pair Trends']['Score'].values[0]

        results.append({
            "Model": model_name,
            "Wasserstein (↓)": round(avg_wd, 4),
            "JSD (↓)": round(avg_jsd, 4),
            "Corr Error (↓)": round(corr_error, 4),
            "SD Col Shape (↑)": round(col_shape_score * 100, 2),
            "SD Pair Trend (↑)": round(col_pair_score * 100, 2)
        })

    if results:
        print("\n" + "="*85)
        print(f"STATISTICAL FIDELITY REPORT - Dataset: {args.dataset.upper()}")
        print("="*85)
        results_df = pd.DataFrame(results)
        print(results_df.to_markdown(index=False))
        results_df.to_csv(os.path.join(base_dir, f"fidelity_results_{args.dataset}.csv"), index=False)
    else:
        print("\n[!] No models were evaluated. Please check the file paths printed above.")

if __name__ == "__main__":
    main()