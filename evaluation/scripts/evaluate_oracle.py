import os
import argparse
import warnings
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=Warning)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score

def load_and_standardize_data(filepath, expected_target):
    """Loads CSV, standardizes the target column name, and removes ORD overlap class."""
    df = pd.read_csv(filepath)
    if 'cond' in df.columns and expected_target not in df.columns:
        df = df.rename(columns={'cond': expected_target})
        
    if expected_target in df.columns and 2 in df[expected_target].values:
        # Drop Overlap (1) and Remap Minority (2 -> 1)
        original_size = len(df)
        df = df[df[expected_target] != 1].reset_index(drop=True)
        df[expected_target] = df[expected_target].replace({2: 1})
            
    return df

def balance_real_data(real_df, target_col):
    """Downsamples the majority class to perfectly balance the real training data."""
    minority = real_df[real_df[target_col] == 1]
    majority = real_df[real_df[target_col] == 0]
    majority_sampled = majority.sample(n=len(minority), random_state=42)
    return pd.concat([minority, majority_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="Reverse Oracle Evaluation")
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--target', type=str, default='income')
    parser.add_argument('--train_path', type=str, default='../data/adult/train.csv')
    args = parser.parse_args()

    # 1. Load and Balance Real Training Data to train the Oracle (FIXED)
    print(f"Training Oracle on Real Data ({args.train_path})...")
    real_train_df = load_and_standardize_data(args.train_path, args.target)
    balanced_real_train = balance_real_data(real_train_df, args.target)
    
    X_oracle_train = balanced_real_train.drop(columns=[args.target])
    y_oracle_train = balanced_real_train[args.target]

    # Convert object types to category
    for col in X_oracle_train.columns:
        if X_oracle_train[col].dtype == 'object':
            X_oracle_train[col] = X_oracle_train[col].astype('category')

    # Train the Oracle Model
    oracle = XGBClassifier(enable_categorical=True, tree_method='hist', random_state=42, eval_metric='logloss')
    oracle.fit(X_oracle_train, y_oracle_train)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_configs = {
        "1. Vanilla (No Triplet/MMD)": os.path.join(base_dir, "ablation", "ablation_1_vanilla", args.dataset, "synthetic.csv"),
        "2. Post-Hoc Filter (Band-Aid)": os.path.join(base_dir, "ablation", "ablation_2_band_aid", args.dataset, "synthetic.csv"),
        "3. MMD Only": os.path.join(base_dir, "ablation", "ablation_3_mmd_only", args.dataset, "synthetic.csv"),
        "4. Triplet Only": os.path.join(base_dir, "ablation", "ablation_4_triplet_only", args.dataset, "synthetic.csv"),
        "5. Champion (Triplet+MMD)": os.path.join(base_dir, "ablation", "ablation_5_final", "synthetic.csv"),
        "Baseline (TabDDPM)": os.path.join(base_dir, "baselines", "tabddpm", args.dataset, f"synthetic.csv")
    }

    results = []

    for model_name, syn_path in model_configs.items():
        if not os.path.exists(syn_path):
            continue
        
        # Load Synthetic Data
        synth_df = load_and_standardize_data(syn_path, args.target)
        X_syn = synth_df.drop(columns=[args.target])
        y_syn_true = synth_df[args.target] 
        
        # EXACT COLUMN ALIGNMENT (FIXED)
        X_syn = X_syn[X_oracle_train.columns]
        for col in X_oracle_train.columns:
            if X_oracle_train[col].dtype.name == 'category':
                X_syn[col] = X_syn[col].astype('category')

        # Oracle predicts on Synthetic Data
        y_syn_pred = oracle.predict(X_syn)
        
        overall_acc = accuracy_score(y_syn_true, y_syn_pred)
        min_acc = recall_score(y_syn_true, y_syn_pred, pos_label=1)
        maj_acc = recall_score(y_syn_true, y_syn_pred, pos_label=0)

        results.append({
            "Model": model_name,
            "Oracle Overall Acc (↑)": overall_acc,
            "Oracle Minority Acc (↑)": min_acc,
            "Oracle Majority Acc (↑)": maj_acc
        })

    print("\n" + "="*80)
    print(f"REVERSE CLASSIFICATION (ORACLE) REPORT - Dataset: {args.dataset.upper()}")
    print("="*80)
    results_df = pd.DataFrame(results)
    for col in results_df.columns[1:]:
        results_df[col] = (results_df[col] * 100).round(2).astype(str) + '%'
    
    print(results_df.to_markdown(index=False))
    results_df.to_csv(os.path.join(base_dir, f"oracle_results_{args.dataset}.csv"), index=False)

if __name__ == "__main__":
    main()