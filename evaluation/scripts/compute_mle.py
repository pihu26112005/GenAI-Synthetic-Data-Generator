import os
import argparse
import warnings
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=Warning)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

import sdmetrics
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import (BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, 
                                    BinaryLogisticRegression, BinaryMLPClassifier)

def find_meta_data(df, UNIQ_THRESHOLD=20):
    """Segregates columns to create metadata required by SDMetrics."""
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
        
    return metadata

def evaluate_xgboost(X_train, y_train, X_test, y_test):
    """Trains XGBoost and computes Macro-Acc, Minority Recall, F1, and AUC-ROC."""
    # Convert object types to category for XGBoost's native categorical handling
    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    for col in X_train_xgb.columns:
        if X_train_xgb[col].dtype == 'object':
            X_train_xgb[col] = X_train_xgb[col].astype('category')
            X_test_xgb[col] = X_test_xgb[col].astype('category')

    # Train XGBoost
    model = XGBClassifier(enable_categorical=True, tree_method='hist', random_state=42, eval_metric='logloss')
    model.fit(X_train_xgb, y_train)
    
    # Predict
    y_pred = model.predict(X_test_xgb)
    y_prob = model.predict_proba(X_test_xgb)[:, 1]

    # Metrics
    min_recall = recall_score(y_test, y_pred, pos_label=1)
    maj_recall = recall_score(y_test, y_pred, pos_label=0)
    macro_acc = (min_recall + maj_recall) / 2.0 # Macro-Average Accuracy
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return {
        "XGB_Macro_Acc": macro_acc,
        "XGB_Min_Recall": min_recall,
        "XGB_F1": f1,
        "XGB_AUC": auc
    }

def evaluate_sdmetrics(train_df, test_df, target, metadata):
    """Computes average efficacy across 4 standard SDMetrics classifiers."""
    try:
        ada = BinaryAdaBoostClassifier.compute(test_data=test_df, train_data=train_df, target=target, metadata=metadata)
        dt = BinaryDecisionTreeClassifier.compute(test_data=test_df, train_data=train_df, target=target, metadata=metadata)
        lr = BinaryLogisticRegression.compute(test_data=test_df, train_data=train_df, target=target, metadata=metadata)
        mlp = BinaryMLPClassifier.compute(test_data=test_df, train_data=train_df, target=target, metadata=metadata)
        return (ada + dt + lr + mlp) / 4.0
    except Exception as e:
        print(f"  [!] SDMetrics Evaluation Failed: {e}")
        return np.nan
    
def load_and_standardize_data(filepath, expected_target):
    """Loads CSV, standardizes the target column name, and removes ORD overlap class."""
    df = pd.read_csv(filepath)
    
    # 1. Standardize column name (rename 'cond' to 'expected_target')
    if 'cond' in df.columns and expected_target not in df.columns:
        df = df.rename(columns={'cond': expected_target})
        
    # 2. Handle ORD Ternary Labels (0 = Majority, 1 = Overlap, 2 = Minority)
    if expected_target in df.columns:
        # Check if the ternary structure exists in this dataset
        if 2 in df[expected_target].values:
            original_size = len(df)
            
            # Step A: Drop the Overlap class (which is now 1)
            df = df[df[expected_target] != 1].reset_index(drop=True)
            print(f"    -> Dropped {original_size - len(df)} Overlap (Class 1) points.")
            
            # Step B: Remap the Minority class (2) back to standard binary (1)
            # This ensures compatibility with XGBoost, SDMetrics, and test.csv
            df[expected_target] = df[expected_target].replace({2: 1})
            print("    -> Remapped Minority class from 2 to 1 for binary evaluation.")
            
    return df

def balance_synthetic_data(synth_df, target_col):
    """Balances the synthetic data by downsampling the majority class to match the minority class."""
    if 1 not in synth_df[target_col].values or 0 not in synth_df[target_col].values:
        return synth_df # Cannot balance if a class is entirely missing
        
    synth_minority = synth_df[synth_df[target_col] == 1]
    synth_majority = synth_df[synth_df[target_col] == 0]
    
    # Sample majority to match minority count
    synth_majority_sampled = synth_majority.sample(n=len(synth_minority), random_state=42, replace=True)
    
    balanced_train = pd.concat([synth_minority, synth_majority_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_train

def main():
    parser = argparse.ArgumentParser(description="Evaluate Machine Learning Efficacy for Synthetic Tabular Data")
    parser.add_argument('--dataset', type=str, default='adult', help="Name of the dataset (e.g., adult)")
    parser.add_argument('--target', type=str, default='income', help="Name of the target column")
    parser.add_argument('--test_path', type=str, default='../data/adult/test.csv', help="Path to the real holdout test set")
    args = parser.parse_args()

    # Load Real Test Data
    print(f"Loading Real Test Data from {args.test_path}...")
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Test file not found at {args.test_path}")
    
    # test_df = pd.read_csv(args.test_path)
    test_df = load_and_standardize_data(args.test_path, args.target)
    X_test = test_df.drop(columns=[args.target])
    y_test = test_df[args.target]

    # Define paths dynamically based on the tree structure
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    model_configs = {
        "1. Vanilla (No Triplet/MMD)": os.path.join(base_dir, "ablation", "ablation_1_vanilla", args.dataset, "synthetic.csv"),
        "2. Post-Hoc Filter (Band-Aid)": os.path.join(base_dir, "ablation", "ablation_2_band_aid", args.dataset, "synthetic.csv"),
        "3. MMD Only": os.path.join(base_dir, "ablation", "ablation_3_mmd_only", args.dataset, "synthetic.csv"),
        "4. Triplet Only": os.path.join(base_dir, "ablation", "ablation_4_triplet_only", args.dataset, "synthetic.csv"),
        "5. Champion (Triplet+MMD)": os.path.join(base_dir, "ablation", "ablation_5_final", "synthetic.csv"), # Assuming it's at the root of folder 5
        "Baseline (TabDDPM)": os.path.join(base_dir, "baselines", "tabddpm", args.dataset, "synthetic.csv")
    }

    results = []

    for model_name, syn_path in model_configs.items():
        print(f"\nEvaluating: {model_name}")
        if not os.path.exists(syn_path):
            print(f"  [!] Missing file: {syn_path}. Skipping.")
            continue
        
        # Load synthetic data
        # synth_df = pd.read_csv(syn_path)
        synth_df = load_and_standardize_data(syn_path, args.target)
        
        # Step 1: Balance the synthetic data for training
        train_df = balance_synthetic_data(synth_df, args.target)
        X_train = train_df.drop(columns=[args.target])
        y_train = train_df[args.target]

        # Step 2: Evaluate using XGBoost (Faster, precise metrics requested)
        xgb_metrics = evaluate_xgboost(X_train, y_train, X_test, y_test)
        
        # Step 3: Evaluate using SDMetrics (Average of 4 classifiers)
        metadata = find_meta_data(train_df)
        sd_avg = evaluate_sdmetrics(train_df, test_df, args.target, metadata)
        
        # Append to results
        results.append({
            "Model": model_name,
            "XGB_Macro_Acc": xgb_metrics["XGB_Macro_Acc"],
            "XGB_Min_Recall": xgb_metrics["XGB_Min_Recall"],
            "XGB_F1": xgb_metrics["XGB_F1"],
            "XGB_AUC_ROC": xgb_metrics["XGB_AUC"],
            "SDMetrics_Avg_MLE": sd_avg
        })

    # Print Final Summary Table
    print("\n" + "="*80)
    print(f"MACHINE LEARNING EFFICACY (MLE) REPORT - Dataset: {args.dataset.upper()}")
    print("="*80)
    results_df = pd.DataFrame(results)
    # Format percentages for readability
    for col in results_df.columns[1:]:
        results_df[col] = (results_df[col] * 100).round(2).astype(str) + '%'
    
    print(results_df.to_markdown(index=False))
    
    # Save to CSV
    out_csv = os.path.join(base_dir, f"mle_results_{args.dataset}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

if __name__ == "__main__":
    main()