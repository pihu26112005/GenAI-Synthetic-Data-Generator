import os
import argparse
import warnings
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=Warning)
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score

def load_and_standardize_data(filepath, expected_target):
    df = pd.read_csv(filepath)
    if 'cond' in df.columns and expected_target not in df.columns:
        df = df.rename(columns={'cond': expected_target})
    if expected_target in df.columns and 2 in df[expected_target].values:
        df = df[df[expected_target] != 1].reset_index(drop=True)
        df[expected_target] = df[expected_target].replace({2: 1})
    return df

def train_and_eval_xgb(train_df, test_df, target_col):
    """Trains XGBoost on a specific augmented set and evaluates on real test."""
    X_train = train_df.drop(columns=[target_col]).copy()
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col]).copy()
    y_test = test_df[target_col]

    # EXACT COLUMN ALIGNMENT (FIXED)
    X_test = X_test[X_train.columns]

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    model = XGBClassifier(enable_categorical=True, tree_method='hist', random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    min_recall = recall_score(y_test, y_pred, pos_label=1)
    macro_acc = (min_recall + recall_score(y_test, y_pred, pos_label=0)) / 2.0
    
    return macro_acc, min_recall, f1_score(y_test, y_pred), roc_auc_score(y_test, y_prob)

def main():
    parser = argparse.ArgumentParser(description="Augmentation Strategy Evaluation")
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--target', type=str, default='income')
    parser.add_argument('--train_path', type=str, default='../data/adult/train.csv')
    parser.add_argument('--test_path', type=str, default='../data/adult/test.csv')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    champ_path = os.path.join(base_dir, "ablation", "ablation_5_final", "synthetic.csv")
    
    if not os.path.exists(champ_path):
        raise FileNotFoundError(f"Champion model synthetic data not found at {champ_path}")

    # Load Data (FIXED)
    real_train = load_and_standardize_data(args.train_path, args.target)
    real_test = load_and_standardize_data(args.test_path, args.target)
    synth_champ = load_and_standardize_data(champ_path, args.target)

    # Break data into fundamental components
    D1 = real_train[real_train[args.target] == 1]
    D0 = real_train[real_train[args.target] == 0]
    
    S1 = synth_champ[synth_champ[args.target] == 1]
    S00 = synth_champ[synth_champ[args.target] == 0]

    # Equalize synthetic classes for base strategy
    S00 = S00.sample(n=len(S1), random_state=42, replace=True)

    # Build Augmentation Strategies
    strategies = {
        "1. S_00 ∪ S_1 (Pure Synthetic)": pd.concat([S00, S1]),
        "2. S_00 ∪ S_1 ∪ D_1 (Syn + Real Minority)": pd.concat([S00, S1, D1]),
        "3. S_00 ∪ S_1 ∪ D_1 ∪ D_0_sub (Full Hybrid)": pd.concat([S00, S1, D1, D0.sample(n=len(D1), random_state=42)])
    }

    results = []
    print(f"\nTesting Augmentation Strategies on Champion Model...")
    
    for strategy_name, train_data in strategies.items():
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        macro_acc, min_recall, f1, auc = train_and_eval_xgb(train_data, real_test, args.target)
        
        results.append({
            "Augmentation Strategy": strategy_name,
            "Macro Acc": macro_acc,
            "Minority Recall": min_recall,
            "F1 Score": f1,
            "AUC-ROC": auc
        })

    print("\n" + "="*80)
    print(f"AUGMENTATION STRATEGY REPORT - Dataset: {args.dataset.upper()}")
    print("="*80)
    results_df = pd.DataFrame(results)
    for col in results_df.columns[1:]:
        results_df[col] = (results_df[col] * 100).round(2).astype(str) + '%'
    
    print(results_df.to_markdown(index=False))
    results_df.to_csv(os.path.join(base_dir, f"augmentation_results_{args.dataset}.csv"), index=False)

if __name__ == "__main__":
    main()