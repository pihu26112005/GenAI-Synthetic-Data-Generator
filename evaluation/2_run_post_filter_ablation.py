import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult', help='Dataset name')
parser.add_argument('--target_column', type=str, default='income', help='Target column (e.g., income, label)')
args = parser.parse_args()

dataname = args.dataname
target_column = args.target_column # Change this to the actual name of your target column (e.g., 'income')

# 1. Load the REAL dataset (to teach the Random Forest what reality looks like)
# Adjust this path to wherever your raw/processed real training CSV is stored
real_data_path = f"data/{dataname}/train.csv" 
real_df = pd.read_csv(real_data_path)

# 2. Load the SYNTHETIC dataset from Model 1 (Vanilla)
vanilla_csv_path = f"evaluation/ablation/ablation_1_vanilla/{dataname}/synthetic.csv"
synth_df = pd.read_csv(vanilla_csv_path)

print(f"Loaded Real Data: {len(real_df)} rows")
print(f"Loaded Vanilla Synthetic Data: {len(synth_df)} rows")

# Prepare Real Data for Training
# Assuming categorical columns are already encoded, or use basic drop/dummies for the RF
X_real = real_df.drop(columns=[target_column])
y_real = real_df[target_column]

X_synth = synth_df.drop(columns=[target_column])
y_synth_actual = synth_df[target_column] # The labels the diffusion model assigned

# 3. Train the "Oracle" Random Forest on Real Data
print("\nTraining Random Forest on Real Data...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_real, y_real)

# 4. Predict probabilities on the Synthetic Data
print("Evaluating Synthetic Data...")
# predict_proba returns the probability for each class [prob_0, prob_1, prob_2]
probabilities = rf.predict_proba(X_synth)

# Get the highest probability for each row (the model's "confidence" in its prediction)
max_confidences = np.max(probabilities, axis=1)

# 5. The Rejection Filter
# We delete points where the model's confidence is below a threshold (e.g., 0.65)
# This means the point is mathematically floating in the "overlap" zone between classes
confidence_threshold = 0.65 

# We also want to keep rows where the diffusion model's generated label matches 
# what the RF thinks the label should be.
predicted_labels = rf.predict(X_synth)
label_match = (predicted_labels == y_synth_actual)
high_confidence = (max_confidences >= confidence_threshold)

# The point survives ONLY if it matches the label AND is high confidence
survivor_mask = label_match & high_confidence

filtered_synth_df = synth_df[survivor_mask]

# 6. Save the results
dropped_count = len(synth_df) - len(filtered_synth_df)
print(f"\n{'='*50}")
print(f"Filter Complete!")
print(f"Original Vanilla Rows: {len(synth_df)}")
print(f"Rows Deleted (Overlap Garbage): {dropped_count} ({(dropped_count/len(synth_df))*100:.1f}%)")
print(f"Final Cleaned Rows: {len(filtered_synth_df)}")
print(f"{'='*50}")

output_path = f"evaluation/ablation/ablation_2_band_aid/{dataname}/synthetic.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
filtered_synth_df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
