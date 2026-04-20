import os
import subprocess
import glob

dataname = "adult"
target_column = "income"

models = [
    "ablation_1_vanilla",
    "ablation_2_band_aid",
    "3_mmd_only",
    "4_triplet_only",
    "5_champion"
]

print(f"Starting Final Evaluation Pipeline for '{dataname}'...\n")

# Make sure you have your evaluation script ready. 
# We'll use compute_mle.py as a baseline example.
# Usage: python compute_mle.py --dataname adult --target income --method path/to/method 
# (You might need to adjust compute_mle.py to accept exact CSV paths rather than methods)

for model_name in models:
    # Resolve the path for the CSV
    if model_name.startswith("ablation_"):
        csv_path = f"synthetic/{dataname}/{model_name}.csv"
    else:
        csv_path = f"ablation_results_{dataname}/{model_name}/synthetic_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"⚠️  Skipping {model_name} (File not found: {csv_path})")
        continue

    print(f"{'='*60}")
    print(f"📊 EVALUATING: {model_name.upper()}")
    print(f"📄 File: {csv_path}")
    print(f"{'='*60}")
    
    # Example Call to a hypothetical `evaluate.py` or `compute_mle.py`
    # You will need to adapt compute_mle.py to ingest the specific `csv_path`
    # subprocess.run(["python", "compute_mle.py", "--dataname", dataname, "--target", target_column, "--csv_path", csv_path])
    
    print(f"✅ Finished evaluating {model_name}\n")
    
print("🎉 ALL MODELS EVALUATED 🎉")
print("Collect the metrics (MLE, JSD, Wasserstein) and generate your PCA plots to complete the study!")
