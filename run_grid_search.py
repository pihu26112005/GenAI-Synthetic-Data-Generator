import os
import subprocess
import itertools
import re
import csv

# Define the hyperparameter grid
# alphas = [0.1, 0.5, 1.0, 2.5, 5.0]
# betas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
alphas = [0.5, 1.0, 2.0]
betas = [0.1, 0.5, 1.0]

dataname = "adult"
gpu = 0

# Set up directories
ckpt_dir = f"ctabsyn/tabsyn/vae/ckpt/{dataname}"
log_dir = "logs"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

combinations = list(itertools.product(alphas, betas))
print(f"Starting Grid Search. Total configurations: {len(combinations)}\n")

custom_env = os.environ.copy()
custom_env["PYTHONPATH"] = os.path.abspath("ctabsyn") + os.pathsep + custom_env.get("PYTHONPATH", "")

# List to store our extracted metrics
all_results = []

for alpha, beta in combinations:
    print(f"{'='*50}")
    print(f"Running VAE: alpha={alpha}, beta={beta}")
    
    command = [
        "python", "ctabsyn/tabsyn/vae/main.py",
        "--dataname", dataname,
        "--gpu", str(gpu),
        "--alpha", str(alpha),
        "--beta", str(beta)
    ]
    
    # 1. CAPTURE THE OUTPUT: text=True and capture_output=True grab the terminal logs
    process = subprocess.run(command, env=custom_env, capture_output=True, text=True)
    output_log = process.stdout + "\n" + process.stderr
    
    # Save the full terminal output to a text file
    log_file_path = os.path.join(log_dir, f"log_alpha{alpha}_beta{beta}.txt")
    with open(log_file_path, "w") as f:
        f.write(output_log)
        
    print(f"Saved run logs to: {log_file_path}")

    # 2. EXTRACT METRICS: Scan the output for the last printed epoch metrics
    # Looking for lines like: "epoch: 124, ... Val MSE:1.234, Val CE:1.11, ... Val Triplet:0.55"
    val_mse, val_triplet = None, None
    
    # Find all matches for Val MSE and Val Triplet
    mse_matches = re.findall(r'Val MSE:\s*([0-9.]+)', output_log)
    triplet_matches = re.findall(r'Val Triplet:\s*([0-9.]+)', output_log)
    
    if mse_matches and triplet_matches:
        # Grab the very last one printed (usually from early stopping or final epoch)
        val_mse = float(mse_matches[-1])
        val_triplet = float(triplet_matches[-1])
        print(f"Result -> Val MSE: {val_mse:.4f} | Val Triplet: {val_triplet:.4f}")
    else:
        print("Result -> Metrics not found! Run may have crashed.")

    # Record the data
    all_results.append({
        "Alpha": alpha,
        "Beta": beta,
        "Final_Val_MSE": val_mse,
        "Final_Val_Triplet": val_triplet,
        "Log_File": log_file_path
    })

    # Rename the latent space file
    original_file = os.path.join(ckpt_dir, "train_z.npy")
    new_file = os.path.join(ckpt_dir, f"train_z_alpha{alpha}_beta{beta}.npy")
    if os.path.exists(original_file):
        os.rename(original_file, new_file)
    print(f"{'='*50}\n")

# 3. LOG THE RESULTS: Save to a CSV file and sort by best Val MSE
summary_file = "grid_search_summary.csv"

# Sort results: prioritize runs that didn't crash, then sort by lowest Val MSE
valid_results = [r for r in all_results if r["Final_Val_MSE"] is not None]
crashed_results = [r for r in all_results if r["Final_Val_MSE"] is None]
valid_results.sort(key=lambda x: x["Final_Val_MSE"]) # Lowest MSE is usually best

final_results = valid_results + crashed_results

with open(summary_file, mode='w', newline='') as csv_file:
    fieldnames = ["Alpha", "Beta", "Final_Val_MSE", "Final_Val_Triplet", "Log_File"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in final_results:
        writer.writerow(row)

print(f"Grid Search Complete! Summary saved to {summary_file}")
if valid_results:
    best = valid_results[0]
    print(f"\n🏆 BEST CONFIGURATION (Lowest Val MSE): Alpha = {best['Alpha']}, Beta = {best['Beta']} (MSE: {best['Final_Val_MSE']})")