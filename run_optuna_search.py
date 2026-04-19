import os
import subprocess
import re
import optuna

dataname = "adult"
gpu = 0

# Set up the environment variable
custom_env = os.environ.copy()
custom_env["PYTHONPATH"] = os.path.abspath("ctabsyn") + os.pathsep + custom_env.get("PYTHONPATH", "")

os.makedirs("logs", exist_ok=True)
os.makedirs(f"ctabsyn/tabsyn/vae/ckpt/{dataname}", exist_ok=True)

def objective(trial):
    # 1. Let Optuna intelligently pick the values!
    # We use log=True because we want it to search exponentially (e.g., 0.01, 0.1, 1.0)
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.01, 2.0, log=True)
    
    print(f"\n{'='*50}")
    print(f"Trial {trial.number}: Testing alpha={alpha:.4f}, beta={beta:.4f}")
    
    command = [
        "python", "ctabsyn/tabsyn/vae/main.py",
        "--dataname", dataname,
        "--gpu", str(gpu),
        "--alpha", str(alpha),
        "--beta", str(beta)
    ]
    
    # 2. Run the training script in our safe subprocess container
    process = subprocess.run(command, env=custom_env, capture_output=True, text=True)
    output_log = process.stdout + "\n" + process.stderr
    
    # Save the log for debugging
    with open(f"logs/optuna_trial_{trial.number}.txt", "w") as f:
        f.write(output_log)
        
    # 3. Extract the Final Validation MSE
    mse_matches = re.findall(r'Val MSE:\s*([0-9.]+)', output_log)
    triplet_matches = re.findall(r'Val Triplet:\s*([0-9.]+)', output_log)
    
    if mse_matches and triplet_matches:
        final_val_mse = float(mse_matches[-1])
        final_val_triplet = float(triplet_matches[-1])
        print(f"Trial {trial.number} finished with Val MSE: {final_val_mse:.4f} | Val Triplet: {final_val_triplet:.4f}")
        return final_val_mse, final_val_triplet
    else:
        # If it crashes (e.g., Mode Collapse), return a terrible score so Optuna learns to avoid this area
        print(f"Trial {trial.number} CRASHED. Penalizing this zone.")
        return 9999.0, 9999.0 

# 4. Create the Study and let Optuna loose!
print("Starting Optuna Bayesian Optimization...")
study = optuna.create_study(directions=["minimize", "minimize"]) # We want the lowest MSE possible
study.optimize(objective, n_trials=20) # Let it run 20 intelligent trials

# 5. Print the Pareto Front (Best Trade-offs)
print(f"\n{'='*50}")
print("🏆 OPTUNA SEARCH COMPLETE 🏆")
print(f"Number of Pareto-optimal trials: {len(study.best_trials)}")

for trial in study.best_trials:
    print(f"\nTrial {trial.number}:")
    print(f"  MSE: {trial.values[0]:.4f}, Triplet: {trial.values[1]:.4f}")
    print(f"  Hyperparameters: {trial.params}")
print(f"{'='*50}")