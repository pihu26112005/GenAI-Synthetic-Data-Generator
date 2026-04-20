import os
import subprocess
import shutil

dataname = "adult"
gpu = 0

# Your custom CTTVAE Ablations
ablation_configs = [
    {"name": "mmd_only", "alpha": 0.0, "beta": 1.0},
    {"name": "triplet_only", "alpha": 1.0, "beta": 0.0},
    # {"name": "final", "alpha": 1.0, "beta": 0.5} # <-- INSERT YOUR OPTUNA WINNERS HERE
]

custom_env = os.environ.copy()
custom_env["PYTHONPATH"] = os.path.abspath("ctabsyn") + os.pathsep + custom_env.get("PYTHONPATH", "")

# The master folder for all our saved artifacts
master_out_dir = f"ablation_results_{dataname}"
os.makedirs(master_out_dir, exist_ok=True)

print(f"Starting CTTVAE Artifact Extraction for '{dataname}'...\n")

for config in ablation_configs:
    name = config["name"]
    alpha = config["alpha"]
    beta = config["beta"]
    
    # Create a dedicated vault for this specific model
    model_vault = os.path.join(master_out_dir, name)
    os.makedirs(model_vault, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"🚀 TRAINING: {name.upper()} (Alpha: {alpha}, Beta: {beta})")
    print(f"{'='*60}")
    
    # 1. Train VAE
    print(f"\n[1/3] Training VAE...")
    subprocess.run(["python", "ctabsyn/main.py", "--method", "vae", "--dataname", dataname, "--gpu", str(gpu), "--alpha", str(alpha), "--beta", str(beta)], env=custom_env, check=True)
    
    # --- SAVE VAE ARTIFACTS ---
    src_train_z = f"ctabsyn/tabsyn/vae/ckpt/{dataname}/train_z.npy"
    src_vae_model = f"ctabsyn/tabsyn/vae/ckpt/{dataname}/model.pt"
    
    if os.path.exists(src_train_z): shutil.copy(src_train_z, os.path.join(model_vault, "train_z.npy"))
    if os.path.exists(src_vae_model): shutil.copy(src_vae_model, os.path.join(model_vault, "vae_model.pt"))
    print("✅ Saved VAE Latents and Weights!")

    # 2. Train Diffusion
    print(f"\n[2/3] Training Diffusion Model...")
    subprocess.run(["python", "ctabsyn/main.py", "--method", "tabsyn", "--mode", "train", "--dataname", dataname, "--gpu", str(gpu)], env=custom_env, check=True)
    
    # --- SAVE DIFFUSION ARTIFACTS ---
    src_diff_model = f"ctabsyn/tabsyn/ckpt/{dataname}/model.pt"
    if os.path.exists(src_diff_model): shutil.copy(src_diff_model, os.path.join(model_vault, "diffusion_model.pt"))
    print("✅ Saved Diffusion Weights!")

    # 3. Generate Data
    print(f"\n[3/3] Generating Synthetic Data...")
    subprocess.run(["python", "ctabsyn/main.py", "--method", "tabsyn", "--mode", "sample", "--dataname", dataname, "--gpu", str(gpu)], env=custom_env, check=True)
    
    # --- SAVE SYNTHETIC CSV ---
    src_csv = f"synthetic/{dataname}/tabsyn.csv"
    if os.path.exists(src_csv): shutil.copy(src_csv, os.path.join(model_vault, "synthetic_data.csv"))
    print(f"✅ Saved Synthetic CSV!\n")

print(f"{'='*60}")
print(f"🎉 ALL ARTIFACTS SECURED IN '{master_out_dir}/' 🎉")
