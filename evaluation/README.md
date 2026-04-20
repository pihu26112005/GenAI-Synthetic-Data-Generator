# Model Evaluation & Ablation Pipeline

This folder organizes the complete rigorous 5-Model Ablation Study and baseline comparisons into a simple, sequential pipeline.

## The 5-Model Ablation Study Structure

1. **Model 1: The Vanilla Baseline (No Triplet, No MMD)** 
   - A standard VAE + Diffusion.
2. **Model 2: The Post-Hoc Filter (The "Band-Aid")**
   - Applies an ORD Random Forest Deletion process on Model 1's generated output.
3. **Model 3: The Regularizer Only (MMD = On, Triplet = Off)**
4. **Model 4: The Separator Only (Triplet = On, MMD = Off)**
5. **Model 5: The Champion (Triplet = On, MMD = On)**

## Pipeline Execution

To run the whole rigorous study, simply execute these scripts in order:

### `1_run_vanilla_and_baselines.md` (Placeholder documentation)
Since Model 1 (Vanilla) and the classic baselines (CTGAN, TVAE, TabDDPM, SMOTE, original TabSyn) use separate, unmodified repositories, run those models in their respective environments and save the generated synthetic data as `synthetic/{dataname}/ablation_1_vanilla.csv` and similarly for the baselines.

### `python 2_run_post_filter_ablation.py`
This simulates the Model 2 "Band-Aid". It trains a Random Forest on real data and drops synthetic points from the Vanilla baseline (Model 1) that land in the "overlap" zone.

### `python 3_run_cttvae_ablations.py`
Runs your custom modified repo to train and extract artifacts for Models 3, 4, and 5. This will create separated folders inside `ablation_results_{dataname}/`. It saves out the weights, the representations, and the generated synthetic data cleanly.

### `4_evaluate_all_models.py` (To be added / customized)
Once you have the CSVs from Models 1-5 and the baselines, you will run your metric suite (Wasserstein Distance, JSD, MLE / Minority Recall) across all CSVs to build the final comparative table and generate PCA plots.
