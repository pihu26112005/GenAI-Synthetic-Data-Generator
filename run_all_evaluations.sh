#!/bin/bash
set -e

echo "========================================================="
echo "   RIGOROUS 5-MODEL ABLATION STUDY PIPELINE              "
echo "========================================================="

echo "\n[Step 1] Ensure you've run Model 1 (Vanilla) and Baselines externally."
echo "Press ENTER if ablation_1_vanilla.csv is ready in synthetic/adult/ ..."
read dummy

echo "\n[Step 2] Running Post-Filter (Band-Aid on Vanilla)..."
python evaluation_pipeline/2_run_post_filter_ablation.py

echo "\n[Step 3] Running CTTVAE Ablations (Models 3, 4, 5)..."
python evaluation_pipeline/3_run_cttvae_ablations.py

echo "\n[Step 4] Running Final Evaluation..."
python evaluation_pipeline/4_evaluate_all_models.py

echo "\nPipeline Complete."
