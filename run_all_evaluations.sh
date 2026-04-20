#!/bin/bash
set -e

DATANAME=${1:-adult}
TARGET=${2:-income}
GPU=${3:-0}

echo "========================================================="
echo "   RIGOROUS 5-MODEL ABLATION STUDY PIPELINE              "
echo "   Dataset: $DATANAME | Target: $TARGET | GPU: $GPU      "
echo "========================================================="

echo "\n[Step 1] Ensure you've run Model 1 (Vanilla) and Baselines externally."
echo "Wait! Since we now have script 1, we can actually just run it!"
echo "Running Original Vanilla CTabSyn..."
python evaluation/1_run_vanilla_ctabsyn.py --dataname $DATANAME --gpu $GPU

echo "\n[Step 2] Running Post-Filter (Band-Aid on Vanilla)..."
python evaluation/2_run_post_filter_ablation.py --dataname $DATANAME --target_column $TARGET

echo "\n[Step 3] Running CTTVAE Ablations (Models 3, 4, 5)..."
python evaluation/3_run_cttvae_ablations.py --dataname $DATANAME --gpu $GPU

echo "\n[Step 4] Running Final Evaluation..."
python evaluation/4_evaluate_all_models.py --dataname $DATANAME --target_column $TARGET

echo "\nPipeline Complete."

