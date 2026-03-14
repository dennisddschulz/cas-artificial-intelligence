#!/bin/bash
# Quick execution script

echo "========================================================================"
echo "FORECAST-AUGMENTED RL TRADING - COMPLETE EXPERIMENTAL SUITE"
echo "========================================================================"
echo ""

cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/

echo "Installing dependencies..."
pip install -q numpy pandas torch gymnasium stable-baselines3 yfinance scikit-learn wandb 2>/dev/null

echo ""
echo "Running complete experiment suite..."
echo "(This will take approximately 60-90 minutes)"
echo ""

python3 main.py

echo ""
echo "========================================================================"
echo "✓ EXPERIMENTS COMPLETE"
echo "========================================================================"
echo ""
echo "Output files generated:"
echo "  - results_comparison.csv (results table)"
echo "  - experiment_results.json (complete results)"
echo "  - ./wandb/ (W&B offline logs)"
echo ""

