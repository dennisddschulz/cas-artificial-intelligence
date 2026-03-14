#!/bin/bash
# Quick start script - run all experiments

echo "========================================================================"
echo "FORECAST-AUGMENTED RL TRADING - QUICK START"
echo "========================================================================"
echo ""
echo "Installing dependencies..."
pip install -q numpy pandas torch gymnasium stable-baselines3 yfinance scikit-learn wandb

echo ""
echo "Running experiments..."
echo "(This will take 60-90 minutes)"
echo ""

cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/

python3 run_experiments.py

echo ""
echo "========================================================================"
echo "✓ EXPERIMENTS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: results_comparison.csv"
echo "W&B logs saved to: ./wandb/"
echo ""

