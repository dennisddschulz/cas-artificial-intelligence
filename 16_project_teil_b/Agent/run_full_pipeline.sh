#!/bin/bash
# run_full_pipeline.sh
#
# Automatische Pipeline: Notebook → Visualisierungen → PowerPoint
#
# Usage:
#   bash run_full_pipeline.sh
#   bash run_full_pipeline.sh --skip-notebook  # Überspringt Notebook wenn bereits trainiert

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SKIP_NOTEBOOK=${1:-""}

echo "=========================================="
echo "FORECAST-AUGMENTED RL TRADING"
echo "Full Pipeline Execution"
echo "=========================================="
echo ""

# Step 1: Run Notebook (optional)
if [ "$SKIP_NOTEBOOK" != "--skip-notebook" ]; then
    echo "[STEP 1/3] Running Jupyter Notebook..."
    echo "   This may take 40-130 minutes depending on GPU/CPU"
    echo "   Running: Project_Part_2_Final_Architecture.ipynb"
    echo ""

    # Try to run notebook with jupyter nbconvert
    if command -v jupyter &> /dev/null; then
        jupyter nbconvert --to notebook --execute \
            --ExecutePreprocessor.timeout=7200 \
            "$SCRIPT_DIR/Project_Part_2_Final_Architecture.ipynb" \
            --output "$SCRIPT_DIR/Project_Part_2_Final_Architecture_executed.ipynb" \
            --ExecutePreprocessor.kernel_name=python3 2>&1 || {
            echo "⚠️  Jupyter nbconvert failed. Trying alternative method..."
            python3 -m jupyter nbconvert --to notebook --execute \
                --ExecutePreprocessor.timeout=7200 \
                "$SCRIPT_DIR/Project_Part_2_Final_Architecture.ipynb" \
                --output "$SCRIPT_DIR/Project_Part_2_Final_Architecture_executed.ipynb" \
                --ExecutePreprocessor.kernel_name=python3
        }
    else
        echo "❌ Error: Jupyter not found in PATH"
        echo "   Install with: pip install jupyter"
        exit 1
    fi

    echo "✓ Notebook execution complete"
    echo "✓ metrics.pkl should be created"
    echo ""
else
    echo "[STEP 1/3] Skipping notebook (--skip-notebook flag)"
    echo "⚠️  Make sure metrics.pkl exists from a previous run!"
    echo ""
fi

# Step 2: Generate Visualizations
echo "[STEP 2/3] Generating visualizations..."
echo ""

if [ -f "$SCRIPT_DIR/metrics.pkl" ]; then
    echo "✓ Found metrics.pkl"
    echo "  Running: python3 create_visualizations.py"
    echo ""

    python3 "$SCRIPT_DIR/create_visualizations.py" \
        --metrics "$SCRIPT_DIR/metrics.pkl" \
        --output_dir "$SCRIPT_DIR/plots" || {
        echo "❌ Error generating visualizations"
        exit 1
    }

    echo "✓ Visualizations generated in ./plots/"
    echo ""
else
    echo "❌ Error: metrics.pkl not found!"
    echo "   Make sure notebook execution completed successfully"
    echo "   metrics.pkl should be created by the notebook"
    exit 1
fi
echo ""

# Step 3: Generate PowerPoint
echo "[STEP 3/3] Generating PowerPoint presentation..."
echo ""

if [ -f "$SCRIPT_DIR/metrics.pkl" ]; then
    if [ -d "$SCRIPT_DIR/plots" ]; then
        echo "✓ Found metrics.pkl and plots directory"
        echo "  Running: python3 generate_presentation.py"
        echo ""

        python3 "$SCRIPT_DIR/generate_presentation.py" \
            --metrics "$SCRIPT_DIR/metrics.pkl" \
            --images "$SCRIPT_DIR/plots" \
            --output "$SCRIPT_DIR/Forecast_Augmented_RL_Trading.pptx" || {
            echo "❌ Error generating PowerPoint"
            exit 1
        }

        echo "✓ PowerPoint created successfully!"
        echo ""
    else
        echo "❌ Error: plots directory not found!"
        echo "   Make sure Step 2 completed successfully"
        exit 1
    fi
else
    echo "❌ Error: metrics.pkl not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  ✓ ./plots/           (12 PNG visualizations)"
echo "    ├─ 01_performance_comparison.png"
echo "    ├─ 02_system_architecture.png"
echo "    ├─ 03_state_space_components.png"
echo "    ├─ 04_reward_function_breakdown.png"
echo "    ├─ 05_training_dynamics.png"
echo "    ├─ 06_model_summary.png"
echo "    ├─ 07_equity_curve.png"
echo "    ├─ 08_position_over_time.png"
echo "    ├─ 09_pnl_analysis.png"
echo "    ├─ 10_returns_analysis.png"
echo "    ├─ 11_risk_metrics.png"
echo "    └─ 12_win_rate_analysis.png"
echo ""
echo "  ✓ Forecast_Augmented_RL_Trading.pptx (23-slide presentation with all visualizations)"
echo ""
echo "Next step: Open PowerPoint and present!"
echo "  open Forecast_Augmented_RL_Trading.pptx"
echo ""

