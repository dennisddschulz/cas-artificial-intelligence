#!/bin/bash
# Run the final project training

cd /home/isc-den/cas-artificial-intelligence/14_project_teil_b

echo "Starting Final Project Training..."
echo "Script: 09_CLEAN_FINAL_PROJECT.py"
echo "Forecasting: N-BEATS"
echo "Time: ~1-2 hours on CPU"
echo ""

# Run the training
python 09_CLEAN_FINAL_PROJECT.py

# Check if it completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "Generated files:"
    ls -lh final_project_results.csv 2>/dev/null && echo "  ✓ final_project_results.csv"
    echo ""
    echo "Results:"
    cat final_project_results.csv 2>/dev/null
else
    echo ""
    echo "❌ Training failed. Check errors above."
fi

