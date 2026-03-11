#!/bin/bash
# wandb_setup.sh
# Setup script for Weights & Biases integration

echo "=========================================="
echo "Weights & Biases Setup Script"
echo "=========================================="
echo ""

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb
else
    echo "✓ wandb is already installed"
fi

echo ""
echo "To login to Weights & Biases, run:"
echo ""
echo "  wandb login"
echo ""
echo "This will open a browser or prompt for your API key."
echo "Your API key can be found at: https://wandb.ai/authorize"
echo ""
echo "Alternative: Set API key via environment variable:"
echo "  export WANDB_API_KEY='your-api-key-here'"
echo ""
echo "Or pass it directly to wandb.init():"
echo "  wandb.init(api_key='your-api-key-here')"
echo ""
echo "=========================================="

