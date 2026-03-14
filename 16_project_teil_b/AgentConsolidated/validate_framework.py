#!/usr/bin/env python3
"""
Validation Script - Check Framework Setup Before Running Experiments
"""

import os
import sys

print("\n" + "="*80)
print("FRAMEWORK VALIDATION SCRIPT")
print("="*80 + "\n")

# Check 1: Python Version
print("1. Checking Python version...")
version = sys.version_info
if version.major >= 3 and version.minor >= 7:
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"   ✗ Python {version.major}.{version.minor} - Need 3.7+")
    sys.exit(1)

# Check 2: Required packages
print("\n2. Checking required packages...")
packages = {
    'numpy': 'np',
    'pandas': 'pd',
    'torch': 'torch',
    'gymnasium': 'gym',
    'sklearn': 'sklearn',
    'matplotlib': 'plt',
    'seaborn': 'sns',
    'yfinance': 'yf',
}

all_installed = True
for pkg, alias in packages.items():
    try:
        __import__(pkg if pkg != 'gymnasium' else 'gymnasium')
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ✗ {pkg} - Missing! Install: pip install {pkg}")
        all_installed = False

if not all_installed:
    sys.exit(1)

# Check 3: GPU availability
print("\n3. Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available - {torch.cuda.get_device_name(0)}")
        print(f"     Device count: {torch.cuda.device_count()}")
    else:
        print(f"   ✓ CPU only (CUDA not available)")
except Exception as e:
    print(f"   ⚠ GPU check failed: {e}")

# Check 4: File structure
print("\n4. Checking file structure...")
required_files = [
    'trading_config.py',
    'trading_framework.py',
    'trading_metrics.py',
    'budget_tracker.py',
    'visualize_metrics.py',
    'run_all_experiments.py',
    'load_and_inspect_metrics.py',
]

all_exist = True
for f in required_files:
    if os.path.exists(f):
        print(f"   ✓ {f}")
    else:
        print(f"   ✗ {f} - Missing!")
        all_exist = False

if not all_exist:
    print("\n   ✗ Some files missing. Check file structure.")
    sys.exit(1)

# Check 5: Data directory
print("\n5. Checking directories...")
if not os.path.exists('./results'):
    os.makedirs('./results')
    print("   ✓ Created ./results/")
else:
    print("   ✓ ./results/ exists")

if not os.path.exists('./plots'):
    os.makedirs('./plots')
    print("   ✓ Created ./plots/")
else:
    print("   ✓ ./plots/ exists")

# Check 6: Configuration
print("\n6. Checking configuration...")
try:
    from trading_config import get_all_experiments
    
    experiments = get_all_experiments()
    print(f"   ✓ Found {len(experiments)} experiments:")
    for key, config in experiments.items():
        print(f"     - {config.experiment_name:<30} (Group: {config.wandb_group})")
    
    if len(experiments) != 6:
        print(f"   ✗ Expected 6 experiments, got {len(experiments)}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Configuration error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 7: WandB
print("\n7. Checking WandB...")
try:
    import wandb
    print(f"   ✓ WandB installed (offline mode)")
except ImportError:
    print(f"   ⚠ WandB not installed (optional - still works)")

# Check 8: Quick import test
print("\n8. Testing imports...")
try:
    from trading_framework import ExperimentRunner
    print("   ✓ ExperimentRunner imported")
    
    from trading_metrics import TradingMetrics
    print("   ✓ TradingMetrics imported")
    
    from budget_tracker import BudgetTracker
    print("   ✓ BudgetTracker imported")
    
    from visualize_metrics import MetricsVisualizer
    print("   ✓ MetricsVisualizer imported")
    
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("✓ ALL CHECKS PASSED - FRAMEWORK READY TO RUN")
print("="*80)
print("\nYou can now run:")
print("  python run_all_experiments.py")
print("\nThis will execute all 6 experiments with:")
print("  ✓ Real-time console metrics")
print("  ✓ Budget tracking")
print("  ✓ Local plots (./plots/)")
print("  ✓ WandB logging with distinct groups")
print("\nEstimated time: ~1-2 hours (depending on hardware)")
print("="*80 + "\n")

