#!/usr/bin/env python3
"""
System Check & Verification
Verify all dependencies and files are in place before running experiments
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("\n1. Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ✗ ERROR: Python 3.8+ required")
        return False
    print("   ✓ OK")
    return True


def check_dependencies():
    """Check required packages"""
    print("\n2. Checking dependencies...")
    
    required = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'torch': 'PyTorch',
        'gymnasium': 'Gymnasium',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'yfinance': 'yfinance',
        'scipy': 'SciPy',
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} (MISSING)")
            missing.append(name)
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install -r requirements.txt")
        return False
    
    print("   ✓ All dependencies OK")
    return True


def check_files():
    """Check required files exist"""
    print("\n3. Checking required files...")
    
    required_files = {
        'run_all_experiments.py': 'Main experiment runner',
        'create_visualizations.py': 'Visualization script',
        'trading_config.py': 'Configuration module',
        'trading_framework.py': 'Framework module',
        'trading_metrics.py': 'Metrics module',
        'requirements.txt': 'Dependencies file',
    }
    
    missing = []
    
    for filename, description in required_files.items():
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            if size > 0:
                print(f"   ✓ {filename} ({size:,} bytes)")
            else:
                print(f"   ✗ {filename} (EMPTY!)")
                missing.append(filename)
        else:
            print(f"   ✗ {filename} (NOT FOUND)")
            missing.append(filename)
    
    if missing:
        print(f"\n   Missing/empty files: {', '.join(missing)}")
        return False
    
    print("   ✓ All files OK")
    return True


def check_modules():
    """Check custom modules can be imported"""
    print("\n4. Checking custom modules...")
    
    try:
        from trading_config import ExperimentConfig, get_ppo_without_forecast_config
        print("   ✓ trading_config")
    except Exception as e:
        print(f"   ✗ trading_config: {e}")
        return False
    
    try:
        from trading_metrics import TradingMetrics, MetricsComparison, EquityCurveAnalyzer
        print("   ✓ trading_metrics")
    except Exception as e:
        print(f"   ✗ trading_metrics: {e}")
        return False
    
    try:
        from trading_framework import ExperimentRunner
        print("   ✓ trading_framework")
    except Exception as e:
        print(f"   ✗ trading_framework: {e}")
        return False
    
    print("   ✓ All modules OK")
    return True


def check_gpu():
    """Check GPU availability"""
    print("\n5. Checking GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available ({torch.cuda.get_device_name(0)})")
            print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("   ℹ CUDA not available (will use CPU)")
            print("   Experiments will run slower, but still work")
            return True
    except Exception as e:
        print(f"   ✗ Error checking GPU: {e}")
        return False


def check_disk_space():
    """Check available disk space"""
    print("\n6. Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        
        print(f"   Available: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print("   ✗ WARNING: Less than 5GB free (may run out of space)")
            return False
        elif free_gb < 20:
            print("   ⚠ WARNING: Less than 20GB free (not much room)")
            return True
        else:
            print("   ✓ Plenty of disk space")
            return True
    except Exception as e:
        print(f"   ✗ Error checking disk: {e}")
        return False


def main():
    """Run all checks"""
    print("="*80)
    print("PPO TRADING FRAMEWORK - SYSTEM CHECK")
    print("="*80)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_files,
        check_modules,
        check_gpu,
        check_disk_space,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_ok = all(results)
    
    if all_ok:
        print("\n✓ ALL CHECKS PASSED - Ready to run experiments!\n")
        print("Next steps:")
        print("  1. python run_all_experiments.py")
        print("  2. Wait 3-6 hours (CPU) or 1-2 hours (GPU)")
        print("  3. Check generated files and plots")
        print("  4. Later: python create_visualizations.py (to regenerate plots)\n")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - Please fix issues above\n")
        print("Common fixes:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Missing files: Check you're in correct directory")
        print("  - Module import errors: Reinstall with pip")
        print("  - Low disk space: Free up at least 20GB\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

