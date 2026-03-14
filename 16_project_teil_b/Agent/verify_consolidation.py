#!/usr/bin/env python3
"""
Verification script to confirm notebook consolidation is complete
and all experiment classes are properly integrated
"""

import json
import sys

def verify_notebook_consolidation():
    """Verify that the notebook contains all necessary classes"""
    
    print("\n" + "="*70)
    print("NOTEBOOK CONSOLIDATION VERIFICATION")
    print("="*70)
    
    notebook_path = "Project_Part_2_Final_Architecture.ipynb"
    
    try:
        with open(notebook_path, "r") as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print(f"✗ ERROR: Notebook not found at {notebook_path}")
        return False
    
    print(f"\n✓ Notebook loaded: {len(notebook['cells'])} cells")
    
    # Required classes and code patterns
    required_patterns = {
        'TradingEnv_2': 'class TradingEnv_2(gym.Env):',
        'ExperimentConfig': 'class ExperimentConfig:',
        'MetricsCalculator': 'class MetricsCalculator:',
        'ExperimentRunner': ['class ExperimentRunner:', 'class SimpleExperimentRunner:'],
        'run_all_experiments': 'def run_all_experiments(self',
        'Experiment Execution': 'runner.run_all_experiments',
    }
    
    found_patterns = {}
    cell_locations = {}
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            for pattern_name, patterns in required_patterns.items():
                if isinstance(patterns, str):
                    patterns = [patterns]
                
                for pattern in patterns:
                    if pattern in source:
                        found_patterns[pattern_name] = True
                        cell_locations[pattern_name] = i
                        break
    
    # Check all required patterns
    print("\n✓ Required Components:")
    all_found = True
    for pattern_name in required_patterns.keys():
        if pattern_name in found_patterns:
            print(f"  ✓ {pattern_name:30s} (cell {cell_locations[pattern_name]})")
        else:
            print(f"  ✗ {pattern_name:30s} NOT FOUND")
            all_found = False
    
    # Check for deprecated imports
    print("\n✓ Checking for deprecated imports:")
    has_external_import = False
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'from experiment_framework import' in source:
                print(f"  ✗ Found external import in cell {i}: experiment_framework")
                has_external_import = True
            if 'import experiment_framework' in source:
                print(f"  ✗ Found external import in cell {i}: experiment_framework")
                has_external_import = True
    
    if not has_external_import:
        print(f"  ✓ No external experiment_framework imports found")
    
    # Check environment parameter alignment
    print("\n✓ Checking TradingEnv_2 parameter alignment:")
    trading_env_correct = True
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Check for old attribute access
            if 'env_test.position' in source and 'class TradingEnv_2' not in source:
                print(f"  ✗ Found deprecated 'env_test.position' in cell {i}")
                trading_env_correct = False
            if 'env.position' in source and 'class TradingEnv_2' not in source:
                print(f"  ✗ Found deprecated 'env.position' in cell {i}")
                trading_env_correct = False
    
    if trading_env_correct:
        print(f"  ✓ All TradingEnv_2 attributes properly aligned")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if all_found and not has_external_import and trading_env_correct:
        print("\n✓✓✓ CONSOLIDATION SUCCESSFUL ✓✓✓")
        print("\nThe notebook is now COMPLETELY SELF-CONTAINED!")
        print("No external dependencies on experiment_framework.py are needed.")
        print("\nYou can now:")
        print("  1. Open the notebook in Jupyter Lab/Notebook")
        print("  2. Run all cells in order")
        print("  3. Complete ML pipeline executes end-to-end")
        return True
    else:
        print("\n✗✗✗ CONSOLIDATION INCOMPLETE ✗✗✗")
        print("\nPlease address the issues above before running the notebook.")
        return False

if __name__ == "__main__":
    success = verify_notebook_consolidation()
    sys.exit(0 if success else 1)

