#!/usr/bin/env python3
"""
Directly examine and fix the notebook structure
"""
import json

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}\n")

# Find the forecast-only and visualization cells
forecast_idx = None
viz_idx = None
comparison_idx = None

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        if 'FORECAST-ONLY STRATEGY IMPLEMENTATION' in source:
            forecast_idx = idx
            print(f"Cell {idx}: FORECAST-ONLY STRATEGY")
        
        if 'Panel 3: Equity Curves Comparison' in source and 'dates = np.arange' in source:
            viz_idx = idx
            print(f"Cell {idx}: VISUALIZATION (Panel 3)")
        
        if 'COMPREHENSIVE THREE-STRATEGY COMPARISON' in source:
            comparison_idx = idx
            print(f"Cell {idx}: COMPARISON ANALYSIS")

print(f"\nCell order:")
print(f"  Forecast-Only: {forecast_idx}")
print(f"  Visualization: {viz_idx}")
print(f"  Comparison: {comparison_idx}")

if forecast_idx is not None and viz_idx is not None:
    if forecast_idx > viz_idx:
        print(f"\n❌ PROBLEM: Forecast cell ({forecast_idx}) comes AFTER Visualization ({viz_idx})")
        print(f"    Need to move Forecast cell BEFORE Visualization")
    else:
        print(f"\n✓ OK: Forecast cell ({forecast_idx}) comes BEFORE Visualization ({viz_idx})")

