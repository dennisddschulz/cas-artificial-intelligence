#!/usr/bin/env python3
"""
Comprehensive fix: Reorder cells so all dependencies are satisfied
"""
import json

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Map cell indices with their content
cell_info = {}
for idx, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Identify cell types
        if 'FORECAST-ONLY STRATEGY IMPLEMENTATION' in source:
            cell_info[idx] = 'FORECAST_IMPL'
        elif 'PPO WITHOUT FORECAST - BASELINE' in source:
            cell_info[idx] = 'PPO_BASELINE'
        elif 'COMPREHENSIVE THREE-STRATEGY COMPARISON' in source:
            cell_info[idx] = 'COMPARISON'
        elif 'dates = np.arange(len(forecast_only_equity))' in source and 'Panel 3' in source:
            cell_info[idx] = 'VISUALIZATION'
        elif 'Generating comprehensive strategy comparison visualizations' in source:
            cell_info[idx] = 'COMPARISON_VIZ'

print("Current cell order:")
for idx in sorted(cell_info.keys()):
    print(f"  Cell {idx}: {cell_info[idx]}")

# Required execution order:
# 1. FORECAST_IMPL - must run first to define forecast_only_equity, etc.
# 2. PPO_BASELINE - must run to define equity_baseline, etc.
# 3. VISUALIZATION - uses data from 1 and 2
# 4. COMPARISON - analysis of all strategies
# 5. COMPARISON_VIZ - visualize comparison

required_order = ['FORECAST_IMPL', 'PPO_BASELINE', 'VISUALIZATION', 'COMPARISON', 'COMPARISON_VIZ']

# Get current positions
current_pos = {v: k for k, v in cell_info.items()}

# Find the target positions
target_indices = []
for cell_type in required_order:
    if cell_type in current_pos:
        target_indices.append(current_pos[cell_type])

print(f"\nRequired order: {required_order}")
print(f"Current positions: {target_indices}")

# Check if we need to reorder
sorted_indices = sorted(target_indices)
if target_indices != sorted_indices:
    print("\n❌ Cells are out of order! Reordering...")
    
    # Get the cells that need reordering
    cells_to_move = [cells[i] for i in target_indices]
    
    # Find the minimum index where we'll insert
    min_idx = min(target_indices)
    
    # Remove cells from their current positions (in reverse order to avoid index shifting)
    for idx in sorted(target_indices, reverse=True):
        cells.pop(idx)
    
    # Insert cells at the minimum position in the correct order
    for i, cell_type in enumerate(required_order):
        if cell_type in current_pos:
            # Find the cell with this type
            for cell in cells_to_move:
                source = ''.join(cell['source'])
                if ((cell_type == 'FORECAST_IMPL' and 'FORECAST-ONLY STRATEGY' in source) or
                    (cell_type == 'PPO_BASELINE' and 'PPO WITHOUT FORECAST' in source) or
                    (cell_type == 'VISUALIZATION' and 'Panel 3' in source and 'dates = np.arange' in source) or
                    (cell_type == 'COMPARISON' and 'THREE-STRATEGY' in source) or
                    (cell_type == 'COMPARISON_VIZ' and 'Generating comprehensive' in source)):
                    cells.insert(min_idx + i, cell)
                    cells_to_move.remove(cell)
                    break
    
    # Save the reordered notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("✓ Notebook cells reordered successfully!")
else:
    print("\n✓ Cells are already in correct order!")

