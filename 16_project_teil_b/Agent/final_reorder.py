#!/usr/bin/env python3
"""
Direct approach: Extract and rebuild notebook with correct cell ordering
"""
import json
import re

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    content = f.read()

try:
    nb = json.loads(content)
except:
    print("ERROR: Could not parse notebook as JSON")
    exit(1)

cells = nb['cells']
code_cells = [c for c in cells if c['cell_type'] == 'code']

print(f"Total cells: {len(cells)}")
print(f"Code cells: {len(code_cells)}")

# Find the problem cells
forecast_cell_idx = None
ppo_baseline_cell_idx = None
viz_cell_idx = None
comparison_cell_idx = None
comparison_viz_cell_idx = None

for idx, cell in enumerate(code_cells):
    source = ''.join(cell['source'])
    
    if 'FORECAST-ONLY STRATEGY IMPLEMENTATION' in source:
        forecast_cell_idx = idx
        print(f"\nFound FORECAST_IMPL at code_cell index {idx}")
    
    if 'PPO WITHOUT FORECAST - BASELINE' in source:
        ppo_baseline_cell_idx = idx
        print(f"Found PPO_BASELINE at code_cell index {idx}")
    
    if 'Panel 3: Equity Curves Comparison' in source and 'forecast_only_equity' in source and idx > (forecast_cell_idx or -1):
        viz_cell_idx = idx
        print(f"Found VISUALIZATION at code_cell index {idx}")
    
    if 'COMPREHENSIVE THREE-STRATEGY COMPARISON' in source:
        comparison_cell_idx = idx
        print(f"Found COMPARISON at code_cell index {idx}")
    
    if 'Generating comprehensive strategy comparison visualizations' in source:
        comparison_viz_cell_idx = idx
        print(f"Found COMPARISON_VIZ at code_cell index {idx}")

# Expected order: FORECAST -> PPO_BASELINE -> VISUALIZATION -> COMPARISON -> COMPARISON_VIZ
expected_order = [
    (forecast_cell_idx, 'FORECAST'),
    (ppo_baseline_cell_idx, 'PPO_BASELINE'),
    (viz_cell_idx, 'VISUALIZATION'),
    (comparison_cell_idx, 'COMPARISON'),
    (comparison_viz_cell_idx, 'COMPARISON_VIZ'),
]

print("\nCell order status:")
for idx, name in expected_order:
    if idx is not None:
        print(f"  {idx}: {name}")

# Check if already in order
indices = [idx for idx, _ in expected_order if idx is not None]
if indices == sorted(indices):
    print("\n✓ Cells are already in correct order!")
else:
    print("\n❌ Cells are OUT OF ORDER! Reordering...")
    
    # Build new cells list by reordering just the code cells
    new_code_cells = []
    
    for cell_idx, name in expected_order:
        if cell_idx is not None:
            new_code_cells.append(code_cells[cell_idx])
    
    # Rebuild full cells list (code cells first, then other cells)
    other_cells = [c for c in cells if c['cell_type'] != 'code']
    
    # Insert code cells back into cells list
    non_code_cell_count = 0
    new_cells = []
    
    for cell in cells:
        if cell['cell_type'] != 'code':
            new_cells.append(cell)
        else:
            non_code_cell_count += 1
    
    # Now insert the reordered code cells before the first position they should go
    # Find position to insert
    insert_pos = 0
    for i, cell in enumerate(new_cells):
        if i > 0:  # Insert after initial cells
            insert_pos = i
            break
    
    # Simplified approach: Just reorder the code cells in place
    # Create mapping of code cells and rebuild
    code_cell_map = {idx: code_cells[idx] for idx, _ in expected_order if idx is not None}
    
    # Rebuild cells list properly
    new_cells = []
    code_cell_counter = 0
    
    for cell in cells:
        if cell['cell_type'] != 'code':
            new_cells.append(cell)
        else:
            # Add next code cell from the ordered list
            for cell_idx, name in expected_order:
                if cell_idx is not None and code_cells[cell_idx] not in new_cells:
                    new_cells.append(code_cells[cell_idx])
                    break
    
    nb['cells'] = new_cells
    
    # Save reordered notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("✓ Notebook reordered and saved!")

print(f"\n✓ Done!")

