#!/usr/bin/env python3
"""
Fix the notebook cell order - move forecast-only strategy BEFORE the visualization
"""
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

try:
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    # Find the cells
    forecast_only_idx = None
    visualization_idx = None
    
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'FORECAST-ONLY STRATEGY IMPLEMENTATION' in source:
                forecast_only_idx = idx
                print(f"Found forecast-only strategy cell at index {idx}", file=sys.stderr)
            if 'dates = np.arange(len(forecast_only_equity))' in source:
                visualization_idx = idx
                print(f"Found visualization cell at index {idx}", file=sys.stderr)
    
    if forecast_only_idx is not None and visualization_idx is not None:
        if forecast_only_idx > visualization_idx:
            print(f"Forecast-only is at {forecast_only_idx}, visualization at {visualization_idx}", file=sys.stderr)
            print("Moving forecast-only BEFORE visualization...", file=sys.stderr)
            
            # Remove the forecast-only cell
            forecast_only_cell = cells.pop(forecast_only_idx)
            
            # Insert it before the visualization cell (accounting for the removal shift)
            new_viz_idx = visualization_idx - 1 if forecast_only_idx < visualization_idx else visualization_idx
            cells.insert(new_viz_idx, forecast_only_cell)
            
            print(f"✓ Moved forecast-only cell from {forecast_only_idx} to {new_viz_idx}", file=sys.stderr)
        else:
            print("Forecast-only is already before visualization - OK!", file=sys.stderr)
    else:
        if forecast_only_idx is None:
            print("ERROR: Could not find forecast-only cell", file=sys.stderr)
        if visualization_idx is None:
            print("ERROR: Could not find visualization cell", file=sys.stderr)
    
    # Save the reordered notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("\n✓ Notebook cells reordered successfully!", file=sys.stderr)
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

