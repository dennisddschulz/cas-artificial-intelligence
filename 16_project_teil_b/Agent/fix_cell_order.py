#!/usr/bin/env python3
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find cells
forecast_idx = None
viz_idx = None

for idx, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        if 'FORECAST-ONLY STRATEGY IMPLEMENTATION' in source:
            forecast_idx = idx
        
        if 'Panel 3: Equity Curves Comparison' in source and 'dates = np.arange' in source and 'forecast_only_equity' in source:
            viz_idx = idx

sys.stdout.write(f"Forecast cell: {forecast_idx}\n")
sys.stdout.write(f"Visualization cell: {viz_idx}\n")
sys.stdout.flush()

# If forecast comes after viz, move it before
if forecast_idx is not None and viz_idx is not None and forecast_idx > viz_idx:
    sys.stdout.write(f"Moving forecast cell from {forecast_idx} to {viz_idx}\n")
    sys.stdout.flush()
    
    forecast_cell = cells.pop(forecast_idx)
    cells.insert(viz_idx, forecast_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    sys.stdout.write("✓ Notebook reordered\n")
    sys.stdout.flush()
else:
    sys.stdout.write("✓ Order is correct or cells not found\n")
    sys.stdout.flush()

