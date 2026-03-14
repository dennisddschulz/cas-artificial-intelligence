#!/usr/bin/env python3
"""
Add the RL agent equity line to the visualization
"""
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

# Code to add after the forecast_only_equity line in the visualization
rl_equity_line_code = '''ax3.plot(dates, equity, label='RL Agent Strategy', linewidth=2.5, color='#1f77b4')
ax3.fill_between(dates, equity, alpha=0.2, color='#1f77b4')
'''

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    # Find the line with ax3.fill_between for forecast_only_equity
    found = False
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'ax3.fill_between(dates, forecast_only_equity' in source and 'ax3.plot(dates, equity' not in source:
                print(f"Found visualization cell at index {idx}", file=sys.stderr)
                
                # Modify the cell to add the RL equity line
                new_source = source.replace(
                    'ax3.fill_between(dates, forecast_only_equity, alpha=0.2, color=\'#C73E1D\')',
                    'ax3.fill_between(dates, forecast_only_equity, alpha=0.2, color=\'#C73E1D\')\nax3.plot(dates, equity, label=\'RL Agent Strategy\', linewidth=2.5, color=\'#1f77b4\')\nax3.fill_between(dates, equity, alpha=0.2, color=\'#1f77b4\')'
                )
                
                # Update the source
                cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.split('\n')[:-1]]
                
                print(f"Added RL agent equity line to visualization", file=sys.stderr)
                found = True
                break
    
    if not found:
        print("WARNING: Could not find the forecast_only_equity visualization cell", file=sys.stderr)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("\n✓ Successfully updated visualization!", file=sys.stderr)
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

