#!/usr/bin/env python3
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"✓ Notebook is valid JSON")
    print(f"✓ Total cells: {len(nb['cells'])}")
    
    # Count code cells
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    print(f"✓ Code cells: {len(code_cells)}")
    
except json.JSONDecodeError as e:
    print(f"✗ JSON Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

