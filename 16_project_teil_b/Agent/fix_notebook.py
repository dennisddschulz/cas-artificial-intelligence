#!/usr/bin/env python3
"""
Fix the df_test_trading NameError in the notebook by replacing it with df_test
"""
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

try:
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Track replacements
    replacements_made = 0
    
    # Iterate through cells and fix df_test_trading references
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            # Fix the source code
            if 'source' in cell and isinstance(cell['source'], list):
                new_source = []
                for line in cell['source']:
                    new_line = line.replace('df_test_trading', 'df_test')
                    if new_line != line:
                        replacements_made += 1
                        print(f"Cell {cell_idx}: Fixed line: {line.strip()}", file=sys.stderr)
                    new_source.append(new_line)
                cell['source'] = new_source
            
            # Also fix outputs that show the error
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output and isinstance(output['text'], list):
                        new_text = []
                        for line in output['text']:
                            new_text.append(line.replace('df_test_trading', 'df_test'))
                        output['text'] = new_text
                    if 'traceback' in output and isinstance(output['traceback'], list):
                        new_traceback = []
                        for line in output['traceback']:
                            new_traceback.append(line.replace('df_test_trading', 'df_test'))
                        output['traceback'] = new_traceback
    
    # Save the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n✓ Notebook fixed successfully!", file=sys.stderr)
    print(f"  - Total replacements made: {replacements_made}", file=sys.stderr)
    print(f"  - Replaced 'df_test_trading' with 'df_test'", file=sys.stderr)
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

