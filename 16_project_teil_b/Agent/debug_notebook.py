import json
import sys

try:
    with open('/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb', 'r') as f:
        nb = json.load(f)
    
    print(f"Successfully loaded notebook with {len(nb['cells'])} cells", file=sys.stderr)
    
    # Find cells with df_test_trading
    found_cells = []
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'df_test_trading' in source:
                found_cells.append((idx, source))
    
    if found_cells:
        print(f"Found {len(found_cells)} cells with 'df_test_trading':", file=sys.stderr)
        for idx, source in found_cells:
            print(f"\nCell {idx}:", file=sys.stderr)
            print(source, file=sys.stderr)
    else:
        print("No cells found with 'df_test_trading'", file=sys.stderr)
        
        # Try forecast_probs_aligned
        for idx, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'forecast_probs_aligned' in source:
                    print(f"\nFound 'forecast_probs_aligned' in cell {idx}:", file=sys.stderr)
                    print(source, file=sys.stderr)
                    break
                    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)

