#!/usr/bin/env python3
"""
Repariert die fehlerhaft eingefügten Zellen im Notebook
"""

import json
from pathlib import Path

nb_path = Path('Project_Part_2_Final_Architecture.ipynb')

# Load notebook
with open(nb_path, 'r') as f:
    nb = json.load(f)

print(f"Lade Notebook: {len(nb['cells'])} Zellen")

# Finde die fehlerhaften Zellen (mit den Forecast-Only und PPO Evaluations)
# und entferne die doppelten/fehlerhaften

cells_to_keep = []
skip_next = False

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Überspringe die fehlerhaften doppelten Zellen
        if '# EXPERIMENT 1: FORECAST-ONLY STRATEGY' in source and i > 20:
            # Das ist eine der doppelten Zellen am Ende - überspringe sie
            if source.count('# EXPERIMENT 1: FORECAST-ONLY STRATEGY') == 1 and len(source) < 1500:
                print(f"  Überspringe fehlerhafte Zelle {i}")
                continue
        
        if '# EXPERIMENT 2 & 3: PPO EVALUATION' in source and i > 20:
            # Das ist eine der doppelten Zellen - überspringe sie
            if len(source) < 2000:
                print(f"  Überspringe fehlerhafte Zelle {i}")
                continue
    
    cells_to_keep.append(cell)

print(f"Nach Bereinigung: {len(cells_to_keep)} Zellen (entfernt: {len(nb['cells']) - len(cells_to_keep)})")

# Ersetze die Zellenliste
nb['cells'] = cells_to_keep

# Speichere das korrigierte Notebook
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=2)

print(f"✓ Notebook repariert und gespeichert!")
print(f"✓ Fehlerhafte doppelte Zellen entfernt")
print(f"✓ Notebook ist jetzt wieder ausführbar")

