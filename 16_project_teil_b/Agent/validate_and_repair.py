#!/usr/bin/env python3
"""
Repariere das beschädigte Notebook
Nutze nbformat zum Validieren und Auto-Repair
"""

import sys
import json

nb_path = 'Project_Part_2_Final_Architecture.ipynb'

print("Versuche Notebook zu reparieren...")

# Lese rohe Datei
with open(nb_path, 'rb') as f:
    content_bytes = f.read()

# Konvertiere zu String
content_str = content_bytes.decode('utf-8', errors='replace')

print(f"Dateigröße: {len(content_str)} Zeichen")

# Versuche JSON zu laden
try:
    nb = json.loads(content_str)
    print(f"✓ JSON ist valid!")
    print(f"✓ {len(nb['cells'])} Zellen gefunden")
except json.JSONDecodeError as e:
    print(f"✗ JSON Error bei Zeichen {e.pos}")
    print(f"  Context: ...{content_str[max(0,e.pos-100):e.pos+100]}...")
    
    # Versuche zu reparieren: Finde den letzten gültigen Punkt
    # und schneide von dort ab
    
    # Finde letztes vollständiges "]" für cells array
    content_fixed = content_str
    
    # Finde die fehlerhaften Zellen am Ende
    cells_start = content_fixed.find('"cells": [')
    if cells_start != -1:
        # Finde das Ende der cells
        # Suche rückwärts nach dem letzten gültigen ]
        last_valid_bracket = content_fixed.rfind(']\n    ]')
        if last_valid_bracket != -1:
            # Korrigiere die Struktur
            content_fixed = content_fixed[:last_valid_bracket+1] + '\n    ]'
            
            # Speichere die reparierte Version
            with open(nb_path, 'w') as f:
                f.write(content_fixed)
            
            print("✓ Notebook repariert!")
            
            # Versuche erneut zu laden
            try:
                nb = json.loads(content_fixed)
                print(f"✓ Nach Reparatur: {len(nb['cells'])} Zellen")
            except:
                print("✗ Reparatur nicht erfolgreich")
                sys.exit(1)
        else:
            print("✗ Konnte keine Reparaturstelle finden")
            sys.exit(1)
    else:
        print("✗ Unerwartete Struktur")
        sys.exit(1)

# Zeige Zellen 28-34
print("\n" + "="*80)
print("Zellen 28-34:")
print("="*80)

for i in range(28, min(35, len(nb['cells']))):
    cell = nb['cells'][i]
    cell_type = cell.get('cell_type', 'unknown')
    
    if cell_type == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        lines = source.split('\n')
        first_line = lines[0][:80] if lines else '(empty)'
        print(f"\nCell {i}: CODE")
        print(f"  First line: {first_line}")
        print(f"  Lines: {len(lines)}")
    elif cell_type == 'markdown':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        first_line = source.split('\n')[0][:80]
        print(f"\nCell {i}: MARKDOWN")
        print(f"  {first_line}")

print("\n✓ Notebook analysiert")

