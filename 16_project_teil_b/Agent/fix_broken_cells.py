#!/usr/bin/env python3
"""
Repariere die fehlerhaft formatierten Zellen 29, 30, 31
Diese Zellen haben den ganzen Code auf einer Zeile statt auf mehreren Zeilen
"""

import json

nb_path = 'Project_Part_2_Final_Architecture.ipynb'

# Lade Notebook
with open(nb_path, 'r') as f:
    nb = json.load(f)

print(f"Lade {len(nb['cells'])} Zellen")

# Zelle 29: Zu einer langen Zeile zusammengeschrumpft - entferne sie
# Zelle 30: Zu einer langen Zeile zusammengeschrumpft - entferne sie  
# Zelle 31: Zu einer langen Zeile zusammengeschrumpft - entferne sie

# Diese Zellen sind fehlerhaft, daher entfernen wir sie
indices_to_remove = []

for i in range(28, min(35, len(nb['cells']))):
    cell = nb['cells'][i]
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        lines = source.split('\n')
        
        # Wenn eine Code-Zelle NUR eine Zeile ist, die mit # anfängt und lang ist,
        # ist sie wahrscheinlich fehlerhaft (Code wurde auf einer Zeile zusammengebaut)
        if len(lines) == 1 and lines[0].startswith('#') and len(lines[0]) > 500:
            print(f"✓ Entferne Zelle {i}: Fehlerhaft formatierter Code (alles auf einer Zeile)")
            indices_to_remove.append(i)

# Entferne die Zellen (rückwärts, damit Indizes nicht verschoben werden)
for i in reversed(indices_to_remove):
    del nb['cells'][i]

print(f"\n✓ Nach Entfernung: {len(nb['cells'])} Zellen")

# Speichere das reparierte Notebook
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Notebook repariert und gespeichert!")
print("\nDie fehlerhaft formatierten Zellen wurden entfernt.")
print("Das Notebook sollte jetzt wieder ausführbar sein.")

