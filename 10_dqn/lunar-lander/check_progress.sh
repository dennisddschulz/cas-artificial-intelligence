#!/bin/bash
# Skript zum Überprüfen des LunarLander Training Fortschritts

echo "======================================================"
echo "LunarLander DQN Training - Fortschrittscheck"
echo "======================================================"
echo ""

# Überprüfe ob Training läuft
echo "📊 Training Status:"
ps aux | grep "train_lunar_fast.py" | grep -v grep > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Training LÄUFT!"
    ps aux | grep "train_lunar_fast.py" | grep -v grep | awk '{print "   PID: "$2", CPU: "$3"%, Memory: "$6" KB"}'
else
    echo "❌ Training nicht mehr laufen..."
fi

echo ""
echo "📈 Letzte Log-Einträge:"
echo "---"
if [ -f "/home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/training.log" ]; then
    tail -10 /home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/training.log | tail -5
else
    echo "Log-Datei existiert noch nicht"
fi
echo "---"

echo ""
echo "📁 Plots Check:"
plots_dir="/home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/plots"
if [ -d "$plots_dir" ]; then
    count=$(ls $plots_dir/*.png 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        echo "✅ Plots generiert! ($count Dateien)"
        ls -lh $plots_dir/
    else
        echo "⏳ Noch keine Plots generiert (Training läuft noch...)"
    fi
else
    echo "❌ Plots-Verzeichnis existiert noch nicht"
fi

echo ""
echo "======================================================"
echo "Warten auf Training-Fertigstellung..."
echo "Fortschritt wird automatisch aktualisiert"
echo "======================================================"
