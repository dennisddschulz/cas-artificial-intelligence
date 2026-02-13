#!/bin/bash

# DQN Results Checker
# This script helps you find and view all generated results

echo "════════════════════════════════════════════════════════════════════════════════"
echo "                     DQN vs Double DQN - Results Checker"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Change to directory
cd /home/isc-den/cas-artificial-intelligence/10_dqn/lunar_lander_guido

echo "📁 Current Directory:"
pwd
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "📊 AVAILABLE RESULTS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check for existing plots/images
echo "🖼️  Images & Plots:"
echo "────────────────────────────────────────────────────────────────────────────────"
if ls *.png *.gif 2>/dev/null | grep -q .; then
    ls -lh *.png *.gif 2>/dev/null | awk '{printf "  ✓ %-40s %8s\n", $9, $5}'
else
    echo "  ⚠️  No PNG or GIF files found yet"
fi

# Check plots directory
if [ -d "plots" ]; then
    echo ""
    echo "  📂 plots/ directory:"
    if ls plots/*.png 2>/dev/null | grep -q .; then
        ls -lh plots/*.png 2>/dev/null | awk '{printf "    ✓ %-38s %8s\n", $9, $5}'
    else
        echo "    ⚠️  Empty"
    fi
fi

echo ""
echo "🎬 Videos & Animations:"
echo "────────────────────────────────────────────────────────────────────────────────"
if ls *.mp4 2>/dev/null | grep -q .; then
    ls -lh *.mp4 2>/dev/null | awk '{printf "  ✓ %-40s %8s\n", $9, $5}'
else
    echo "  ⚠️  No MP4 files found yet"
fi

# Check videos directory
if [ -d "videos" ]; then
    echo ""
    echo "  📂 videos/ directory:"
    if ls videos/*.mp4 2>/dev/null | grep -q .; then
        ls -lh videos/*.mp4 2>/dev/null | awk '{printf "    ✓ %-38s %8s\n", $9, $5}'
    else
        echo "    ⚠️  Empty"
    fi
fi

echo ""
echo "💾 Data Files:"
echo "────────────────────────────────────────────────────────────────────────────────"
if ls *.pkl *.pt *.pth 2>/dev/null | grep -q .; then
    ls -lh *.pkl *.pt *.pth 2>/dev/null | awk '{printf "  ✓ %-40s %8s\n", $9, $5}'
else
    echo "  ⚠️  No data files found yet"
fi

echo ""
echo "📝 Log Files:"
echo "────────────────────────────────────────────────────────────────────────────────"
if ls *.log *.txt 2>/dev/null | grep -q .; then
    ls -lh *.log *.txt 2>/dev/null | awk '{printf "  ✓ %-40s %8s\n", $9, $5}'
else
    echo "  ⚠️  No log files found"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "⚙️  TRAINING STATUS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check running processes
running_processes=$(ps aux | grep -E "dqn_analysis|quick_comparison" | grep python | grep -v grep)
if [ -n "$running_processes" ]; then
    echo "🔄 Training processes RUNNING:"
    echo ""
    ps aux | grep -E "dqn_analysis|quick_comparison" | grep python | grep -v grep | \
        awk '{printf "  Process %s: %s (Running: %s)\n", $2, $11" "$12, $10}'
    echo ""
    echo "  ℹ️  Training is in progress. New results will appear when complete."
else
    echo "✓ No training processes running"
    echo ""
    echo "  All training complete or not started."
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "👁️  HOW TO VIEW RESULTS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Count available files
png_count=$(ls *.png plots/*.png 2>/dev/null | wc -l)
gif_count=$(ls *.gif 2>/dev/null | wc -l)
mp4_count=$(ls *.mp4 videos/*.mp4 2>/dev/null | wc -l)

if [ $png_count -gt 0 ] || [ $gif_count -gt 0 ] || [ $mp4_count -gt 0 ]; then
    echo "✅ Results available! Here's how to view them:"
    echo ""

    if [ $png_count -gt 0 ]; then
        echo "📊 View Training Plots:"
        echo "  # Open in image viewer:"
        ls *.png plots/*.png 2>/dev/null | head -3 | while read file; do
            echo "  xdg-open $file"
        done
        [ $png_count -gt 3 ] && echo "  ... and $((png_count - 3)) more"
    fi

    if [ $gif_count -gt 0 ]; then
        echo ""
        echo "🎬 View Agent Animations:"
        ls *.gif 2>/dev/null | while read file; do
            echo "  xdg-open $file"
        done
    fi

    if [ $mp4_count -gt 0 ]; then
        echo ""
        echo "🎥 Play Videos:"
        ls *.mp4 videos/*.mp4 2>/dev/null | head -3 | while read file; do
            echo "  vlc $file  # or: mpv $file"
        done
        [ $mp4_count -gt 3 ] && echo "  ... and $((mp4_count - 3)) more"
    fi

    echo ""
    echo "🌐 View in Browser:"
    echo "  firefox results_viewer.html &"
    echo "  # or:"
    echo "  python3 -m http.server 8080"
    echo "  # then open: http://localhost:8080/results_viewer.html"

    echo ""
    echo "📓 View in Jupyter:"
    echo "  jupyter notebook 13_DQN_LunarLander.ipynb"

else
    echo "⏳ No results available yet."
    echo ""
    echo "Training is in progress. Results will appear in:"
    echo "  - Current directory (PNG, GIF files)"
    echo "  - plots/ directory (comprehensive analysis)"
    echo "  - videos/ directory (MP4 videos)"
    echo ""
    echo "Check back in 10-30 minutes or run this script again:"
    echo "  ./check_results.sh"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📋 QUICK COMMANDS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "View specific result:"
echo "  xdg-open lunarlander_training.png       # Training plot"
echo "  xdg-open dqn_lunarlander.gif            # DQN animation"
echo "  xdg-open ddqn_lunarlander.gif           # Double DQN animation"
echo ""
echo "Check training progress:"
echo "  ./check_status.sh                       # Overall status"
echo "  tail -f quick_results.log               # Watch training"
echo ""
echo "Open all images at once:"
echo "  for f in *.png; do xdg-open \$f & done"
echo ""
echo "List all results:"
echo "  find . -name '*.png' -o -name '*.gif' -o -name '*.mp4'"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

