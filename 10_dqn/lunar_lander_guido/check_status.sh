#!/bin/bash

# DQN Analysis Status Checker

echo "=================================="
echo "DQN Analysis Status"
echo "=================================="
echo ""

# Check running processes
echo "Running Processes:"
ps aux | grep "dqn_analysis" | grep -v grep | while read line; do
    echo "  $line"
done
echo ""

# Check for output files
echo "Generated Files:"
echo ""

if [ -d "plots" ]; then
    echo "  Plots directory:"
    ls -lh plots/*.png 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
else
    echo "  Plots directory: Not yet created"
fi
echo ""

if [ -d "videos" ]; then
    echo "  Videos directory:"
    ls -lh videos/*.mp4 2>/dev/null | head -5 | awk '{print "    " $9 " (" $5 ")"}'
    video_count=$(ls videos/*.mp4 2>/dev/null | wc -l)
    if [ $video_count -gt 5 ]; then
        echo "    ... and $((video_count - 5)) more videos"
    fi
else
    echo "  Videos directory: Not yet created"
fi
echo ""

echo "  Other files:"
[ -f "statistics_report.txt" ] && echo "    ✓ statistics_report.txt ($(stat -c%s statistics_report.txt 2>/dev/null | awk '{print int($1/1024)}')KB)" || echo "    ✗ statistics_report.txt"
[ -f "training_log.txt" ] && echo "    ✓ training_log.txt ($(stat -c%s training_log.txt 2>/dev/null | awk '{print int($1/1024)}')KB)" || echo "    ✗ training_log.txt"
[ -f "quick_training_log.txt" ] && echo "    ✓ quick_training_log.txt ($(stat -c%s quick_training_log.txt 2>/dev/null | awk '{print int($1/1024)}')KB)" || echo "    ✗ quick_training_log.txt"
[ -f "dqn_analysis_results.pkl" ] && echo "    ✓ dqn_analysis_results.pkl" || echo "    ✗ dqn_analysis_results.pkl"

echo ""
echo "Checkpoints:"
ls -lh *_checkpoint.pt 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
if [ $? -ne 0 ]; then
    echo "    No checkpoints yet"
fi

echo ""
echo "=================================="
echo "Recommendations:"
echo "=================================="
echo ""
echo "1. For immediate results:"
echo "   jupyter notebook DQN_vs_DoubleDQN_Comprehensive_Analysis.ipynb"
echo ""
echo "2. To check progress of running script:"
echo "   tail -f training_log.txt"
echo ""
echo "3. To run quick version (2-3 hours):"
echo "   python dqn_analysis_quick.py > quick_log.txt 2>&1 &"
echo ""

