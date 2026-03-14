# Extended Visualization and Presentation Generation Guide

## Overview

Two new scripts have been created to handle visualization and presentation generation for the extended reward function experiments:

1. **create_visualizations_extended.py** - Generates high-quality PNG visualizations
2. **generate_presentation_extended.py** - Generates PowerPoint presentations

## Quick Start

### 1. Generate Visualizations from Reward Comparison CSV

```bash
# After running reward ablation experiments
python create_visualizations_extended.py \
    --reward_csv reward_ablation_comparison.csv \
    --output_dir ./plots
```

**Output files:**
- `reward_metrics_comparison.png` - 4-panel comparison (return, sharpe, drawdown, volatility)
- `reward_heatmap.png` - Normalized performance heatmap
- `reward_scatter.png` - Risk vs Return scatter with bubble sizing
- `reward_ranking.png` - Rankings for each metric with medals

### 2. Generate Visualizations from All Metrics CSV

```bash
# Generate comparison of all 10 experiments
python create_visualizations_extended.py \
    --metrics_csv metrics_comparison.csv \
    --output_dir ./plots
```

### 3. Generate Visualizations from Pickle File

```bash
# Use metrics.pkl from run_all_experiments.py
python create_visualizations_extended.py \
    --metrics_pkl metrics.pkl \
    --output_dir ./plots
```

### 4. Generate Presentation from Reward Results Only

```bash
# Quick presentation for reward ablation study
python generate_presentation_extended.py \
    --reward_csv reward_ablation_comparison.csv \
    --reward_only \
    --output reward_ablation_analysis.pptx
```

**Slides included:**
- Title slide
- Study overview
- Reward function types (all 8)
- Key findings
- Recommendations by use case
- 5 visualization slides (auto-embedded if images exist)
- Metrics summary table
- Conclusion

### 5. Generate Comprehensive Presentation

```bash
# Full presentation with all experiments and images
python generate_presentation_extended.py \
    --metrics metrics.pkl \
    --reward_csv reward_ablation_comparison.csv \
    --images_dir ./plots \
    --output trading_analysis.pptx
```

**Slides included:**
- Executive summary
- Experiments overview
- Reward comparison results
- All generated visualizations (auto-embedded)
- Next steps

## Complete Workflow Example

```bash
# 1. Run all experiments
python run_all_experiments.py

# 2. Generate all visualizations
python create_visualizations_extended.py \
    --reward_csv reward_ablation_comparison.csv \
    --metrics_csv metrics_comparison.csv \
    --output_dir ./plots

# 3. Generate comprehensive presentation
python generate_presentation_extended.py \
    --metrics metrics.pkl \
    --reward_csv reward_ablation_comparison.csv \
    --images_dir ./plots \
    --output trading_analysis_complete.pptx

# 4. View results
open trading_analysis_complete.pptx  # or your preferred viewer
```

## Visualization Types

### create_visualizations_extended.py

#### 1. Metrics Comparison (4-panel)
- **Top Left**: Total Return comparison (%)
- **Top Right**: Sharpe Ratio comparison
- **Bottom Left**: Max Drawdown comparison (%)
- **Bottom Right**: Volatility comparison (%)

Each bar is colored differently, with values labeled on top.

#### 2. Heatmap
- **Rows**: Performance metrics
- **Columns**: Reward functions
- **Colors**: Green = good, Red = bad (normalized)
- **Values**: Actual metric values overlaid

#### 3. Scatter Plot
- **X-axis**: Total Return (%)
- **Y-axis**: Sharpe Ratio
- **Bubble Size**: Volatility
- **Labels**: Reward function names
- **Reference**: Sharpe=1.0 target line

#### 4. Ranking Visualization
- **Separate panel for each metric**
- **Rewards ranked top to bottom**
- **Medals**: 🥇🥈🥉 for top 3
- **Values**: Displayed on bars

### generate_presentation_extended.py

#### Reward Ablation Presentation (reward_only mode)
1. Title slide
2. Study overview
3. 8 reward types explanation
4. Key findings
5. Recommendations
6-10. Visualization slides (if images exist)
11. Metrics summary table
12. Conclusion

#### Comprehensive Presentation
1. Title & Executive summary
2. Experiments overview
3. Reward comparison results
4-N. All visualization slides
N+1. Next steps

## Features

### Smart Image Embedding
- Automatically embeds generated PNG files
- Gracefully handles missing images
- Scales images to fit slide width

### Dynamic Table Generation
- Creates metrics summary tables
- Color-coded headers
- Scalable to any metric set

### Color Schemes
- **Dark Blue**: Headers and titles
- **Orange**: Accents and highlights
- **Green**: Success/positive metrics
- **Gray**: Neutral backgrounds
- **Red**: Negative/risk indicators

## Advanced Usage

### Custom Visualization

```python
from create_visualizations_extended import RewardVisualizationGenerator
import pandas as pd

# Load data
df = pd.read_csv('reward_ablation_comparison.csv', index_col=0)

# Create generator
gen = RewardVisualizationGenerator('./my_plots')

# Generate specific visualizations
gen.create_reward_metrics_comparison(df, 'custom_metrics.png')
gen.create_reward_scatter(df, 'custom_scatter.png')
```

### Custom Presentation

```python
from generate_presentation_extended import PresentationGenerator

gen = PresentationGenerator()

# Add custom slides
gen.add_title_slide("My Analysis", "Custom Trading Report")
gen.add_content_slide("Results", [
    "• Custom finding 1",
    "• Custom finding 2",
    "• Custom finding 3"
])
gen.add_image_slide("My Chart", "my_image.png")

# Save
gen.prs.save('my_presentation.pptx')
```

## Requirements

### For Visualizations
```bash
pip install matplotlib seaborn pandas numpy
```

### For Presentations
```bash
pip install python-pptx
```

### Full Setup
```bash
pip install matplotlib seaborn pandas numpy python-pptx
```

## Troubleshooting

### Issue: "module not found: pptx"
**Solution**: Install python-pptx
```bash
pip install python-pptx
```

### Issue: PNG files not embedding in presentation
**Ensure**:
1. Images exist in the specified directory
2. Image paths are correct
3. Image files are in PNG format
4. Running from correct working directory

### Issue: CSV file not found
**Solution**: Provide full path to CSV file
```bash
python create_visualizations_extended.py \
    --reward_csv /full/path/to/reward_ablation_comparison.csv
```

### Issue: Out of memory with large files
**Solution**: Process files separately or reduce DPI
```bash
# In the code, change:
plt.savefig(..., dpi=150)  # instead of 300
```

## Output File Sizes

Typical file sizes when generated:

| File Type | Size | Notes |
|-----------|------|-------|
| PNG visualization | 200-500 KB | High DPI (300) for presentation |
| PowerPoint presentation | 2-5 MB | With embedded images |
| Metrics CSV | 10-50 KB | Very lightweight |
| Pickle file | 100-500 KB | Binary metrics storage |

## Integration with Experiments

### After `run_all_experiments.py`
```bash
# You have:
# - metrics_comparison.csv
# - reward_comparison_detailed.csv
# - metrics.pkl
# - detailed_results.json

# Generate visualizations
python create_visualizations_extended.py \
    --reward_csv reward_comparison_detailed.csv \
    --metrics_csv metrics_comparison.csv \
    --output_dir ./plots

# Generate presentation
python generate_presentation_extended.py \
    --metrics metrics.pkl \
    --reward_csv reward_comparison_detailed.csv \
    --images_dir ./plots
```

### After `run_reward_ablation.py`
```bash
# You have:
# - reward_ablation_comparison.csv
# - reward_ablation_results.json

# Generate visualizations
python create_visualizations_extended.py \
    --reward_csv reward_ablation_comparison.csv \
    --output_dir ./plots

# Generate presentation
python generate_presentation_extended.py \
    --reward_csv reward_ablation_comparison.csv \
    --reward_only
```

## Customization Guide

### Change Color Scheme
Edit in `generate_presentation_extended.py`:
```python
self.DARK_BLUE = RGBColor(46, 134, 171)  # Change RGB values
self.ACCENT_ORANGE = RGBColor(242, 142, 43)  # Your colors
```

### Add Custom Metrics to Presentation
```python
gen.add_metrics_table_slide("Custom Metrics", {
    'metric_1': 0.95,
    'metric_2': 0.87,
    'metric_3': 0.76
})
```

### Adjust Visualization Sizes
Edit in `create_visualizations_extended.py`:
```python
plt.rcParams['figure.figsize'] = (18, 12)  # Default is (16, 10)
```

## Performance Tips

1. **Faster generation**: Reduce DPI from 300 to 150
2. **Smaller files**: Use PNG instead of PDF
3. **Selective output**: Generate only needed visualizations
4. **Parallel processing**: Run on different data subsets

## File References

**Input Files Generated by:**
- `metrics_comparison.csv` ← `run_all_experiments.py`
- `reward_ablation_comparison.csv` ← `run_reward_ablation.py`
- `reward_comparison_detailed.csv` ← `run_all_experiments.py`
- `metrics.pkl` ← `run_all_experiments.py`
- Generated PNG images ← These scripts

**Output Files Created:**
- `*.png` visualization files
- `*.pptx` presentation files

## Next Steps

1. **Review metrics**: Open CSV files in spreadsheet application
2. **View visualizations**: Check generated PNG files
3. **Share analysis**: Distribute PowerPoint presentation
4. **Iterate**: Modify parameters and re-run as needed


