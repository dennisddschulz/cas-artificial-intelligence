#!/usr/bin/env python3
"""
seed_utilities.py

Utilities to check, modify, and manage random seeds across experiments.

Usage:
    python seed_utilities.py --check              # Show current seed
    python seed_utilities.py --set-seed 10        # Set seed in config
    python seed_utilities.py --compare-seeds      # Compare results with different seeds
"""

import argparse
from pathlib import Path
from typing import Optional

try:
    from trading_config import ExperimentConfig
    IMPORTS_AVAILABLE = True
except:
    IMPORTS_AVAILABLE = False


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def check_current_seed() -> Optional[int]:
    """Check the current default seed in trading_config.py"""
    print_header("CHECKING CURRENT SEED CONFIGURATION")
    
    config_path = Path("trading_config.py")
    
    if not config_path.exists():
        print("✗ trading_config.py not found")
        return None
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Find seed definition
        import re
        match = re.search(r'seed:\s*int\s*=\s*(\d+)', content)
        
        if match:
            seed = int(match.group(1))
            print(f"✓ Current default seed: {seed}")
            print(f"\nLocation: trading_config.py, ExperimentConfig dataclass")
            print(f"This seed is used for ALL experiments unless overridden\n")
            
            # Show what this means
            print("IMPLICATIONS:")
            print(f"  • All neural network initializations use seed {seed}")
            print(f"  • All random actions sampled with seed {seed}")
            print(f"  • Results are completely deterministic and reproducible")
            print(f"  • Same seed = IDENTICAL results every run")
            print(f"\nYOUR RESULTS:")
            print(f"  Final Equity: $143,611.60 with seed {seed}")
            print(f"  Total Return: 43.63% with seed {seed}")
            print(f"  These are reproducible because of fixed seed\n")
            
            return seed
        else:
            print("✗ Could not find seed definition in trading_config.py")
            return None
    
    except Exception as e:
        print(f"✗ Error reading config file: {e}")
        return None


def set_seed_in_config(new_seed: int) -> bool:
    """
    Change the default seed in trading_config.py
    
    Parameters:
    -----------
    new_seed : int
        The new seed value to set
    """
    print_header(f"SETTING NEW DEFAULT SEED: {new_seed}")
    
    config_path = Path("trading_config.py")
    
    if not config_path.exists():
        print("✗ trading_config.py not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Replace seed
        import re
        new_content = re.sub(
            r'(seed:\s*int\s*=\s*)(\d+)',
            f'\\1{new_seed}',
            content
        )
        
        if new_content == content:
            print("✗ Could not find seed to replace")
            return False
        
        with open(config_path, 'w') as f:
            f.write(new_content)
        
        print(f"✓ Successfully updated trading_config.py")
        print(f"  Old seed: {int(re.search(r'seed.*?(\d+)', content).group(1))}")
        print(f"  New seed: {new_seed}")
        print(f"\n✓ Next runs will use seed {new_seed}\n")
        
        return True
    
    except Exception as e:
        print(f"✗ Error updating config: {e}")
        return False


def show_seed_comparison_commands():
    """Show commands to compare results with different seeds"""
    print_header("HOW TO COMPARE RESULTS WITH DIFFERENT SEEDS")
    
    print("Option 1: Quick test with seeds [10, 20, 30]")
    print("-" * 80)
    print("  python multi_seed_testing.py --seeds 10 20 30 --mode rewards\n")
    
    print("Option 2: Extended test with more seeds")
    print("-" * 80)
    print("  python multi_seed_testing.py --seeds 1 5 10 15 20 --mode full\n")
    
    print("Option 3: Save stability report to CSV")
    print("-" * 80)
    print("  python multi_seed_testing.py --seeds 10 20 30 --output_csv stability.csv\n")
    
    print("Option 4: One-time seed override in code")
    print("-" * 80)
    print("""
  from trading_config import get_ppo_without_forecast_config
  config = get_ppo_without_forecast_config()
  config.seed = 10  # Override for this run only
  
  from trading_framework import ExperimentRunner
  runner = ExperimentRunner(config)
  results = runner.run()
    """)
    
    print("\nOption 5: Change default seed permanently")
    print("-" * 80)
    print("  python seed_utilities.py --set-seed 10\n")
    
    print("WHY TEST MULTIPLE SEEDS?")
    print("-" * 80)
    print("""
  • Assess robustness: How sensitive is your strategy to random initialization?
  • Measure stability: Do you get similar results across different seeds?
  • Validate findings: Is 43.63% return consistent or lucky?
  • Report uncertainty: Better to report: 43.63% ± 2.5% (mean ± std)
  • Improve credibility: Multiple seeds = more trustworthy results
    """)


def generate_seed_report():
    """Generate a report about seed configuration"""
    print_header("SEED CONFIGURATION REPORT")
    
    current_seed = check_current_seed()
    
    if current_seed is None:
        return
    
    print("\n" + "="*80)
    print("WHAT YOU SHOULD KNOW ABOUT YOUR RESULTS")
    print("="*80 + "\n")
    
    print("Current Situation:")
    print(f"  ✓ Default seed: {current_seed}")
    print(f"  ✓ Your results: 43.63% return, $143,611.60 final equity")
    print(f"  ⚠ These are deterministic (same seed = same results)")
    print(f"  ⚠ No randomness across runs = impossible to assess robustness\n")
    
    print("Questions to Answer:")
    print(f"  1. Is 43.63% actually good, or just lucky with seed {current_seed}?")
    print(f"  2. Would you get similar results with seed 10, 20, 30?")
    print(f"  3. What's the typical return across different seeds?")
    print(f"  4. How sensitive is the strategy to initialization?\n")
    
    print("Recommended Next Steps:")
    print(f"  1. Test with seeds [10, 20, 30]:")
    print(f"     python multi_seed_testing.py --seeds 10 20 30 --mode rewards")
    print(f"\n  2. If results vary ±5%:")
    print(f"     Strategy is stable and robust ✓")
    print(f"\n  3. If results vary ±15%+:")
    print(f"     Strategy is unstable ✗")
    print(f"     - Increase training steps")
    print(f"     - Adjust hyperparameters")
    print(f"     - Run multiple seeds and average\n")
    
    print("Expected Outcomes:")
    print(f"""
  Scenario A: Very Stable (CV < 5%)
    └─ Seed 10: 43% return
    └─ Seed 20: 44% return
    └─ Seed 30: 43% return
    └─ Conclusion: Results are reliable ✓

  Scenario B: Moderately Stable (CV 10-20%)
    └─ Seed 10: 42% return
    └─ Seed 20: 46% return
    └─ Seed 30: 41% return
    └─ Conclusion: Results vary, report mean±std ⚠

  Scenario C: Unstable (CV > 20%)
    └─ Seed 10: 35% return
    └─ Seed 20: 55% return
    └─ Seed 30: 25% return
    └─ Conclusion: Strategy unreliable, needs work ✗
    """)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Manage and test random seeds in trading experiments'
    )
    parser.add_argument('--check', action='store_true',
                       help='Show current seed configuration')
    parser.add_argument('--set-seed', type=int, metavar='SEED',
                       help='Change default seed in config')
    parser.add_argument('--compare-seeds', action='store_true',
                       help='Show commands to compare results across seeds')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive seed report')
    
    args = parser.parse_args()
    
    # If no arguments, show report
    if not any([args.check, args.set_seed, args.compare_seeds, args.report]):
        args.report = True
    
    if args.check:
        check_current_seed()
    
    if args.set_seed is not None:
        set_seed_in_config(args.set_seed)
    
    if args.compare_seeds:
        show_seed_comparison_commands()
    
    if args.report:
        generate_seed_report()
    
    print("\n" + "="*80)
    print("For more details, read: MULTI_SEED_TESTING_GUIDE.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

