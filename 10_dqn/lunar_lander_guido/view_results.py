#!/usr/bin/env python3
"""
Interactive Results Viewer for DQN Analysis
Opens and displays all available results
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

def find_results():
    """Find all result files"""
    results = {
        'plots': [],
        'animations': [],
        'videos': [],
        'data': [],
        'logs': []
    }
    
    # Find PNG files
    results['plots'].extend(glob.glob('*.png'))
    results['plots'].extend(glob.glob('plots/*.png'))
    
    # Find GIF files
    results['animations'].extend(glob.glob('*.gif'))
    
    # Find MP4 files
    results['videos'].extend(glob.glob('*.mp4'))
    results['videos'].extend(glob.glob('videos/*.mp4'))
    
    # Find data files
    results['data'].extend(glob.glob('*.pkl'))
    results['data'].extend(glob.glob('*.pt'))
    results['data'].extend(glob.glob('*.pth'))
    
    # Find log files
    results['logs'].extend(glob.glob('*.log'))
    results['logs'].extend(glob.glob('*.txt'))
    
    return results

def open_file(filepath):
    """Open file with default application"""
    try:
        if sys.platform == 'linux':
            subprocess.Popen(['xdg-open', filepath])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', filepath])
        elif sys.platform == 'win32':
            os.startfile(filepath)
        print(f"  ✓ Opened: {filepath}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to open {filepath}: {e}")
        return False

def display_menu(results):
    """Display interactive menu"""
    print("\n" + "="*80)
    print(" " * 25 + "DQN Results Viewer")
    print("="*80)
    
    all_items = []
    
    if results['plots']:
        print("\n📊 Training Plots:")
        for i, f in enumerate(results['plots'], 1):
            idx = len(all_items) + 1
            print(f"  [{idx}] {f}")
            all_items.append(('plot', f))
    
    if results['animations']:
        print("\n🎬 Agent Animations (GIF):")
        for f in results['animations']:
            idx = len(all_items) + 1
            print(f"  [{idx}] {f}")
            all_items.append(('animation', f))
    
    if results['videos']:
        print("\n🎥 Videos (MP4):")
        for f in results['videos']:
            idx = len(all_items) + 1
            print(f"  [{idx}] {f}")
            all_items.append(('video', f))
    
    if results['data']:
        print("\n💾 Data Files:")
        for f in results['data']:
            idx = len(all_items) + 1
            print(f"  [{idx}] {f}")
            all_items.append(('data', f))
    
    if not any(results.values()):
        print("\n⚠️  No results found yet.")
        print("   Training may still be in progress.")
        return None
    
    print("\n" + "-"*80)
    print("\nOptions:")
    print("  [number] - Open specific file")
    print("  [a] - Open all plots")
    print("  [g] - Open all GIF animations")
    print("  [v] - Open all videos")
    print("  [w] - Open HTML viewer")
    print("  [j] - Open Jupyter notebook")
    print("  [r] - Refresh / Reload")
    print("  [q] - Quit")
    print("="*80)
    
    return all_items

def main():
    """Main function"""
    os.chdir('/home/isc-den/cas-artificial-intelligence/10_dqn/lunar_lander_guido')
    
    while True:
        results = find_results()
        all_items = display_menu(results)
        
        if all_items is None:
            print("\n💡 Tip: Run './check_results.sh' to see training status")
            break
        
        try:
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 'q':
                print("\n👋 Goodbye!")
                break
            
            elif choice == 'r':
                print("\n🔄 Refreshing...")
                continue
            
            elif choice == 'a':
                print("\n📊 Opening all plots...")
                for item_type, filepath in all_items:
                    if item_type == 'plot':
                        open_file(filepath)
            
            elif choice == 'g':
                print("\n🎬 Opening all animations...")
                for item_type, filepath in all_items:
                    if item_type == 'animation':
                        open_file(filepath)
            
            elif choice == 'v':
                print("\n🎥 Opening all videos...")
                for item_type, filepath in all_items:
                    if item_type == 'video':
                        open_file(filepath)
            
            elif choice == 'w':
                print("\n🌐 Opening HTML viewer...")
                if os.path.exists('results_viewer.html'):
                    open_file('results_viewer.html')
                else:
                    print("  ⚠️  results_viewer.html not found")
            
            elif choice == 'j':
                print("\n📓 Opening Jupyter notebook...")
                subprocess.Popen(['jupyter', 'notebook', '13_DQN_LunarLander.ipynb'])
                print("  ✓ Jupyter should open in your browser")
                break
            
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(all_items):
                    item_type, filepath = all_items[idx]
                    print(f"\n📂 Opening: {filepath}")
                    open_file(filepath)
                else:
                    print(f"\n❌ Invalid number. Choose 1-{len(all_items)}")
            
            else:
                print("\n❌ Invalid choice. Try again.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    main()

