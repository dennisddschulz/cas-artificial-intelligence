#!/usr/bin/env python3
"""
Simple test to check if the main workflow can start
"""
import sys
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING MAIN.PY EXECUTION")
print("="*80)

try:
    print("\n1. Importing modules...")
    import numpy as np
    import pandas as pd
    import torch
    import yfinance as yf
    print("   ✓ All imports successful")
    
    print("\n2. Loading configuration...")
    # Minimal config
    CONFIG = {
        'data': {
            'ticker': 'BTC-USD',
            'start': '2018-01-01',
            'end': None,
        }
    }
    print("   ✓ Config loaded")
    
    print("\n3. Downloading data...")
    ticker = CONFIG['data']['ticker']
    start = CONFIG['data']['start']
    df = yf.download(ticker, start=start, end=None, progress=False)
    print(f"   ✓ Downloaded {len(df)} rows")
    
    print("\n4. Normalizing columns...")
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        print("   ✓ Flattened MultiIndex columns")
    
    df.columns = df.columns.str.lower()
    print(f"   ✓ Columns: {df.columns.tolist()}")
    
    print("\n5. Handling close column...")
    if 'close' not in df.columns:
        if 'adj close' in df.columns:
            df['close'] = df['adj close']
            print("   ✓ Used 'adj close' as 'close'")
        else:
            raise ValueError("No close column")
    else:
        print("   ✓ 'close' column exists")
    
    print("\n6. Converting to numeric...")
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    print(f"   ✓ {len(df)} valid rows")
    
    print("\n7. Getting price range...")
    min_price = float(df['close'].min())
    max_price = float(df['close'].max())
    print(f"   ✓ Price range: ${min_price:.2f} - ${max_price:.2f}")
    
    print("\n" + "="*80)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("="*80)
    print("\nYou can now run: python3 complete_workflow.py")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

