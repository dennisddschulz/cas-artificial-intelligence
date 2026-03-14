#!/usr/bin/env python3
import sys
try:
    import yfinance as yf
    import pandas as pd
    
    ticker = 'BTC-USD'
    start = '2018-01-01'
    df = yf.download(ticker, start=start, end=None, progress=False)
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Handle close column
    if 'close' not in df.columns and 'adj close' in df.columns:
        df['close'] = df['adj close']
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    
    print(f"✓ Loaded {len(df)} days")
    print(f"✓ Price: {float(df['close'].min()):.2f} - {float(df['close'].max()):.2f}")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

