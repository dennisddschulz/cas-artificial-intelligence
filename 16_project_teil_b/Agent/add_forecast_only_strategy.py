#!/usr/bin/env python3
"""
Add the forecast-only strategy implementation to the notebook
"""
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

# The code cell to be added
forecast_only_code = '''# ============================================================
# FORECAST-ONLY STRATEGY IMPLEMENTATION
# ============================================================
# This strategy simply buys when LSTM forecasts up movement (prob > 0.5)
# and sells when it forecasts down movement (prob < 0.5)

print("\\n" + "="*70)
print("Computing Forecast-Only Strategy Equity Curve")
print("="*70)

# Initialize tracking
forecast_only_equity = [INITIAL_EQUITY]
forecast_only_positions = [0.0]
forecast_only_cash = [INITIAL_EQUITY]
forecast_only_returns = []

current_equity = INITIAL_EQUITY
current_cash = INITIAL_EQUITY
current_position = 0.0
last_price = df_test['close'].iloc[0]

# Simulate the forecast-only strategy
for t in range(1, len(df_test)):
    current_price = df_test['close'].iloc[t]
    forecast_prob = forecast_probs_aligned[t]
    
    # Simple rule: position = 1.0 if bullish forecast, -1.0 if bearish
    target_position = 1.0 if forecast_prob > 0.5 else -1.0
    
    # Calculate position change
    position_change = target_position - current_position
    
    # Transaction cost: fee + kappa penalty for position change
    transaction_cost = abs(position_change) * current_price * FEE
    if position_change != 0:
        kappa_penalty = abs(position_change) * current_price * KAPPA
    else:
        kappa_penalty = 0.0
    
    total_cost = transaction_cost + kappa_penalty
    
    # Update cash (cost reduces equity)
    current_cash -= total_cost
    
    # Price movement PnL
    price_move = current_price - last_price
    pnl = current_position * price_move
    current_cash += pnl
    
    # Update equity
    current_equity = current_cash + current_position * current_price
    
    # Store history
    forecast_only_equity.append(current_equity)
    forecast_only_positions.append(target_position)
    forecast_only_cash.append(current_cash)
    
    # Daily return
    if len(forecast_only_equity) > 1:
        daily_return = (forecast_only_equity[-1] - forecast_only_equity[-2]) / forecast_only_equity[-2]
        forecast_only_returns.append(daily_return)
    
    # Update for next iteration
    current_position = target_position
    last_price = current_price

# Convert to numpy arrays
forecast_only_equity = np.array(forecast_only_equity)
forecast_only_positions = np.array(forecast_only_positions)
forecast_only_returns = np.array(forecast_only_returns)

# Calculate metrics
forecast_only_return = (forecast_only_equity[-1] - INITIAL_EQUITY) / INITIAL_EQUITY
forecast_only_sharpe = (
    np.mean(forecast_only_returns) / np.std(forecast_only_returns) * np.sqrt(252)
    if len(forecast_only_returns) > 1 and np.std(forecast_only_returns) > 0
    else 0.0
)
forecast_only_maxdd = (1.0 - np.min(forecast_only_equity) / np.max(forecast_only_equity[:np.argmin(forecast_only_equity)])) if len(forecast_only_equity) > 0 else 0.0

print(f"\\n✓ Forecast-Only Strategy Results:")
print(f"  Final Equity: ${forecast_only_equity[-1]:,.2f}")
print(f"  Total Return: {forecast_only_return*100:.2f}%")
print(f"  Sharpe Ratio: {forecast_only_sharpe:.4f}")
print(f"  Max Drawdown: {forecast_only_maxdd*100:.2f}%")
print(f"  Total Trades: {np.sum(np.abs(np.diff(forecast_only_positions)) > 0.1)}")
print()
'''

try:
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the cell containing "Aligned forecast probabilities" and add the new cell after it
    inserted = False
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Aligned forecast probabilities' in source and 'forecast_probs_aligned' in source:
                print(f"Found alignment cell at index {idx}", file=sys.stderr)
                
                # Create new code cell with the forecast-only strategy implementation
                new_cell = {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': forecast_only_code.split('\n')
                }
                
                # Insert the new cell after the current one
                nb['cells'].insert(idx + 1, new_cell)
                
                print(f"Inserted forecast-only strategy cell after index {idx}", file=sys.stderr)
                inserted = True
                break
    
    if not inserted:
        print("WARNING: Could not find the alignment cell. Adding to end of notebook.", file=sys.stderr)
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': forecast_only_code.split('\n')
        }
        nb['cells'].append(new_cell)
    
    # Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("\n✓ Successfully added forecast-only strategy implementation!", file=sys.stderr)
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

