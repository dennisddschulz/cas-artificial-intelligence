#!/usr/bin/env python3
"""
Script to add missing experiment code cells to the notebook
Inserts code that calculates:
- forecast_only_equity, forecast_only_positions, forecast_only_returns
- forecast_only_return, forecast_only_sharpe, forecast_only_maxdd
- ppo_equity, ppo_position_hist, ppo_pnl_hist, ppo_cost_hist
"""

import json
from pathlib import Path

# Load notebook
nb_path = Path('Project_Part_2_Final_Architecture.ipynb')
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Code to add for Forecast-Only experiment
forecast_only_code = '''# ============================================================
# EXPERIMENT 1: FORECAST-ONLY STRATEGY
# ============================================================
print("\\n5. EXPERIMENT 1: FORECAST-ONLY STRATEGY...")

# Run forecast-only strategy on test set
forecast_only_equity = [INITIAL_EQUITY]
forecast_only_positions = []
cash = INITIAL_EQUITY
pos = 0.0

for t in range(len(df_test)):
    signal = forecast_probs_aligned[t]  # Use aligned forecasts
    new_pos = 1.0 if signal > 0.5 else -1.0
    forecast_only_positions.append(new_pos)
    
    price = df_test['close'].iloc[t]
    if t > 0:
        prev_price = df_test['close'].iloc[t-1]
        pnl_step = pos * (price - prev_price)
        cash += pnl_step
    
    pos = new_pos
    equity_val = cash + pos * price
    forecast_only_equity.append(equity_val)

forecast_only_equity = np.array(forecast_only_equity)
forecast_only_positions = np.array(forecast_only_positions)

# Calculate returns
forecast_only_returns = np.diff(forecast_only_equity) / forecast_only_equity[:-1]

# Calculate metrics
forecast_only_return = (forecast_only_equity[-1] - forecast_only_equity[0]) / forecast_only_equity[0]
if len(forecast_only_returns) > 0 and np.std(forecast_only_returns) > 0:
    forecast_only_sharpe = np.mean(forecast_only_returns) / np.std(forecast_only_returns) * np.sqrt(252)
else:
    forecast_only_sharpe = 0.0

drawdown = (forecast_only_equity - np.maximum.accumulate(forecast_only_equity)) / np.maximum.accumulate(forecast_only_equity)
forecast_only_maxdd = np.min(drawdown)

print(f"✓ Forecast-Only Strategy:")
print(f"  Return: {forecast_only_return*100:.2f}%")
print(f"  Sharpe: {forecast_only_sharpe:.4f}")
print(f"  Max DD: {forecast_only_maxdd*100:.2f}%")
'''

# Code to add for PPO evaluation without forecast
ppo_no_forecast_code = '''# ============================================================
# EXPERIMENT 2 & 3: PPO EVALUATION (Without and With Forecast)
# ============================================================
print("\\n6. EVALUATING PPO MODELS...")

def run_equity_curve(model, df_eval):
    """Run model on evaluation set and collect equity curve"""
    env_eval = TradingEnv_2(
        df_eval,
        fee=FEE,
        kappa=KAPPA,
        max_leverage=LEVERAGE_MAX,
    )
    obs, _ = env_eval.reset()
    done = False
    
    equity = [env_eval.equity]
    pos_hist = [env_eval.pos]
    pnl_hist = []
    cost_hist = []
    
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist, _ = model(obs_t)
            u = dist.mean
            a = squash(u)
            a_scaled = a * LEVERAGE_MAX
        
        obs, reward, terminated, truncated, info = env_eval.step(a_scaled.detach().cpu().numpy()[0])
        done = terminated or truncated
        equity.append(env_eval.equity)
        pos_hist.append(env_eval.pos)
        pnl_hist.append(info.get('pnl', 0.0))
        cost_hist.append(info.get('cost', 0.0))
    
    return np.array(equity), np.array(pos_hist), np.array(pnl_hist), np.array(cost_hist)

# Evaluate PPO model (without forecast signal in this case)
ppo_equity, ppo_position_hist, ppo_pnl_hist, ppo_cost_hist = run_equity_curve(model, df_test)

ppo_return = (ppo_equity[-1] - ppo_equity[0]) / ppo_equity[0]
ppo_returns = np.diff(ppo_equity) / ppo_equity[:-1]
if len(ppo_returns) > 0 and np.std(ppo_returns) > 0:
    ppo_sharpe = np.mean(ppo_returns) / np.std(ppo_returns) * np.sqrt(252)
else:
    ppo_sharpe = 0.0

ppo_drawdown = (ppo_equity - np.maximum.accumulate(ppo_equity)) / np.maximum.accumulate(ppo_equity)
ppo_maxdd = np.min(ppo_drawdown)

print(f"✓ PPO Model:")
print(f"  Return: {ppo_return*100:.2f}%")
print(f"  Sharpe: {ppo_sharpe:.4f}")
print(f"  Max DD: {ppo_maxdd*100:.2f}%")

print(f"\\n✓ ALL VARIABLES NOW SET:")
print(f"  forecast_only_equity: {len(forecast_only_equity)} values")
print(f"  forecast_only_positions: {len(forecast_only_positions)} values")
print(f"  forecast_only_returns: {len(forecast_only_returns)} values")
print(f"  ppo_equity: {len(ppo_equity)} values")
print(f"  ppo_position_hist: {len(ppo_position_hist)} values")
print(f"  ppo_pnl_hist: {len(ppo_pnl_hist)} values")
print(f"  ppo_cost_hist: {len(ppo_cost_hist)} values")
'''

# Find cell right after run_equity_curve definition (around line 1350-1400)
# Look for "def run_equity_curve" cell
insert_forecast_cell_index = None
insert_ppo_cell_index = None

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Find where to insert forecast-only code (after forecasts are generated)
        if 'forecast_probs_aligned' in source and 'LSTM Forecasts Prepared' in source:
            insert_forecast_cell_index = i + 1
        
        # Find where to insert PPO evaluation (after run_equity_curve definition)
        if 'def run_equity_curve' in source:
            insert_ppo_cell_index = i + 1

print(f"Would insert forecast-only code after cell: {insert_forecast_cell_index}")
print(f"Would insert PPO code after cell: {insert_ppo_cell_index}")

# Create new cells
def make_code_cell(code, metadata=None):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': metadata or {},
        'outputs': [],
        'source': code.split('\n')
    }

# Insert cells (in reverse order to maintain indices)
if insert_ppo_cell_index:
    nb['cells'].insert(insert_ppo_cell_index, make_code_cell(ppo_no_forecast_code))
    print(f"✓ Inserted PPO evaluation code at cell {insert_ppo_cell_index}")

if insert_forecast_cell_index:
    nb['cells'].insert(insert_forecast_cell_index, make_code_cell(forecast_only_code))
    print(f"✓ Inserted Forecast-Only code at cell {insert_forecast_cell_index}")

# Save modified notebook
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✓ Notebook updated: {nb_path}")
print(f"✓ Total cells: {len(nb['cells'])}")
print(f"\nAll required variables will now be calculated before visualizations!")

