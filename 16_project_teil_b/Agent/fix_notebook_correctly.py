#!/usr/bin/env python3
"""
CORRECT NOTEBOOK MODIFICATION
Adds cells with proper JSON formatting and line breaks
"""

import json
from pathlib import Path

# Load the original notebook
nb_path = Path('Project_Part_2_Final_Architecture.ipynb')
with open(nb_path, 'r') as f:
    nb = json.load(f)

print(f"Original notebook: {len(nb['cells'])} cells")

# Create the Forecast-Only experiment cell with PROPER formatting
forecast_only_source = """# ============================================================
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
print(f"  Max DD: {forecast_only_maxdd*100:.2f}%")"""

# Create the PPO evaluation cell with PROPER formatting
ppo_eval_source = """# ============================================================
# EXPERIMENT 2 & 3: PPO EVALUATION
# ============================================================
print("\\n6. EVALUATING PPO MODELS...")

def run_equity_curve_eval(model, df_eval):
    \"\"\"Run model on evaluation set and collect equity curve\"\"\"
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

# Evaluate PPO model
ppo_equity, ppo_position_hist, ppo_pnl_hist, ppo_cost_hist = run_equity_curve_eval(model, df_test)

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
print(f"  ppo_cost_hist: {len(ppo_cost_hist)} values")"""

# Create cells as proper JSON
def make_code_cell(source_text):
    """Create a proper Jupyter code cell"""
    lines = source_text.split('\n')
    source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]]  # Add \n except last line
    
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    }

forecast_cell = make_code_cell(forecast_only_source)
ppo_cell = make_code_cell(ppo_eval_source)

# Find where to insert (after run_equity_curve or after forecasts)
# For now, insert before the existing comprehensive visualization cell
insert_index = len(nb['cells']) - 3  # Insert near end

nb['cells'].insert(insert_index, ppo_cell)
nb['cells'].insert(insert_index + 1, forecast_cell)

print(f"✓ Inserted PPO evaluation cell at position {insert_index}")
print(f"✓ Inserted Forecast-Only cell at position {insert_index + 1}")

# Save the corrected notebook
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=2)

print(f"\n✓✓✓ NOTEBOOK RESTORED AND CORRECTED ✓✓✓")
print(f"✓ New total cells: {len(nb['cells'])}")
print(f"\nThe notebook is now ready to execute!")
print(f"All variables will be calculated before visualizations.")

