#!/usr/bin/env python3
"""
Add a variant that trains PPO WITHOUT forecast for comparison
"""
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

ppo_without_forecast_code = '''# ============================================================
# PPO WITHOUT FORECAST - BASELINE COMPARISON
# ============================================================
# Train a PPO agent WITHOUT the LSTM forecast signal
# This allows us to isolate the impact of forecast information on RL performance

print("\\n" + "="*70)
print("Training PPO Agent WITHOUT Forecast (Baseline)")
print("="*70)

# Create environment without forecast feature
# We'll create a simpler observation that excludes forecast_prob

class TradingEnv_Baseline(gym.Env):
    """Trading environment WITHOUT forecast signal for baseline comparison"""
    
    def __init__(self, df, fee=0.0005, kappa=0.1, max_leverage=1.0, initial_equity=100000.0):
        self.df = df.reset_index(drop=True)
        self.fee = fee
        self.kappa = kappa
        self.max_leverage = max_leverage
        self.initial_equity = float(initial_equity)
        
        # State: [position, cash/equity, price_momentum, volatility] - NO FORECAST
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-max_leverage, high=max_leverage, shape=(1,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        self.t = 0
        self.equity = self.initial_equity
        self.cash = self.initial_equity
        self.pos = 0.0
        self.peak_equity = self.initial_equity
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Observation WITHOUT forecast - baseline features only"""
        price = self.df['close'].iloc[self.t]
        
        # Momentum (price change over last 5 days)
        if self.t >= 5:
            momentum = (price - self.df['close'].iloc[self.t - 5]) / self.df['close'].iloc[self.t - 5]
        else:
            momentum = 0.0
        
        # Volatility (std of returns over last 20 days)
        if self.t >= 20:
            returns = self.df['close'].iloc[self.t-20:self.t].pct_change().dropna()
            volatility = np.std(returns) if len(returns) > 0 else 0.01
        else:
            volatility = 0.01
        
        # Normalized state
        obs = np.array([
            self.pos,                              # Current position
            self.cash / self.equity if self.equity > 0 else 0.5,  # Cash ratio
            momentum,                               # Momentum (NO FORECAST!)
            volatility,                             # Volatility (NO FORECAST!)
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        action = float(action[0])
        target_pos = np.clip(action, -self.max_leverage, self.max_leverage)
        
        price = self.df['close'].iloc[self.t]
        
        # Calculate costs for position change
        position_change = target_pos - self.pos
        transaction_cost = abs(position_change) * price * self.fee
        kappa_penalty = abs(position_change) * price * self.kappa if position_change != 0 else 0
        
        # Update position
        old_pos = self.pos
        self.pos = target_pos
        
        # Update cash (transaction costs reduce cash)
        self.cash -= (transaction_cost + kappa_penalty)
        
        # Move to next time step
        self.t += 1
        
        if self.t < len(self.df):
            new_price = self.df['close'].iloc[self.t]
            price_move = new_price - price
            
            # PnL from position
            pnl = self.pos * price_move
            self.cash += pnl
        
        # Calculate equity
        self.equity = self.cash + self.pos * self.df['close'].iloc[self.t] if self.t < len(self.df) else self.cash
        
        # Reward: change in equity
        reward = (self.equity / self.initial_equity - 1.0)
        
        # Penalty for large drawdowns
        drawdown = 1.0 - self.equity / self.peak_equity if self.peak_equity > 0 else 0
        risk_penalty = -0.1 * drawdown
        
        self.peak_equity = max(self.peak_equity, self.equity)
        
        total_reward = reward + risk_penalty
        terminated = self.equity <= 0 or self.t >= len(self.df) - 1
        
        return self._get_obs(), total_reward, terminated, False, {}


# Train PPO without forecast
print("\\nCreating baseline trading environment (without forecast)...")
env_baseline = TradingEnv_Baseline(
    df_test,
    fee=FEE,
    kappa=KAPPA,
    max_leverage=LEVERAGE_MAX,
    initial_equity=INITIAL_EQUITY
)

print(f"Environment observation space: {env_baseline.observation_space}")
print(f"Environment action space: {env_baseline.action_space}")

# Create a simpler PPO model for baseline (same architecture, just different input size)
print("\\nTraining PPO agent without forecast signal...")
print("(Training for 5000 steps for quick baseline comparison)")

model_baseline = PPO(
    "MlpPolicy",
    env_baseline,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,
    verbose=0
)

# Quick training
model_baseline.learn(total_timesteps=5000)

print("✓ Baseline PPO training complete!")

# Evaluate baseline model on test data
print("\\nEvaluating baseline PPO (without forecast)...")
obs_baseline, _ = env_baseline.reset()
equity_baseline = [env_baseline.equity]
pos_baseline = [env_baseline.pos]
returns_baseline = []

while env_baseline.t < len(df_test) - 1:
    obs_t = torch.as_tensor(obs_baseline, dtype=torch.float32, device=device).unsqueeze(0)
    action, _ = model_baseline.policy(obs_t)
    obs_baseline, reward, terminated, _, _ = env_baseline.step(action.detach().cpu().numpy())
    
    equity_baseline.append(env_baseline.equity)
    pos_baseline.append(env_baseline.pos)
    
    if len(equity_baseline) > 1:
        daily_return = (equity_baseline[-1] - equity_baseline[-2]) / equity_baseline[-2]
        returns_baseline.append(daily_return)
    
    if terminated:
        break

equity_baseline = np.array(equity_baseline)
pos_baseline = np.array(pos_baseline)
returns_baseline = np.array(returns_baseline)

# Calculate baseline metrics
baseline_return = (equity_baseline[-1] - INITIAL_EQUITY) / INITIAL_EQUITY
baseline_sharpe = (np.mean(returns_baseline) / np.std(returns_baseline) * np.sqrt(252)) if len(returns_baseline) > 1 and np.std(returns_baseline) > 0 else 0.0
baseline_vol = np.std(returns_baseline) * np.sqrt(252) if len(returns_baseline) > 0 else 0.0
baseline_maxdd = 1.0 - np.min(equity_baseline) / np.max(equity_baseline[:np.argmin(np.maximum.cumsum(np.zeros(len(equity_baseline))))])
baseline_turnover = np.sum(np.abs(np.diff(pos_baseline))) / len(pos_baseline)

print(f"\\n✓ Baseline Results (WITHOUT Forecast):")
print(f"  Final Equity: ${equity_baseline[-1]:,.2f}")
print(f"  Total Return: {baseline_return*100:.2f}%")
print(f"  Sharpe Ratio: {baseline_sharpe:.4f}")
print(f"  Volatility: {baseline_vol*100:.2f}%")
print(f"  Max Drawdown: {baseline_maxdd*100:.2f}%")
print(f"  Turnover: {baseline_turnover:.4f}")

print("\\n" + "="*70)
'''

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find a good place to insert this - after the main PPO training but before the comparison
    # Insert it before the comparison analysis cell
    comparison_idx = None
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'COMPREHENSIVE STRATEGY COMPARISON' in source:
                comparison_idx = idx
                break
    
    if comparison_idx is not None:
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line if line.endswith('\n') else line + '\n' for line in ppo_without_forecast_code.split('\n')[:-1]]
        }
        
        nb['cells'].insert(comparison_idx, new_cell)
        print(f"Inserted baseline PPO cell at index {comparison_idx}", file=sys.stderr)
    else:
        print("WARNING: Could not find comparison cell. Appending baseline PPO code.", file=sys.stderr)
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line if line.endswith('\n') else line + '\n' for line in ppo_without_forecast_code.split('\n')[:-1]]
        }
        nb['cells'].append(new_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("\n✓ Successfully added PPO baseline (without forecast)!", file=sys.stderr)
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

