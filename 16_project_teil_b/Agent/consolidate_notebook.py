#!/usr/bin/env python3
"""
Consolidate experiment_framework code into the notebook
Removes external dependency on experiment_framework.py
"""
import json

# Read notebook
with open("Project_Part_2_Final_Architecture.ipynb", "r") as f:
    nb = json.load(f)

# Complete experiment framework code to inject
experiment_code = '''# ============================================================
# EXPERIMENT FRAMEWORK (from experiment_framework.py)
# ============================================================

class ExperimentConfig:
    """Centralized experiment configuration"""
    TICKER = "BTC-USD"
    INITIAL_EQUITY = 100000.0
    FEE = 0.0005
    KAPPA = 0.1
    LEVERAGE_MAX = 1.0
    PPO_LEARNING_RATE = 3e-4
    PPO_N_STEPS = 2048
    PPO_BATCH_SIZE = 64
    PPO_N_EPOCHS = 10
    PPO_TOTAL_TIMESTEPS = 100000
    PPO_ENT_COEF = 0.01
    PPO_GAMMA = 0.99
    PPO_GAE_LAMBDA = 0.95
    TEST_SPLIT = 0.8

class MetricsCalculator:
    """Calculate performance metrics"""
    @staticmethod
    def calculate_metrics(equity_curve, returns):
        """Calculate comprehensive metrics"""
        final_equity = equity_curve[-1]
        initial_equity = equity_curve[0]
        total_return = (final_equity - initial_equity) / initial_equity
        
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'final_equity': float(final_equity),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'volatility': float(volatility),
            'max_drawdown': float(abs(max_drawdown)),
        }

class ExperimentRunner:
    """Orchestrates experimental runs"""
    
    def __init__(self, config=None):
        self.config = config or ExperimentConfig()
        self.results = {}
    
    def run_forecast_only_experiment(self, df_test, forecast_signal):
        """Run forecast-only baseline"""
        print("\\n" + "="*70)
        print("EXPERIMENT: Forecast-Only Strategy")
        print("="*70)
        
        equity = [self.config.INITIAL_EQUITY]
        positions = [0.0]
        returns = []
        
        for t in range(1, len(df_test)):
            target_pos = 1.0 if forecast_signal[t] > 0.5 else -1.0
            positions.append(target_pos)
            
            if t > 0:
                current_price = df_test['close'].iloc[t-1]
                next_price = df_test['close'].iloc[t]
                pnl = positions[-2] * (next_price - current_price)
                equity.append(equity[-1] + pnl)
            else:
                equity.append(equity[0])
        
        equity = np.array(equity)
        returns = np.diff(equity) / equity[:-1]
        
        metrics = MetricsCalculator.calculate_metrics(equity, returns)
        metrics['turnover'] = np.mean(np.abs(np.diff(positions)))
        
        self.results['forecast-only'] = metrics
        
        print(f"✓ Return: {metrics['total_return']*100:.2f}%")
        print(f"✓ Sharpe: {metrics['sharpe_ratio']:.4f}")
        
        return metrics
    
    def run_ppo_experiment(self, df_train, df_test, forecast_signal=None, run_name="ppo"):
        """Run PPO training and evaluation"""
        include_forecast = forecast_signal is not None
        exp_name = f"ppo-{'with' if include_forecast else 'without'}-forecast"
        
        print(f"\\n{'='*70}")
        print(f"EXPERIMENT: {exp_name.upper()}")
        print(f"{'='*70}")
        
        # Create training environment
        env = TradingEnv_2(
            df_train,
            fee=self.config.FEE,
            kappa=self.config.KAPPA,
            slippage_coef=0.0,
            smoothing_alpha=1.0,
            max_leverage=self.config.LEVERAGE_MAX,
            reward_scale=1.0,
            include_turnover=False,
            initial_equity=self.config.INITIAL_EQUITY,
            forecast_probs=forecast_signal[:len(df_train)] if forecast_signal is not None else None,
        )
        
        # Train PPO
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.PPO_LEARNING_RATE,
            n_steps=self.config.PPO_N_STEPS,
            batch_size=self.config.PPO_BATCH_SIZE,
            n_epochs=self.config.PPO_N_EPOCHS,
            ent_coef=self.config.PPO_ENT_COEF,
            gamma=self.config.PPO_GAMMA,
            gae_lambda=self.config.PPO_GAE_LAMBDA,
            verbose=0
        )
        
        model.learn(total_timesteps=self.config.PPO_TOTAL_TIMESTEPS)
        print(f"✓ Training complete")
        
        # Evaluate on test set
        env_test = TradingEnv_2(
            df_test,
            fee=self.config.FEE,
            kappa=self.config.KAPPA,
            slippage_coef=0.0,
            smoothing_alpha=1.0,
            max_leverage=self.config.LEVERAGE_MAX,
            reward_scale=1.0,
            include_turnover=False,
            initial_equity=self.config.INITIAL_EQUITY,
            forecast_probs=forecast_signal[len(df_train):] if forecast_signal is not None else None,
        )
        
        obs, _ = env_test.reset()
        equity = [env_test.equity]
        positions = []
        
        done = False
        while not done and env_test.t < len(df_test) - 1:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env_test.step(action)
            equity.append(env_test.equity)
            positions.append(env_test.pos)
        
        equity = np.array(equity)
        returns = np.diff(equity) / equity[:-1]
        
        metrics = MetricsCalculator.calculate_metrics(equity, returns)
        metrics['turnover'] = np.mean(np.abs(np.diff(positions)))
        
        self.results[exp_name] = metrics
        
        print(f"✓ Return: {metrics['total_return']*100:.2f}%")
        print(f"✓ Sharpe: {metrics['sharpe_ratio']:.4f}")
        
        return metrics, model
    
    def run_all_experiments(self, df_test, forecast_signal):
        """Run all configured experiments"""
        print("\\n" + "="*80)
        print("STARTING COMPREHENSIVE EXPERIMENT SUITE")
        print("="*80)
        
        split_idx = int(len(df_test) * self.config.TEST_SPLIT)
        df_train = df_test.iloc[:split_idx]
        df_test_eval = df_test.iloc[split_idx:]
        
        # Experiment 1: Forecast-Only
        self.run_forecast_only_experiment(df_test_eval, forecast_signal[split_idx:])
        
        # Experiment 2: PPO Without Forecast
        self.run_ppo_experiment(df_train, df_test_eval, forecast_signal=None, run_name="ppo-no-forecast")
        
        # Experiment 3: PPO With Forecast
        self.run_ppo_experiment(df_train, df_test_eval, forecast_signal=forecast_signal, run_name="ppo-with-forecast")
        
        return self.results

print("✓ Experiment classes loaded (from consolidated code)")
'''

# Find where to insert this code - look for any existing import or class definition
insert_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        # Find first code cell that mentions TradingEnv_2 class
        if 'class TradingEnv_2' in source:
            insert_index = i + 1
            break

if insert_index is None:
    # Otherwise, find the last cell and append
    insert_index = len(nb['cells'])

print(f"Inserting experiment framework code at cell index {insert_index}")

# Create new cell with experiment code
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": experiment_code.split('\n')
}

# Insert the cell
nb['cells'].insert(insert_index, new_cell)

# Save updated notebook
with open("Project_Part_2_Final_Architecture.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("✓ Notebook updated with consolidated experiment framework")
print(f"  Total cells: {len(nb['cells'])}")

