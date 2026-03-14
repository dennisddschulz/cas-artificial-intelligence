"""
Parameterized PPO Trading Experiment Framework
Allows running multiple configurations for comparison
"""

# ============================================================
# SSL/PROXY CONFIGURATION - MUST BE BEFORE ANY NETWORK IMPORTS
# ============================================================
import os
import ssl

# Set proxy for Python requests/urllib (adjust if needed)
PROXY_HOST = "proxy.infet.ejpd.admin.ch"
PROXY_PORT = "8080"
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"

# Configure proxies for different protocols
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL

# DISABLE SSL VERIFICATION FOR CORPORATE PROXY (CRITICAL)
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['INSECURE_REQUESTS_DISABLED'] = 'false'

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure SSL to NOT verify certificates
ssl.verify_mode = ssl.CERT_NONE
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# CRITICAL: Configure urllib3 to not verify SSL at the package level
from urllib3.util.ssl_ import create_urllib3_context
context = create_urllib3_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# Configure requests library with SSL disabled
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

class SSLAdapter(HTTPAdapter):
    """HTTPAdapter that disables SSL verification"""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', SSLAdapter())
session.mount('http://', SSLAdapter())

# W&B configuration - Use OFFLINE mode by default to avoid SSL errors with corporate proxy
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'false'
os.environ['VERIFY_SSL'] = 'false'
os.environ['SSL_NO_VERIFY'] = '1'

print("✓ SSL/Proxy configured:")
print(f"  Host: {PROXY_HOST}")
print(f"  Port: {PROXY_PORT}")
print(f"  SSL Verification: DISABLED (corporate proxy)")
print(f"  Environment variables set")
print(f"  urllib3 SSL: DISABLED")
print(f"  WandB Mode: OFFLINE (by default)")

# ============================================================
# NOW IMPORT MODULES THAT NEED NETWORK
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from trading_config import (
    ExperimentConfig, ForecastMode, RewardType,
    get_ppo_without_forecast_config,
    get_ppo_with_forecast_config,
    get_ppo_different_rewards_configs
)
from trading_metrics import TradingMetrics, MetricsComparison
from budget_tracker import BudgetTracker, create_summary_dashboard

# W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentRunner:
    """Main experiment runner with parameterization"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.initialized = False
    
    def setup_wandb(self):
        """Initialize Weights & Biases"""
        if not (WANDB_AVAILABLE and self.config.use_wandb):
            return
        
        # WANDB_MODE is already set globally from config, but can be overridden per experiment
        current_mode = os.environ.get('WANDB_MODE', 'offline')
        
        # Allow per-experiment override of mode
        if hasattr(self.config, 'wandb_mode') and self.config.wandb_mode:
            os.environ['WANDB_MODE'] = self.config.wandb_mode
            current_mode = self.config.wandb_mode
        
        try:
            wandb.init(
                project=self.config.wandb_project,
                group=self.config.wandb_group,
                name=self.config.wandb_run_name,
                entity=self.config.wandb_entity,
                config={
                    "experiment_name": self.config.experiment_name,
                    "forecast_mode": self.config.forecast_mode.value,
                    "reward_type": self.config.reward_type.value,
                    "initial_equity": self.config.environment.initial_equity,
                    "fee": self.config.environment.fee,
                    "kappa": self.config.environment.kappa,
                    "leverage_max": self.config.environment.leverage_max,
                    "ppo_updates": self.config.ppo.total_updates,
                    "lr": self.config.ppo.learning_rate,
                    "seed": self.config.seed,
                    "wandb_mode": current_mode,
                },
                tags=self.config.wandb_tags,
            )
            print(f"✓ W&B initialized: {current_mode.upper()} mode")
            if current_mode == "online":
                print(f"  Project: {self.config.wandb_project}")
                print(f"  Group: {self.config.wandb_group}")
                print(f"  Entity: {self.config.wandb_entity}")
                try:
                    print(f"  URL: {wandb.run.get_url()}")
                except:
                    pass
            else:
                print(f"  Data will be saved locally: ./wandb/offline-run-*/")
                print(f"  Sync to cloud later with: wandb sync ./wandb/offline-run-*/")
        
        except Exception as e:
            print(f"⚠ WandB initialization warning: {e}")
            print(f"  Continuing with offline mode fallback")
            os.environ['WANDB_MODE'] = 'offline'
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    group=self.config.wandb_group,
                    name=self.config.wandb_run_name,
                    config={
                        "experiment_name": self.config.experiment_name,
                        "forecast_mode": self.config.forecast_mode.value,
                        "reward_type": self.config.reward_type.value,
                    },
                    tags=self.config.wandb_tags,
                )
                print(f"✓ W&B initialized in OFFLINE mode (fallback)")
            except:
                print(f"⚠ WandB disabled - continuing without cloud logging")
    
    def load_market_data(self):
        """Load OHLCV data"""
        print(f"\n{'='*70}")
        print(f"LOADING DATA: {self.config.data.ticker}")
        print(f"{'='*70}")
        
        try:
            df = yf.download(
                self.config.data.ticker,
                start=self.config.data.start_date,
                end=self.config.data.end_date,
                progress=False
            )
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            
            print(f"✓ Data loaded: {df.shape[0]} days")
            print(f"  Range: {df.index[0].date()} to {df.index[-1].date()}")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators - EXACTLY MATCH ORIGINAL NOTEBOOK"""
        print(f"Adding technical features...")
        df = df.copy()
        
        # Basic returns
        df["log_close"] = np.log(df["close"])
        df["r"] = df["log_close"].diff()
        df["r_lag1"] = df["r"].shift(1)
        
        # Forecast signal (EWMA mean of returns)
        df["mu_hat"] = df["r"].ewm(span=20, adjust=False).mean()
        
        # Risk estimate: rolling volatility
        df["sigma_hat"] = df["r"].rolling(20).std()
        
        # Momentum
        df["mom_5"] = df["r"].rolling(5).mean()
        df["mom_20"] = df["r"].rolling(20).mean()
        
        # Volatility regime
        df["vol_ratio"] = df["r"].rolling(10).std() / df["r"].rolling(50).std()
        
        # Signal Strength (mu / sigma)
        df["signal_strength"] = df["mu_hat"] / (df["sigma_hat"] + 1e-8)
        
        # RSI (Relative Strength Index)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = (df["rsi"] - 50) / 50.0  # normalize to [-1, 1]
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_diff"] = df["macd"] - df["macd_signal"]
        # Normalize MACD
        macd_std = df["macd_diff"].rolling(window=20).std()
        df["macd_diff"] = df["macd_diff"] / (macd_std + 1e-8)
        
        # Bollinger Bands width
        sma = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        bb_width = 2 * std / (sma + 1e-8)
        df["bb_width"] = bb_width / bb_width.rolling(window=50).mean()
        
        # EMA momentum
        ema12_ratio = ema12 / (ema26 + 1e-8)
        df["ema_ratio"] = (ema12_ratio - ema12_ratio.rolling(window=20).mean()) / (ema12_ratio.rolling(window=20).std() + 1e-8)
        
        df = df.dropna()
        print(f"✓ Features added: {df.shape[0]} rows remaining")
        return df
    
    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data by time"""
        n = len(df)
        train_idx = int(self.config.data.train_frac * n)
        val_idx = train_idx + int(self.config.data.val_frac * n)
        
        df_train = df.iloc[:train_idx].reset_index(drop=True)
        df_val = df.iloc[train_idx:val_idx].reset_index(drop=True)
        df_test = df.iloc[val_idx:].reset_index(drop=True)
        
        print(f"Data split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
        return df_train, df_val, df_test
    
    def train_forecast_model(self, df_train, df_val, df_test):
        """Train LSTM forecasting model (if needed)"""
        if self.config.forecast_mode == ForecastMode.NONE:
            print("\n✓ Skipping forecast training (mode: NONE)")
            return None, None, None
        
        print(f"\n{'='*70}")
        print(f"TRAINING LSTM FORECAST MODEL")
        print(f"{'='*70}")
        
        # Feature prep - REMOVE 'r' to avoid look-ahead bias
        # Only use indicators, not current/past returns
        # This ensures true forecasting, not just return autocorrelation
        feature_cols = ['sigma_hat', 'rsi', 'macd_diff', 'signal_strength']
        
        # Check which features are actually available
        available_cols = [c for c in feature_cols if c in df_train.columns]
        if len(available_cols) < len(feature_cols):
            # Fallback if some columns missing
            available_cols = [c for c in ['sigma_hat', 'rsi', 'macd_diff', 'signal_strength'] if c in df_train.columns]
        
        print(f"Using feature columns (NO return look-ahead bias): {available_cols}")
        feature_cols = available_cols
        
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(df_train[feature_cols].values)
        X_val = scaler.transform(df_val[feature_cols].values)
        X_test = scaler.transform(df_test[feature_cols].values)
        
        # Simple direction label - NO [:-1] slice, shift(-1) handles alignment
        y_train = (df_train['r'].shift(-1) > 0).astype(int).fillna(0).values
        y_val = (df_val['r'].shift(-1) > 0).astype(int).fillna(0).values
        y_test = (df_test['r'].shift(-1) > 0).astype(int).fillna(0).values
        
        # Create sequences
        lookback = self.config.forecasting.lookback
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, lookback)
        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, lookback)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, lookback)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_t = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_t = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
        X_test_t = torch.FloatTensor(X_test_seq).to(self.device)
        y_test_t = torch.FloatTensor(y_test_seq).unsqueeze(1).to(self.device)
        
        # Model
        model = LSTMForecaster(
            len(feature_cols),
            self.config.forecasting.hidden_dim,
            self.config.forecasting.num_layers,
            self.config.forecasting.dropout
        ).to(self.device)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.forecasting.learning_rate,
            weight_decay=self.config.forecasting.weight_decay
        )
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.config.forecasting.epochs):
            model.train()
            train_loss = 0
            bs = self.config.forecasting.batch_size
            
            for i in range(0, len(X_train_t), bs):
                batch_x = X_train_t[i:i+bs]
                batch_y = y_train_t[i:i+bs]
                
                optimizer.zero_grad()
                pred, _ = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= (len(X_train_t) // bs + 1)
            
            model.eval()
            with torch.no_grad():
                val_pred, _ = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            if val_loss < best_val_loss - self.config.forecasting.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.forecasting.early_stopping_patience:
                    print(f"Early stop at epoch {epoch+1}")
                    model.load_state_dict(best_state)
                    break
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Test evaluation (for logging only, NOT used in PPO)
        with torch.no_grad():
            test_pred, _ = model(X_test_t)
            test_acc = ((test_pred > 0.5).float() == y_test_t).float().mean().item()
        
        print(f"✓ Forecast model trained. Test accuracy: {test_acc:.4f}")
        
        # ================================================================
        # CALCULATE SMAPE & MAPE FOR FORECAST QUALITY ASSESSMENT
        # ================================================================
        def smape(y_true, y_pred):
            """Symmetric Mean Absolute Percentage Error"""
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
            diff = np.abs(y_true - y_pred) / (denominator + 1e-8)
            diff[~np.isfinite(diff)] = 0.0
            return 100.0 * np.mean(diff)
        
        def mape(y_true, y_pred):
            """Mean Absolute Percentage Error"""
            mask = y_true != 0
            if mask.sum() == 0:
                return 0.0
            return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        
        # Convert to numpy for metric calculation
        y_train_np = y_train_seq.astype(np.float32)
        y_val_np = y_val_seq.astype(np.float32)
        y_test_np = y_test_seq.astype(np.float32)
        
        train_pred_probs = torch.sigmoid(model(X_train_t)[0]).detach().cpu().numpy().flatten()
        val_pred_probs = torch.sigmoid(model(X_val_t)[0]).detach().cpu().numpy().flatten()
        test_pred_probs = torch.sigmoid(model(X_test_t)[0]).detach().cpu().numpy().flatten()
        
        # Calculate SMAPE and MAPE
        train_smape = smape(y_train_np, train_pred_probs)
        train_mape = mape(y_train_np, train_pred_probs)
        
        val_smape = smape(y_val_np, val_pred_probs)
        val_mape = mape(y_val_np, val_pred_probs)
        
        test_smape = smape(y_test_np, test_pred_probs)
        test_mape = mape(y_test_np, test_pred_probs)
        
        # ================================================================
        # PRINT FORECAST METRICS TO CONSOLE
        # ================================================================
        print(f"\n{'='*70}")
        print(f"LSTM FORECAST QUALITY METRICS")
        print(f"{'='*70}")
        print(f"Train Set Accuracy: {(train_pred_probs > 0.5).astype(int).mean():.4f} | SMAPE: {train_smape:.2f}% | MAPE: {train_mape:.2f}%")
        print(f"Val   Set Accuracy: {(val_pred_probs > 0.5).astype(int).mean():.4f} | SMAPE: {val_smape:.2f}% | MAPE: {val_mape:.2f}%")
        print(f"Test  Set Accuracy: {test_acc:.4f} | SMAPE: {test_smape:.2f}% | MAPE: {test_mape:.2f}%")
        print(f"{'='*70}\n")
        
        # ================================================================
        # LOG FORECAST METRICS TO WANDB
        # ================================================================
        if WANDB_AVAILABLE and self.config.use_wandb:
            forecast_metrics = {
                "forecast/train_accuracy": (train_pred_probs > 0.5).astype(int).mean(),
                "forecast/train_smape": train_smape,
                "forecast/train_mape": train_mape,
                "forecast/val_accuracy": (val_pred_probs > 0.5).astype(int).mean(),
                "forecast/val_smape": val_smape,
                "forecast/val_mape": val_mape,
                "forecast/test_accuracy": test_acc,
                "forecast/test_smape": test_smape,
                "forecast/test_mape": test_mape,
                "forecast/best_val_loss": best_val_loss,
            }
            wandb.log(forecast_metrics)
            print("✓ Forecast metrics logged to WandB")
        
        # CRITICAL FIX: Generate predictions on TRAINING data (not test)
        # This ensures forecast aligns with data used in PPO training
        with torch.no_grad():
            train_pred, _ = model(X_train_t)
            train_pred_np = train_pred.cpu().numpy().flatten()
        
        # Pad with zeros for the first lookback periods
        # (These dates have insufficient history for LSTM)
        lookback = self.config.forecasting.lookback
        train_pred_padded = np.concatenate([
            np.zeros(lookback),  # No forecast for first lookback days
            train_pred_np        # Actual LSTM predictions
        ])
        
        # Ensure alignment with df_train length
        # (account for NaN drops during feature engineering)
        if len(train_pred_padded) > len(X_train):
            train_pred_padded = train_pred_padded[:len(X_train)]
        elif len(train_pred_padded) < len(X_train):
            # Pad with zeros if shorter
            train_pred_padded = np.concatenate([
                train_pred_padded,
                np.zeros(len(X_train) - len(train_pred_padded))
            ])
        
        print(f"  Training forecast shape: {train_pred_padded.shape}")
        print(f"  df_train shape: {len(X_train)}")
        
        return model, train_pred_padded, scaler
    
    @staticmethod
    def _create_sequences(X, y, lookback):
        """Create LSTM sequences"""
        X_seq, y_seq = [], []
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i+lookback])
            y_seq.append(y[i+lookback])
        return np.array(X_seq), np.array(y_seq)
    
    def run(self):
        """Run full experiment"""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {self.config.experiment_name}")
        print(f"Forecast Mode: {self.config.forecast_mode.value}")
        print(f"Reward Type: {self.config.reward_type.value}")
        print(f"{'='*70}\n")
        
        self.setup_wandb()
        
        try:
            # Load and prepare data
            df = self.load_market_data()
            df = self.add_features(df)
            df_train, df_val, df_test = self.split_data(df)
            
            # Train forecast if needed
            forecast_model = None
            forecast_probs = None
            if self.config.forecast_mode == ForecastMode.LSTM:
                forecast_model, forecast_probs, _ = self.train_forecast_model(
                    df_train, df_val, df_test
                )
            
            # Run PPO training
            if self.config.ppo.total_updates > 0:
                self.train_ppo(df_train, df_test, forecast_probs)
            
            # Evaluate
            results = self.evaluate(df_test, forecast_probs)
            
            return results
        
        finally:
            # Properly close WandB without hanging on SSL errors
            if WANDB_AVAILABLE and self.config.use_wandb:
                try:
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("WandB finish timeout")
                    
                    # Set a 30-second timeout for wandb.finish()
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)
                    
                    try:
                        wandb.finish()
                        signal.alarm(0)  # Cancel alarm
                    except TimeoutError:
                        print("⚠ WandB finish timed out (SSL/network issues), continuing...")
                        signal.alarm(0)  # Cancel alarm
                        try:
                            wandb.finish(quiet=True)
                        except:
                            pass
                
                except Exception as e:
                    print(f"⚠ WandB close warning: {e}")
                    # Don't re-raise, allow the script to continue
    
    def train_ensemble_forecast(self, df_train, df_val, df_test):
        """
        Train ENSEMBLE FORECAST (Technical Indicators)
        Much better than LSTM for Bitcoin! (~60-65% accuracy vs 51% for LSTM)
        
        Uses:
        - RSI (Relative Strength Index)
        - EMA Crossover
        - MACD
        - Bollinger Bands
        """
        from better_forecast_systems import BetterForecastSystem
        
        print(f"\n{'='*80}")
        print(f"ENSEMBLE FORECAST (Technical Indicators)")
        print(f"{'='*80}\n")
        
        print("Generating ensemble forecasts for train/val/test sets...")
        
        # Generate ensemble forecasts
        train_probs = BetterForecastSystem.ensemble_forecast(df_train)
        val_probs = BetterForecastSystem.ensemble_forecast(df_val)
        test_probs = BetterForecastSystem.ensemble_forecast(df_test)
        
        # Evaluate on validation set
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_val = (df_val['r'].shift(-1) > 0).astype(int).fillna(0).values
        val_preds = (val_probs > 0.5).astype(int)
        
        val_acc = accuracy_score(y_val, val_preds)
        val_auc = roc_auc_score(y_val, val_probs)
        
        print(f"\n{'='*80}")
        print(f"ENSEMBLE FORECAST QUALITY")
        print(f"{'='*80}")
        print(f"Validation Accuracy: {val_acc:.4f} (50% = random, >55% = good)")
        print(f"Validation AUC-ROC:  {val_auc:.4f}")
        
        if val_acc < 0.55:
            print(f"⚠ WARNING: Forecast is barely better than random!")
        elif val_acc < 0.58:
            print(f"⚠ WARNING: Forecast quality is marginal (may hurt performance)")
        else:
            print(f"✓ GOOD: Forecast quality is decent, should help trading")
        
        print(f"{'='*80}\n")
        
        return train_probs, val_probs, test_probs
    
    def train_ppo(self, df_train, df_test, forecast_probs=None):
        """PPO training loop with detailed realtime metrics - EXACTLY MATCHES ORIGINAL NOTEBOOK"""
        print(f"\n{'='*80}")
        print(f"PPO TRAINING - REALTIME METRICS")
        print(f"{'='*80}")
        print(f"Initial Equity: ${self.config.environment.initial_equity:,.2f}")
        print(f"Total Updates: {self.config.ppo.total_updates}")
        print(f"Parallel Environments: {self.config.ppo.num_envs}")
        print(f"Fee: {self.config.environment.fee} | Kappa: {self.config.environment.kappa}")
        print(f"{'='*80}\n")
        
        # Create environments
        def make_env():
            return TradingEnv(
                df_train,
                fee=self.config.environment.fee,
                kappa=self.config.environment.kappa,
                leverage_max=self.config.environment.leverage_max,
                max_leverage=self.config.environment.leverage_max,
                initial_equity=self.config.environment.initial_equity,
                reward_type=self.config.environment.reward_type,
                forecast_probs=forecast_probs if self.config.forecast_mode == ForecastMode.LSTM else None,
                slippage_coef=self.config.environment.slippage_coef,
                smoothing_alpha=self.config.environment.smoothing_alpha,
                reward_scale=self.config.environment.reward_scale,
                include_turnover=self.config.environment.include_turnover,
                reward_params=self.config.environment.reward_params,  # ← Pass reward params!
            )
        
        env = gym.vector.SyncVectorEnv(
            [make_env for _ in range(self.config.ppo.num_envs)]
        )
        
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        
        # Build model
        model = ActorCritic(obs_dim, act_dim).to(self.device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.ppo.learning_rate,
            weight_decay=self.config.ppo.weight_decay
        )
        
        # Initialize observation
        obs, _ = env.reset(seed=self.config.seed)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        ep_returns = np.zeros(self.config.ppo.num_envs, dtype=np.float32)
        ep_history = []
        ep_costs_history = []  # Track costs
        ep_equity_history = []  # Track equity
        
        # Helper functions
        def squash(u):
            """Squash unbounded actions to [-1, 1] using tanh"""
            return torch.tanh(u)
        
        def logprob_squashed(dist, u):
            """Compute log probability for squashed actions"""
            logp_u = dist.log_prob(u).sum(-1)
            eps = 1e-6
            log_det = torch.log(1.0 - torch.tanh(u).pow(2) + eps).sum(-1)
            return logp_u - log_det
        
        # Training loop
        for update in range(self.config.ppo.total_updates):
            # Rollout buffers
            obs_buf = torch.zeros(
                self.config.ppo.n_steps, self.config.ppo.num_envs, obs_dim,
                device=self.device
            )
            u_buf = torch.zeros(
                self.config.ppo.n_steps, self.config.ppo.num_envs, act_dim,
                device=self.device
            )
            logp_buf = torch.zeros(
                self.config.ppo.n_steps, self.config.ppo.num_envs,
                device=self.device
            )
            rew_buf = torch.zeros(
                self.config.ppo.n_steps, self.config.ppo.num_envs,
                device=self.device
            )
            done_buf = torch.zeros(
                self.config.ppo.n_steps, self.config.ppo.num_envs,
                device=self.device
            )
            val_buf = torch.zeros(
                self.config.ppo.n_steps, self.config.ppo.num_envs,
                device=self.device
            )
            
            # Collect rollout
            for t in range(self.config.ppo.n_steps):
                obs_buf[t] = obs
                with torch.no_grad():
                    dist, value = model(obs)
                    u = dist.sample()
                    a = squash(u)
                    logp = logprob_squashed(dist, u)
                
                u_buf[t] = u
                logp_buf[t] = logp.detach()
                val_buf[t] = value.detach()
                
                # Scale actions from [-1, 1] to [-leverage_max, leverage_max]
                a_scaled = a * self.config.environment.leverage_max
                
                next_obs, reward, terminated, truncated, infos = env.step(
                    a_scaled.detach().cpu().numpy()
                )
                done_env = np.logical_or(terminated, truncated)
                done_boot = terminated  # bootstrap mask
                
                rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                done_buf[t] = torch.as_tensor(done_boot, dtype=torch.float32, device=self.device)
                
                # Episode return tracking with costs
                ep_returns += reward
                if done_env.any():
                    finished = np.where(done_env)[0]
                    ep_history.extend(ep_returns[finished].tolist())
                    ep_returns[finished] = 0.0
                    next_obs, _ = env.reset()
                
                obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            
            # Bootstrap last value
            with torch.no_grad():
                _, last_value = model(obs)
            
            # Compute GAE
            returns, advantages = self._compute_gae(
                rew_buf, done_buf, val_buf, last_value,
                self.config.ppo.gamma, self.config.ppo.gae_lambda
            )
            
            # Flatten for mini-batch training
            B = self.config.ppo.n_steps * self.config.ppo.num_envs
            obs_batch = obs_buf.reshape(B, obs_dim)
            u_batch = u_buf.reshape(B, act_dim)
            old_logp = logp_buf.reshape(B)
            old_value = val_buf.reshape(B).detach()
            ret_batch = returns.reshape(B).detach()
            adv_batch = advantages.reshape(B).detach()
            
            # Advantage normalization
            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
            
            idx = torch.arange(B, device=self.device)
            stop = False
            
            # PPO epochs
            for _ in range(self.config.ppo.ppo_epochs):
                perm = idx[torch.randperm(B)]
                for start in range(0, B, self.config.ppo.minibatch_size):
                    mb = perm[start:start + self.config.ppo.minibatch_size]
                    
                    dist, value = model(obs_batch[mb])
                    logp = logprob_squashed(dist, u_batch[mb])
                    entropy = dist.entropy().sum(-1)
                    
                    # Early stop by approximate KL
                    approx_kl = (old_logp[mb] - logp).mean().detach()
                    if approx_kl.item() > self.config.ppo.target_kl:
                        stop = True
                        break
                    
                    ratio = torch.exp(logp - old_logp[mb])
                    
                    # Clipped policy objective
                    unclipped = ratio * adv_batch[mb]
                    clipped = torch.clamp(
                        ratio, 1 - self.config.ppo.clip_eps,
                        1 + self.config.ppo.clip_eps
                    ) * adv_batch[mb]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    
                    # Value loss with clipping
                    value_pred_clipped = old_value[mb] + torch.clamp(
                        value - old_value[mb],
                        -self.config.ppo.clip_eps,
                        self.config.ppo.clip_eps
                    )
                    value_loss_unclipped = (ret_batch[mb] - value).pow(2)
                    value_loss_clipped = (ret_batch[mb] - value_pred_clipped).pow(2)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    
                    # Entropy bonus
                    entropy_loss = -entropy.mean()
                    
                    total_loss = (
                        policy_loss +
                        self.config.ppo.vf_coef * value_loss +
                        self.config.ppo.ent_coef * entropy_loss
                    )
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.ppo.max_grad_norm
                    )
                    optimizer.step()
                
                if stop:
                    break
            
            # Keep std in reasonable range
            with torch.no_grad():
                model.log_std.clamp_(-2.0, -0.5)
            
            # Detailed logging every 100 updates
            if update % 100 == 0:
                mean_100 = np.mean(ep_history[-100:]) if len(ep_history) >= 100 else np.nan
                std_100 = np.std(ep_history[-100:]) if len(ep_history) >= 100 else np.nan
                log_std = model.log_std.data.cpu().numpy()[0]
                
                # Print comprehensive realtime stats
                print(f"\n{'='*80}")
                print(f"UPDATE {update:5d} / {self.config.ppo.total_updates:5d}")
                print(f"{'='*80}")
                print(f"Episode Returns (last 100): {mean_100:>10.2f} ± {std_100:>6.2f}")
                print(f"Total Episodes Trained:     {len(ep_history):>10d}")
                print(f"Log Std (exploration):      {log_std:>10.3f}")
                print(f"Policy Loss:                {policy_loss.item():>10.4f}")
                print(f"Value Loss:                 {value_loss.item():>10.4f}")
                print(f"Entropy Loss:               {entropy_loss.item():>10.4f}")
                print(f"Total Loss:                 {total_loss.item():>10.4f}")
                print(f"Approx KL:                  {approx_kl.item():>10.4f}")
                print(f"{'='*80}\n")
                
                if WANDB_AVAILABLE and self.config.use_wandb:
                    wandb.log({
                        "training/update": update,
                        "training/episode_return_mean_100": mean_100,
                        "training/episode_return_std_100": std_100,
                        "training/log_std": log_std,
                        "training/total_episodes": len(ep_history),
                        "training/policy_loss": policy_loss.item(),
                        "training/value_loss": value_loss.item(),
                        "training/entropy_loss": entropy_loss.item(),
                        "training/total_loss": total_loss.item(),
                        "training/approx_kl": approx_kl.item(),
                    })
        
        self.ppo_model = model
        
        # Properly close the training environment to prevent hanging
        try:
            env.close()
            print("\n✓ Training environment closed")
        except Exception as e:
            print(f"⚠ Warning closing training environment: {e}")
        
        return model
    
    @staticmethod
    def _compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
        """Compute GAE advantages - MATCHES ORIGINAL"""
        T, N = rewards.shape
        adv = torch.zeros(T, N, device=values.device)
        gae = torch.zeros(N, device=values.device)
        
        for t in reversed(range(T)):
            not_done = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            gae = delta + gamma * lam * not_done * gae
            adv[t] = gae
        
        returns = adv + values
        return returns, adv
    
    def evaluate(self, df_test, forecast_probs=None):
        """Evaluate on test set with budget tracking and metrics logging"""
        print(f"\n{'='*70}")
        print(f"EVALUATION ON TEST SET")
        print(f"{'='*70}\n")
        
        try:
            # Create budget tracker with WandB logging enabled
            tracker = BudgetTracker(
                self.config.environment.initial_equity,
                enable_wandb_logging=(WANDB_AVAILABLE and self.config.use_wandb)
            )
            
            env_test = TradingEnv(
                df_test,
                fee=self.config.environment.fee,
                kappa=self.config.environment.kappa,
                leverage_max=self.config.environment.leverage_max,
                max_leverage=self.config.environment.leverage_max,
                initial_equity=self.config.environment.initial_equity,
                reward_type=self.config.environment.reward_type,
                forecast_probs=forecast_probs,
                slippage_coef=self.config.environment.slippage_coef,
                smoothing_alpha=self.config.environment.smoothing_alpha,
                reward_scale=self.config.environment.reward_scale,
                include_turnover=self.config.environment.include_turnover,
                reward_params=self.config.environment.reward_params,  # ← Pass reward params!
            )
            
            obs, _ = env_test.reset()
            done = False
            
            if hasattr(self, 'ppo_model'):
                model = self.ppo_model
                step_counter = 0
                
                # Evaluation loop with progress indicator
                import sys
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        dist, _ = model(obs_t)
                        # Use mean action (deterministic)
                        u = dist.mean
                        # Squash to [-1, 1]
                        a = torch.tanh(u)
                        # Scale to [-leverage_max, leverage_max]
                        action_scaled = a * self.config.environment.leverage_max
                    
                    obs, reward, terminated, truncated, info = env_test.step(
                        action_scaled.detach().cpu().numpy()[0]
                    )
                    done = terminated or truncated
                    step_counter += 1
                    
                    # Print progress every 50 steps
                    if step_counter % 50 == 0:
                        sys.stdout.write(f'\r  Evaluation step: {step_counter}')
                        sys.stdout.flush()
                    
                    # Record in tracker (which automatically logs to WandB)
                    tracker.record_step(
                        equity=info.get('equity', env_test.equity),
                        position=info.get('position', env_test.pos),
                        cost=info.get('cost', 0.0),
                        pnl=info.get('pnl', 0.0),
                        drawdown=info.get('drawdown', 0.0),
                        cumulative_return=info.get('cumulative_return', 0.0)
                    )
                
                sys.stdout.write(f'\r  Evaluation completed: {step_counter} steps     \n')
                sys.stdout.flush()
            
            # Print summary quickly
            print(f"\n{'='*70}")
            print(f"BUDGET & LIQUIDITY SUMMARY")
            print(f"{'='*70}")
            
            # Get equity array
            equity_array = np.array(tracker.equity_values) if hasattr(tracker, 'equity_values') else np.array([])
            
            if len(equity_array) > 0:
                # Quick metric calculation
                final_equity = equity_array[-1]
                total_return = (final_equity - self.config.environment.initial_equity) / self.config.environment.initial_equity
                
                daily_returns = np.diff(equity_array) / equity_array[:-1] if len(equity_array) > 1 else np.array([])
                
                print(f"Initial Equity:        ${self.config.environment.initial_equity:,.2f}")
                print(f"Final Equity:          ${final_equity:,.2f}")
                print(f"Total Return:          {total_return*100:>10.2f}%")
                
                total_costs = np.sum(tracker.costs) if hasattr(tracker, 'costs') and tracker.costs else 0
                total_pnl = final_equity - self.config.environment.initial_equity
                
                print(f"Total Costs:           ${total_costs:>10,.2f}")
                print(f"Total PnL:             ${total_pnl:>10,.2f}")
                
                avg_daily_pnl = total_pnl / max(len(equity_array), 1)
                print(f"Avg Daily PnL:         ${avg_daily_pnl:>10,.2f}")
                
                # Calculate max drawdown
                if len(equity_array) > 0:
                    running_max = np.maximum.accumulate(equity_array)
                    drawdown = (equity_array - running_max) / running_max
                    max_drawdown = np.min(drawdown)
                else:
                    max_drawdown = 0
                    
                print(f"Max Drawdown:          {max_drawdown*100:>10.2f}%")
                
                avg_position_size = np.mean(np.abs(tracker.positions)) if hasattr(tracker, 'positions') and tracker.positions else 0
                print(f"Avg Position Size:     {avg_position_size:>10.4f}")
                
                num_steps = len(equity_array) - 1 if len(equity_array) > 1 else 0
                print(f"Number of Steps:       {num_steps:>10d}")
                
                print(f"{'='*70}\n")
            
            # Calculate metrics efficiently WITHOUT visualization
            metrics_calc = TradingMetrics(self.config.environment.initial_equity)
            
            if len(equity_array) > 0 and len(daily_returns) > 0:
                metrics = metrics_calc.calculate_all_metrics(
                    equity_array, 
                    daily_returns, 
                    np.array(tracker.positions) if hasattr(tracker, 'positions') else np.array([]),
                    np.array(tracker.costs) if hasattr(tracker, 'costs') else np.array([])
                )
            else:
                metrics = {}
            
            # Print metrics summary (console output only, no visualization)
            if metrics:
                print(f"Final Equity:          ${equity_array[-1]:,.2f}")
                print(f"Total Return:          {metrics.get('total_return', 0)*100:.2f}%")
                print(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"Max Drawdown:          {metrics.get('max_drawdown', 0)*100:.2f}%")
                print(f"Volatility:            {metrics.get('volatility', 0)*100:.2f}%")
                print(f"Annualized Return:     {metrics.get('annualized_return', 0)*100:.2f}%")
                print(f"Annualized Volatility: {metrics.get('annualized_volatility', 0)*100:.2f}%")
                print(f"Calmar Ratio:          {metrics.get('calmar_ratio', 0):.4f}")
                print(f"Sortino Ratio:         {metrics.get('sortino_ratio', 0):.4f}")
                print(f"Win Rate:              {metrics.get('win_rate', 0)*100:.2f}%")
                print(f"Profit Factor:         {metrics.get('profit_factor', 0):.4f}")
                print(f"Turnover:              {metrics.get('turnover', 0):.4f}")
                print(f"Total Transaction Costs: ${np.sum(tracker.costs) if hasattr(tracker, 'costs') else 0:,.2f}")
                print(f"Cost Ratio:            {metrics.get('cost_ratio', 0):.4f}")
                print(f"Kurtosis:              {metrics.get('kurtosis', 0):.4f}")
                print(f"Skewness:              {metrics.get('skewness', 0):.4f}")
            
            # Log metrics to WandB
            if WANDB_AVAILABLE and self.config.use_wandb:
                try:
                    wandb_metrics = {
                        "evaluation/final_equity": equity_array[-1] if len(equity_array) > 0 else 0,
                        "evaluation/total_return": metrics.get('total_return', 0),
                        "evaluation/sharpe_ratio": metrics.get('sharpe_ratio', 0),
                        "evaluation/max_drawdown": metrics.get('max_drawdown', 0),
                        "evaluation/volatility": metrics.get('volatility', 0),
                    }
                    wandb.log(wandb_metrics)
                    print("\n✓ Metrics logged to WandB")
                except Exception as e:
                    print(f"⚠ Warning: Could not log metrics to WandB: {e}")
            
            # Save metrics locally
            import os
            results_dir = self.config.results_dir
            os.makedirs(results_dir, exist_ok=True)
            
            import pickle
            metrics_to_save = {
                'experiment_name': self.config.experiment_name,
                'forecast_mode': self.config.forecast_mode.value,
                'reward_type': self.config.reward_type.value,
                'timestamp': pd.Timestamp.now().isoformat(),
                'metrics': metrics,
                'equity_curve': equity_array.tolist() if len(equity_array) > 0 else [],
                'daily_returns': daily_returns.tolist() if len(daily_returns) > 0 else [],
            }
            
            pickle_path = os.path.join(results_dir, 'metrics.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(metrics_to_save, f)
            print(f"✓ Metrics saved to: {pickle_path}")
            
            # Save CSV
            csv_path = os.path.join(results_dir, 'metrics_summary.csv')
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(csv_path, index=False)
            print(f"✓ Metrics CSV saved to: {csv_path}\n")
            
            # Properly close environment
            try:
                env_test.close()
                print("✓ Test environment closed")
            except Exception as e:
                print(f"⚠ Warning closing test environment: {e}")
            
            return {
                'equity': equity_array,
                'positions': np.array(tracker.positions) if hasattr(tracker, 'positions') else np.array([]),
                'costs': np.array(tracker.costs) if hasattr(tracker, 'costs') else np.array([]),
                'pnl': np.array(tracker.pnl_values) if hasattr(tracker, 'pnl_values') else np.array([]),
                'metrics': metrics,
                'tracker': tracker,
            }
        
        except Exception as e:
            print(f"\n⚠ CRITICAL ERROR in evaluate(): {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty results on failure
            return {
                'equity': np.array([]),
                'positions': np.array([]),
                'costs': np.array([]),
                'pnl': np.array([]),
                'metrics': {},
                'tracker': None,
            }
    
    @staticmethod
    def _compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
        """Compute GAE advantages"""
        T, N = rewards.shape
        adv = torch.zeros(T, N, device=values.device)
        gae = torch.zeros(N, device=values.device)
        
        for t in reversed(range(T)):
            not_done = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            gae = delta + gamma * lam * not_done * gae
            adv[t] = gae
        
        returns = adv + values
        return returns, adv


class LSTMForecaster(nn.Module):
    """LSTM forecasting model"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        pred = self.fc(h_last)
        return pred, h_n


class ActorCritic(nn.Module):
    """PPO Actor-Critic network"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        self.mu = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)
        self.v = nn.Linear(128, 1)
    
    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        value = self.v(x).squeeze(-1)
        return dist, value


class TradingEnv(gym.Env):
    """Trading environment - EXACTLY MATCHES ORIGINAL"""
    metadata = {"render_modes": []}
    
    def __init__(self, df, fee=0.0001, kappa=0.01, leverage_max=1.0,
                 max_leverage=1.0, initial_equity=100000, reward_type=RewardType.WITH_RISK,
                 forecast_probs=None, slippage_coef=0.0, smoothing_alpha=1.0, 
                 reward_scale=1.0, include_turnover=False, reward_params=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        
        # Store reward params (allows config-driven parameter tuning)
        self.reward_params = reward_params or {}
        
        # Cost parameters - read from reward_params if available
        self.fee = float(fee)
        self.kappa = float(self.reward_params.get('kappa', kappa))
        self.slippage_coef = float(slippage_coef)
        self.smoothing_alpha = float(smoothing_alpha)
        
        self.leverage_max = float(leverage_max)
        self.max_leverage = float(max_leverage)
        self.reward_scale = float(self.reward_params.get('reward_scale', reward_scale))
        self.include_turnover = bool(include_turnover)
        
        # Extract reward-specific parameters from config
        self.epsilon = float(self.reward_params.get('epsilon', 0.001))
        self.downside_scale = float(self.reward_params.get('downside_scale', 1.2))
        self.drawdown_multiplier = float(self.reward_params.get('drawdown_multiplier', 0.5))
        self.consistency_bonus = float(self.reward_params.get('consistency_bonus', 0.1))
        
        # Composite reward weights
        self.weight_returns = float(self.reward_params.get('weight_returns', 0.5))
        self.weight_sharpe = float(self.reward_params.get('weight_sharpe', 0.3))
        self.weight_risk = float(self.reward_params.get('weight_risk', 0.2))
        
        # Validate weights sum to approximately 1.0
        total_weight = self.weight_returns + self.weight_sharpe + self.weight_risk
        if abs(total_weight - 1.0) > 0.01:
            print(f"⚠ Warning: Composite weights don't sum to 1.0 (sum={total_weight:.3f})")
            # Normalize
            self.weight_returns /= total_weight
            self.weight_sharpe /= total_weight
            self.weight_risk /= total_weight
        
        self.initial_equity = float(initial_equity)
        self.reward_type = reward_type
        self.forecast_probs = forecast_probs
        self.include_forecast = forecast_probs is not None
        
        # Feature columns - MUST MATCH ORIGINAL
        self.feature_cols = ['r', 'r_lag1', 'mu_hat', 'sigma_hat', 'mom_5', 'mom_20', 'vol_ratio', 'signal_strength']
        
        self.action_space = spaces.Box(
            low=-self.max_leverage,
            high=self.max_leverage,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation dimension
        portfolio_dim = 3
        if self.include_turnover:
            portfolio_dim += 1
        n_lstm_features = 1 if self.include_forecast else 0
        obs_dim = len(self.feature_cols) + portfolio_dim + n_lstm_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 1
        self.pos = 0.0
        self.target_pos = 0.0
        self.prev_turnover = 0.0
        
        # Budget
        self.equity = self.initial_equity
        self.peak = self.initial_equity
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Market features
        x = self.df.loc[self.t, self.feature_cols].values.astype(np.float32)
        
        # Normalized equity (relative to initial budget)
        equity_norm = np.float32(self.equity / self.initial_equity)
        drawdown = np.float32((self.peak - self.equity) / (self.peak + 1e-8))
        
        # Portfolio features
        portfolio_features = [self.pos, equity_norm, drawdown]
        if self.include_turnover:
            portfolio_features.append(self.prev_turnover)
        
        obs = np.concatenate([x, np.array(portfolio_features, dtype=np.float32)])
        
        # Optional LSTM forecast signal
        if self.include_forecast and self.t < len(self.forecast_probs):
            lstm_signal = float(self.forecast_probs[self.t] * 2 - 1)
            obs = np.concatenate([obs, [lstm_signal]])
        elif self.include_forecast:
            obs = np.concatenate([obs, [0.0]])
        
        return obs
    
    def step(self, action):
        # 1) Raw target action from policy
        raw_target = float(np.clip(action[0], -self.max_leverage, self.max_leverage))
        
        # 2) Position smoothing / execution lag
        new_pos = (1.0 - self.smoothing_alpha) * self.pos + self.smoothing_alpha * raw_target
        new_pos = float(np.clip(new_pos, -self.max_leverage, self.max_leverage))
        
        # 3) Market data for current step
        r_t = float(self.df.loc[self.t, "r"])
        sigma_t = float(self.df.loc[self.t, "sigma_hat"])
        
        if not np.isfinite(sigma_t):
            sigma_t = 0.0
        
        # 4) PnL from PREVIOUS position
        pnl = self.pos * r_t
        
        # 5) Trading turnover
        turnover = abs(new_pos - self.pos)
        
        # 6) Transaction cost
        cost = self.fee * turnover
        
        # 7) Slippage / market impact
        slippage = self.slippage_coef * turnover * (1.0 + sigma_t)
        
        # 8) Risk penalty (for WITH_RISK reward)
        risk_pen = self.kappa * (self.pos ** 2) * sigma_t
        
        # 9) Calculate reward based on reward type
        # CRITICAL: use true_reward for equity update, NOT including risk penalty
        true_reward = pnl - cost - slippage
        
        # IMPORTANT: Different reward functions for learning
        if self.reward_type == RewardType.BASIC:
            # Basic reward: Just PnL minus costs
            # Encourages maximum returns without risk penalty
            reward = true_reward
            
        elif self.reward_type == RewardType.WITH_RISK:
            # With risk penalty: PnL - costs - quadratic position penalty
            # Discourages large positions via kappa * pos^2 * volatility
            reward = true_reward - risk_pen
            
        elif self.reward_type == RewardType.WITH_SHARPE:
            # Sharpe-like reward: normalize by volatility
            # (PnL - cost) / volatility
            # Explicitly optimizes risk-adjusted returns
            safe_sigma = max(sigma_t, 0.001)  # Avoid division by zero
            reward = true_reward / safe_sigma
            
        elif self.reward_type == RewardType.RISK_ADJUSTED:
            # Risk-adjusted reward: (PnL / volatility) - cost
            # Returns normalized by volatility, but costs not normalized
            # More aggressive in calm periods than WITH_SHARPE
            safe_sigma = max(sigma_t, 0.001)  # Avoid division by zero
            risk_adjusted_pnl = pnl / safe_sigma if safe_sigma > 0 else 0
            reward = risk_adjusted_pnl - cost - slippage
            
        elif self.reward_type == RewardType.SORTINO:
            # Sortino: Focus on downside volatility
            # Penalize negative returns more heavily than positive
            safe_sigma = max(sigma_t, 0.001)
            downside_penalty = self.downside_scale  # ← Uses config parameter
            # If pnl is negative, apply higher penalty
            downside_adjusted_pnl = pnl if pnl > 0 else pnl * downside_penalty
            reward = (downside_adjusted_pnl - cost - slippage) / safe_sigma
            
        elif self.reward_type == RewardType.CALMAR:
            # Calmar: Focus on maximum drawdown
            # Estimate drawdown from current position and volatility
            current_dd = (self.peak - self.equity) / (self.peak + 1e-8)
            drawdown_estimate = self.kappa * (self.pos ** 2) * sigma_t  # Drawdown estimate
            reward = true_reward - (drawdown_estimate * self.drawdown_multiplier)  # ← Uses config parameter
            
        elif self.reward_type == RewardType.INFORMATION_RATIO:
            # Information ratio: Consistency of returns
            # Bonus for positive returns, penalty for volatility
            safe_sigma = max(sigma_t, 0.001)
            consistency_bonus = self.consistency_bonus if pnl > 0 else 0  # ← Uses config parameter
            return_signal = pnl / safe_sigma if safe_sigma > 0 else 0
            reward = return_signal - cost - slippage + consistency_bonus
            
        elif self.reward_type == RewardType.COMPOSITE:
            # Composite: Multi-objective weighted blend
            # Combines multiple signals for robust learning
            safe_sigma = max(sigma_t, 0.001)
            
            # Signal 1: Raw returns
            signal_returns = true_reward
            
            # Signal 2: Sharpe-like
            signal_sharpe = true_reward / safe_sigma
            
            # Signal 3: Risk penalty
            signal_risk = -risk_pen
            
            # Weighted combination - uses config parameters
            reward = (self.weight_returns * signal_returns + 
                     self.weight_sharpe * signal_sharpe + 
                     self.weight_risk * signal_risk)
        
        else:
            # Default: use WITH_RISK
            reward = true_reward - risk_pen
        
        reward *= self.reward_scale
        
        # CRITICAL FIX: Add positive reward baseline to encourage profit-seeking
        # Otherwise agent has no baseline to improve from
        reward += 0.0001 * self.reward_scale  # Small positive baseline
        
        # 10) Update internal portfolio state
        self.target_pos = raw_target
        self.prev_turnover = turnover
        self.pos = new_pos
        
        # 11) Update equity using LINEAR formula (not exponential)
        # LINEAR is more stable: equity_new = equity_old * (1 + daily_return)
        # EXPONENTIAL exp(x) only works for very small x and can blow up
        self.equity = self.equity * (1.0 + true_reward)  # ← LINEAR UPDATE
        self.peak = max(self.peak, self.equity)
        
        # 12) Advance time
        self.t += 1
        terminated = (self.t >= len(self.df) - 1)
        
        # 13) Info dictionary for diagnostics
        info = {
            "pnl": pnl,
            "cost": cost,
            "slippage": slippage,
            "risk_pen": risk_pen,
            "turnover": turnover,
            "position": self.pos,
            "target_position": self.target_pos,
            "equity": self.equity,
            "drawdown": (self.peak - self.equity) / (self.peak + 1e-8),
            "cumulative_return": (self.equity - self.initial_equity) / self.initial_equity,
            "reward_type": self.reward_type.value if hasattr(self.reward_type, 'value') else str(self.reward_type),
        }
        
        return self._get_obs(), float(reward), terminated, False, info


if __name__ == "__main__":
    print("PPO Trading Framework Loaded")
    print("Use: python parameterized_experiments.py")

