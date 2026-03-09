"""
Enhanced Trading Environment with:
- Long/Short mechanics with leverage
- Clean PnL tracking (realized + unrealized)
- Budget/Liquidity constraints
- Advanced state representation
- Risk metrics
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class EnhancedTradingEnv(gym.Env):
    """
    Advanced trading environment for RL agents.

    Action space: Continuous [-1, 1] representing target position fraction
        - -1: fully short with leverage
        -  0: neutral (cash)
        -  1: fully long with leverage

    State includes:
        - Market features (returns, volatility, forecast signal)
        - Position state (current position, exposure)
        - Portfolio metrics (equity, drawdown, cash available)
        - Risk metrics (sharpe, volatility)
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        fee=0.0005,
        kappa=0.1,
        initial_cash=100000.0,
        max_leverage=2.0,
        risk_free_rate=0.04,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with features (r, sigma_hat, mu_hat, etc.)
        fee : float
            Transaction cost per unit position change (fraction)
        kappa : float
            Risk penalty weight
        initial_cash : float
            Starting capital
        max_leverage : float
            Maximum allowed leverage (1.0 = no margin, 2.0 = 2x margin)
        risk_free_rate : float
            Annual risk-free rate for Sharpe ratio
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.fee = float(fee)
        self.kappa = float(kappa)
        self.initial_cash = float(initial_cash)
        self.max_leverage = float(max_leverage)
        self.daily_risk_free_rate = risk_free_rate / 252.0

        # Action space: target position in [-max_leverage, max_leverage]
        self.action_space = spaces.Box(
            low=-self.max_leverage,
            high=self.max_leverage,
            shape=(1,),
            dtype=np.float32
        )

        # Feature columns
        self.feature_cols = ["r", "r_lag1", "mu_hat", "sigma_hat"]

        # Check if forecast features exist
        self.has_forecast = "forecast" in self.df.columns
        if self.has_forecast:
            self.feature_cols.append("forecast")

        # Observation space dimension
        # Market features: len(feature_cols)
        # Position state: position (1), leverage_used (1), long_exposure (1), short_exposure (1)
        # Portfolio: equity (1), cash_ratio (1), drawdown (1), sharpe_estimate (1)
        # Risk: volatility (1), max_position_change (1)
        obs_dim = len(self.feature_cols) + 10
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 1  # Start from 1 to use lagged features
        self.cash = float(self.initial_cash)
        self.position = 0.0  # Position size (can be long or short)
        self.equity = float(self.initial_cash)
        self.peak_equity = float(self.initial_cash)

        # PnL tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.cumulative_fees = 0.0

        # Return history for Sharpe calculation
        self.daily_returns = []

        return self._get_obs(), {}

    def _get_obs(self):
        """Construct observation vector with market and portfolio state."""
        # Market features
        market_features = self.df.loc[self.t, self.feature_cols].values.astype(np.float32)

        # Position state
        current_price = float(self.df.loc[self.t, "close"])
        long_exposure = max(0, self.position)
        short_exposure = abs(min(0, self.position))
        leverage_used = (long_exposure + short_exposure) / (self.cash / (current_price + 1e-8) + 1e-8)
        leverage_used = np.clip(leverage_used, 0, self.max_leverage)

        # Portfolio metrics
        equity_norm = np.float32(self.equity / self.initial_cash)
        drawdown = np.float32((self.peak_equity - self.equity) / (self.peak_equity + 1e-8))
        cash_ratio = np.float32(self.cash / self.equity if self.equity > 0 else 0)

        # Risk metrics
        if len(self.daily_returns) >= 20:
            volatility = np.float32(np.std(self.daily_returns[-20:]))
            # Rolling Sharpe (20-day)
            ret_mean = np.mean(self.daily_returns[-20:])
            sharpe = np.float32((ret_mean - self.daily_risk_free_rate) / (volatility + 1e-8))
        else:
            volatility = np.float32(0.0)
            sharpe = np.float32(0.0)

        # Max position change (for constraint awareness)
        max_pos_change = np.float32(self.max_leverage)

        # Assemble observation
        position_state = np.array(
            [
                np.float32(np.clip(self.position, -self.max_leverage, self.max_leverage)),
                np.float32(leverage_used),
                np.float32(long_exposure),
                np.float32(short_exposure),
            ],
            dtype=np.float32
        )

        portfolio_state = np.array(
            [
                equity_norm,
                drawdown,
                cash_ratio,
                sharpe,
                volatility,
                max_pos_change,
            ],
            dtype=np.float32
        )

        obs = np.concatenate([market_features, position_state, portfolio_state])
        return obs

    def step(self, action):
        """
        Execute one step of the environment.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        # Clip action to valid range
        target_position = float(np.clip(action[0], -self.max_leverage, self.max_leverage))

        # Get market data
        current_price = float(self.df.loc[self.t, "close"])
        r_t = float(self.df.loc[self.t, "r"])
        sigma_t = float(self.df.loc[self.t, "sigma_hat"])
        if not np.isfinite(sigma_t):
            sigma_t = 0.0

        # === PnL Calculation ===
        # Unrealized PnL from existing position
        unrealized_pnl = self.position * current_price * r_t

        # Transaction cost from position change
        position_delta = abs(target_position - self.position)
        transaction_cost = self.fee * position_delta * current_price

        # Risk penalty (volatility-adjusted)
        risk_penalty = self.kappa * (target_position ** 2) * (sigma_t + 1e-8)

        # Calculate immediate reward
        reward = unrealized_pnl - transaction_cost - risk_penalty

        # === Update Portfolio State ===
        # Update cash (position change cost)
        cash_delta = -(target_position - self.position) * current_price
        self.cash += cash_delta

        # Update position
        self.position = target_position

        # Check budget constraint
        budget_ok = self.cash >= 0

        # Update tracking variables
        self.unrealized_pnl = self.position * current_price * np.exp(r_t)
        self.cumulative_fees += transaction_cost

        # Update equity and peak
        daily_return = reward / (self.equity + 1e-8)
        self.equity += reward
        self.equity = max(0.0, self.equity)  # Can't go negative
        self.peak_equity = max(self.peak_equity, self.equity)

        # Track daily returns for Sharpe
        self.daily_returns.append(daily_return)

        # Penalty for violating budget
        if not budget_ok:
            reward -= 1000.0  # Large penalty

        # Step forward in time
        self.t += 1
        terminated = self.t >= len(self.df) - 1
        truncated = False

        # Info dict
        info = {
            "equity": float(self.equity),
            "position": float(self.position),
            "cash": float(self.cash),
            "realized_pnl": float(self.realized_pnl),
            "cumulative_fees": float(self.cumulative_fees),
            "budget_ok": bool(budget_ok),
        }

        # If we've reached the end, return a terminal observation (can't access self.t anymore)
        if terminated:
            # Create terminal observation from last available state
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return obs, float(reward), terminated, truncated, info

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode="human"):
        """Optional rendering."""
        pass


class TradingEnvWithForecast(EnhancedTradingEnv):
    """
    Trading environment that includes forecast signals in the state.
    Inherits all functionality from EnhancedTradingEnv.
    """
    def __init__(self, df, forecast_column="forecast", **kwargs):
        self.forecast_column = forecast_column
        super().__init__(df, **kwargs)


class TradingEnvWithoutForecast(EnhancedTradingEnv):
    """
    Trading environment without forecast signals (baseline).
    Useful for comparing impact of forecasts.
    """
    def _get_obs(self):
        """Construct observation without forecast column."""
        # Temporarily remove forecast from feature cols
        original_cols = self.feature_cols
        self.feature_cols = [c for c in self.feature_cols if c != "forecast"]

        obs = super()._get_obs()

        # Restore original
        self.feature_cols = original_cols
        return obs


if __name__ == "__main__":
    # Test the environment
    import yfinance as yf

    print("Testing EnhancedTradingEnv...")

    # Load sample data
    df = yf.download("BTC-USD", start="2023-01-01", end="2024-01-01", progress=False)
    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    # Add features
    df["log_close"] = np.log(df["close"])
    df["r"] = df["log_close"].diff()
    df["mu_hat"] = df["r"].ewm(span=20, adjust=False).mean()
    df["sigma_hat"] = df["r"].rolling(20).std()
    df["r_lag1"] = df["r"].shift(1)
    df = df.dropna()

    # Create environment
    env = EnhancedTradingEnv(df)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Run a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.4f}, Equity: {info['equity']:.2f}")
        if terminated:
            break

    print("✅ Environment test passed!")

