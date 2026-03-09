"""
Forecasting Module for Time-Series Prediction

Includes:
- LSTM-based price forecasting
- Volatility forecasting
- Signal generation for RL integration
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    """Dataset for time-series forecasting."""

    def __init__(self, data, lookback=20, forecast_horizon=5):
        """
        Parameters
        ----------
        data : np.ndarray
            Input features [N, features]
        lookback : int
            Number of past timesteps to use
        forecast_horizon : int
            Number of steps ahead to predict
        """
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback : idx + self.lookback + self.forecast_horizon, 0]
        return x, y


class LSTMForecaster(nn.Module):
    """LSTM model for time-series forecasting."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, forecast_horizon=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_size, seq_len, input_size]

        Returns
        -------
        torch.Tensor
            [batch_size, forecast_horizon]
        """
        _, (h_n, _) = self.lstm(x)
        # Use last hidden state
        out = self.fc(h_n[-1])  # [batch_size, forecast_horizon]
        return out


class TimeSeriesForecaster:
    """Wrapper for training and inference."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, forecast_horizon=5, device=None):
        self.device = device or torch.device("cpu")
        self.model = LSTMForecaster(input_size, hidden_size, num_layers, forecast_horizon).to(self.device)
        self.forecast_horizon = forecast_horizon

    def train(self, train_data, val_data=None, epochs=50, batch_size=32, lr=0.001):
        """
        Train the forecaster.

        Parameters
        ----------
        train_data : np.ndarray
            Training features [N, features]
        val_data : np.ndarray, optional
            Validation features
        epochs : int
        batch_size : int
        lr : float
            Learning rate

        Returns
        -------
        dict
            Training history
        """
        train_dataset = TimeSeriesDataset(train_data, lookback=20, forecast_horizon=self.forecast_horizon)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data is not None:
            val_dataset = TimeSeriesDataset(val_data, lookback=20, forecast_horizon=self.forecast_horizon)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        pred = self.model(x)
                        loss = criterion(pred, y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f}")

        return history

    def predict(self, data):
        """
        Make predictions on new data.

        Parameters
        ----------
        data : np.ndarray
            Features [N, features]

        Returns
        -------
        np.ndarray
            Forecasts [N, forecast_horizon]
        """
        self.model.eval()
        dataset = TimeSeriesDataset(data, lookback=20, forecast_horizon=self.forecast_horizon)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        predictions = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())

        return np.vstack(predictions) if predictions else np.array([])

    def save(self, path):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


class ForecastSignalGenerator:
    """Convert forecasts into actionable trading signals."""

    @staticmethod
    def compute_signal(forecast_mean, forecast_std=None, threshold=0.01):
        """
        Convert forecast to trading signal.

        Parameters
        ----------
        forecast_mean : np.ndarray
            Average forecast value
        forecast_std : np.ndarray, optional
            Std of forecast (for confidence weighting)
        threshold : float
            Threshold for signal generation

        Returns
        -------
        np.ndarray
            Signal in [-1, 1]
        """
        signal = np.tanh(forecast_mean / (threshold + 1e-8))

        # Weight by forecast confidence if available
        if forecast_std is not None:
            confidence = np.exp(-forecast_std)  # More confident = higher weight
            signal = signal * confidence

        return np.clip(signal, -1, 1)


def create_forecast_features(df, lookback=20, forecast_horizon=5, split_ratio=0.8):
    """
    Create forecast-augmented features for the trading environment.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with basic features
    lookback : int
    forecast_horizon : int
    split_ratio : float
        Train/test split

    Returns
    -------
    df_train, df_test, forecaster
    """
    # Prepare data for forecasting
    features = df[["r", "mu_hat", "sigma_hat"]].values

    # Split
    split_idx = int(len(features) * split_ratio)
    train_features = features[:split_idx]
    test_features = features[split_idx:]

    # Initialize and train forecaster
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster = TimeSeriesForecaster(
        input_size=features.shape[1],
        hidden_size=64,
        num_layers=2,
        forecast_horizon=forecast_horizon,
        device=device
    )

    print("Training forecast model...")
    history = forecaster.train(train_features, val_data=test_features, epochs=100, batch_size=32)

    # Generate forecasts
    all_forecasts = forecaster.predict(features)

    # Add forecast signal to dataframe
    df_with_forecast = df.copy()

    # Pad with zeros for the first few rows
    pad_len = lookback + forecast_horizon - 1
    forecasts_padded = np.vstack([
        np.zeros((pad_len, forecast_horizon)),
        all_forecasts
    ])

    # Take mean forecast as signal
    df_with_forecast["forecast"] = forecasts_padded[:len(df_with_forecast), 0]
    df_with_forecast["forecast"] = df_with_forecast["forecast"].fillna(0)

    return df_with_forecast, forecaster, history


if __name__ == "__main__":
    # Test the forecasting module
    import yfinance as yf

    print("Testing TimeSeriesForecaster...")

    # Load data
    df = yf.download("BTC-USD", start="2023-01-01", end="2024-01-01", progress=False)
    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    # Add basic features
    df["log_close"] = np.log(df["close"])
    df["r"] = df["log_close"].diff()
    df["mu_hat"] = df["r"].ewm(span=20, adjust=False).mean()
    df["sigma_hat"] = df["r"].rolling(20).std()
    df["r_lag1"] = df["r"].shift(1)
    df = df.dropna()

    # Create forecast features
    df_with_forecast, forecaster, history = create_forecast_features(df, forecast_horizon=5)

    print("✅ Forecasting test passed!")
    print(f"Forecast column added: {'forecast' in df_with_forecast.columns}")
    print(f"Forecast values: {df_with_forecast['forecast'].head()}")

