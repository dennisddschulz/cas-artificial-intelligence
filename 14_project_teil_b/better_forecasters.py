"""
Better Forecasting Models for Trading
Includes: N-BEATS, Transformer, and Ensemble
Much better than LSTM for financial time-series
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================================
# 1. N-BEATS (Fast, Accurate, Interpretable)
# ============================================================================

class NBeatsBlock(nn.Module):
    """Single N-BEATS block with basis expansion."""

    def __init__(self, lookback, forecast_horizon, hidden_size=128, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon

        # Main stack (fully connected layers)
        self.stack = nn.Sequential(
            nn.Linear(lookback, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Basis expansion heads
        self.backcast_basis = nn.Linear(hidden_size, lookback)
        self.forecast_basis = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_size, lookback]

        Returns
        -------
        forecast, backcast : torch.Tensor
            [batch_size, forecast_horizon], [batch_size, lookback]
        """
        h = self.stack(x)
        forecast = self.forecast_basis(h)
        backcast = self.backcast_basis(h)
        return forecast, backcast


class NBeatsForecaster(nn.Module):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time-series

    Won M4 competition (best accuracy across 100k+ time-series)
    """

    def __init__(self, lookback, forecast_horizon, num_blocks=4, hidden_size=128, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.num_blocks = num_blocks

        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            NBeatsBlock(lookback, forecast_horizon, hidden_size, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_size, lookback]

        Returns
        -------
        torch.Tensor
            [batch_size, forecast_horizon]
        """
        residual = x.clone()
        forecast_output = 0.0

        for block in self.blocks:
            forecast, backcast = block(residual)
            forecast_output = forecast_output + forecast
            residual = residual - backcast

        return forecast_output


# ============================================================================
# 2. Transformer (Better for Regime Changes)
# ============================================================================

class TransformerForecaster(nn.Module):
    """
    Transformer-based forecaster
    Good for: Capturing market regime changes, multi-horizon prediction
    """

    def __init__(self, lookback, forecast_horizon, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model

        # Input embedding
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(lookback, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(d_model * lookback, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, forecast_horizon)
        )

    def _create_positional_encoding(self, seq_len, d_model):
        """Create positional encoding for transformer."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_size, lookback]

        Returns
        -------
        torch.Tensor
            [batch_size, forecast_horizon]
        """
        batch_size = x.size(0)

        # Reshape to [batch_size, lookback, 1] then embed
        x = x.unsqueeze(-1)  # [batch, lookback, 1]
        x = self.input_projection(x)  # [batch, lookback, d_model]

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Transformer encoder
        x = self.transformer_encoder(x)  # [batch, lookback, d_model]

        # Flatten and decode
        x = x.reshape(batch_size, -1)  # [batch, lookback * d_model]
        x = self.decoder(x)  # [batch, forecast_horizon]

        return x


# ============================================================================
# 3. Ensemble (Combine Multiple Models)
# ============================================================================

class EnsembleForecaster(nn.Module):
    """
    Combines multiple forecasting models with learned weights.
    Usually better than any single model.
    """

    def __init__(self, lookback, forecast_horizon, device='cpu'):
        super().__init__()
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.device = device

        # Create individual models
        self.nbeats = NBeatsForecaster(lookback, forecast_horizon, num_blocks=3).to(device)
        self.transformer = TransformerForecaster(lookback, forecast_horizon).to(device)

        # Learned combination weights
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # Will be learned

    def forward(self, x):
        """Combine predictions from multiple models."""
        nbeats_pred = self.nbeats(x)
        transformer_pred = self.transformer(x)

        # Softmax weights for proper combination
        weights = torch.softmax(self.weights, dim=0)

        # Weighted ensemble
        return weights[0] * nbeats_pred + weights[1] * transformer_pred


# ============================================================================
# Training Utilities
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for time-series forecasting."""

    def __init__(self, data, lookback=20, forecast_horizon=5):
        """
        Parameters
        ----------
        data : np.ndarray
            [N, features] - usually just close price or returns
        lookback : int
        forecast_horizon : int
        """
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback : idx + self.lookback + self.forecast_horizon]
        return x, y


def train_forecaster(model, train_data, val_data=None, epochs=100, batch_size=32,
                     lr=0.001, device='cpu', lookback=20, forecast_horizon=5):
    """
    Train any forecasting model.

    Parameters
    ----------
    model : nn.Module
    train_data : np.ndarray
        [N, features]
    val_data : np.ndarray, optional
    epochs : int
    batch_size : int
    lr : float
    device : torch.device
    lookback : int
    forecast_horizon : int

    Returns
    -------
    dict
        Training history
    """
    # Data loaders
    train_dataset = TimeSeriesDataset(train_data, lookback, forecast_horizon)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_data is not None:
        val_dataset = TimeSeriesDataset(val_data, lookback, forecast_horizon)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')
        else:
            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f}')

    return history


def predict_forecaster(model, data, lookback=20, batch_size=32, device='cpu'):
    """
    Generate predictions on new data.

    Parameters
    ----------
    model : nn.Module
    data : np.ndarray
        [N, features]
    lookback : int
    batch_size : int
    device : torch.device

    Returns
    -------
    np.ndarray
        Predictions [N-lookback, forecast_horizon]
    """
    model.eval()
    dataset = TimeSeriesDataset(data, lookback=lookback, forecast_horizon=1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            pred = model(x)
            predictions.append(pred.cpu().numpy())

    return np.vstack(predictions) if predictions else np.array([])


# ============================================================================
# Comparison Utility
# ============================================================================

def compare_models(train_data, val_data, test_data, lookback=20, forecast_horizon=5):
    """
    Train and compare all three models.

    Returns
    -------
    dict
        Comparison results and trained models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    results = {}

    # 1. N-BEATS
    print("Training N-BEATS...")
    nbeats = NBeatsForecaster(lookback, forecast_horizon, num_blocks=3).to(device)
    history_nbeats = train_forecaster(nbeats, train_data, val_data, epochs=100, device=device,
                                     lookback=lookback, forecast_horizon=forecast_horizon)
    val_loss_nbeats = history_nbeats['val_loss'][-1] if history_nbeats['val_loss'] else float('inf')
    print(f"✓ N-BEATS Val Loss: {val_loss_nbeats:.6f}\n")
    results['nbeats'] = {'model': nbeats, 'history': history_nbeats, 'val_loss': val_loss_nbeats}

    # 2. Transformer
    print("Training Transformer...")
    transformer = TransformerForecaster(lookback, forecast_horizon).to(device)
    history_transformer = train_forecaster(transformer, train_data, val_data, epochs=100, device=device,
                                          lookback=lookback, forecast_horizon=forecast_horizon)
    val_loss_transformer = history_transformer['val_loss'][-1] if history_transformer['val_loss'] else float('inf')
    print(f"✓ Transformer Val Loss: {val_loss_transformer:.6f}\n")
    results['transformer'] = {'model': transformer, 'history': history_transformer, 'val_loss': val_loss_transformer}

    # 3. Ensemble
    print("Training Ensemble...")
    ensemble = EnsembleForecaster(lookback, forecast_horizon, device=device)
    history_ensemble = train_forecaster(ensemble, train_data, val_data, epochs=100, device=device,
                                       lookback=lookback, forecast_horizon=forecast_horizon)
    val_loss_ensemble = history_ensemble['val_loss'][-1] if history_ensemble['val_loss'] else float('inf')
    print(f"✓ Ensemble Val Loss: {val_loss_ensemble:.6f}\n")
    results['ensemble'] = {'model': ensemble, 'history': history_ensemble, 'val_loss': val_loss_ensemble}

    # Summary
    print("="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"N-BEATS Validation Loss:    {val_loss_nbeats:.6f}")
    print(f"Transformer Validation Loss: {val_loss_transformer:.6f}")
    print(f"Ensemble Validation Loss:    {val_loss_ensemble:.6f}")
    print(f"\nBest Model: {min(results.items(), key=lambda x: x[1]['val_loss'])[0].upper()}")
    print("="*60)

    return results


if __name__ == "__main__":
    # Test example
    print("Testing forecasting models...")

    # Create dummy data
    np.random.seed(42)
    trend = np.sin(np.linspace(0, 8*np.pi, 500))
    noise = np.random.normal(0, 0.1, 500)
    data = trend + noise

    # Split
    split1 = int(0.6 * len(data))
    split2 = int(0.8 * len(data))
    train = data[:split1].reshape(-1, 1)
    val = data[split1:split2].reshape(-1, 1)
    test = data[split2:].reshape(-1, 1)

    # Compare
    results = compare_models(train, val, test, lookback=20, forecast_horizon=5)
    print("\n✅ All models trained successfully!")

