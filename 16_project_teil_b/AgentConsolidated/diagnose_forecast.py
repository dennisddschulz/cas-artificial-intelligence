"""
Diagnose Script für Forecast-Qualität
Checkt ob LSTM gut genug ist und warum Performance schlecht ist
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from trading_config import ForecastingConfig, DataConfig

# ============================================================
# LOAD DATA - EXAKT WIE IN trading_framework.py
# ============================================================

data_config = DataConfig()
print(f"\n{'='*70}")
print(f"LOADING DATA: {data_config.ticker}")
print(f"{'='*70}")

df = yf.download(
    data_config.ticker,
    start=data_config.start_date,
    end=data_config.end_date,
    progress=False
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [c.lower() for c in df.columns]
df = df.dropna()

print(f"✓ Loaded {len(df)} days of data")
print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

# ============================================================
# ADD FEATURES - EXAKT WIE IN trading_framework.py
# ============================================================

df["log_close"] = np.log(df["close"])
df["r"] = df["log_close"].diff()

# Historical volatility (20-day window)
df["sigma_hat"] = df["r"].rolling(window=20, min_periods=1).std()
df["sigma_hat"] = df["sigma_hat"].fillna(df["sigma_hat"].mean())

# Momentum (20-day SMA)
df["mu_hat"] = df["r"].ewm(span=20, adjust=False).mean()

# Lagged return
df["r_lag1"] = df["r"].shift(1)

# MACD
df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
df["macd_diff"] = df["ema_12"] - df["ema_26"]

# RSI
delta = df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-8)
df["rsi"] = 100 - (100 / (1 + rs))

# Signal strength (simple)
df["signal_strength"] = np.sign(df["r"]).rolling(window=5).mean()

df = df.dropna()

print(f"✓ Added features")
print(f"  Data shape: {df.shape}")

# ============================================================
# SPLIT DATA
# ============================================================

n_total = len(df)
n_train = int(n_total * 0.6)
n_val = int(n_total * 0.2)

df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train + n_val]
df_test = df.iloc[n_train + n_val:]

print(f"\n✓ Data split:")
print(f"  Train: {len(df_train)} days ({len(df_train)/n_total*100:.1f}%)")
print(f"  Val:   {len(df_val)} days ({len(df_val)/n_total*100:.1f}%)")
print(f"  Test:  {len(df_test)} days ({len(df_test)/n_total*100:.1f}%)")

# ============================================================
# PREPARE DATA FOR LSTM - EXAKT WIE IN trading_framework.py
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_cols = ['sigma_hat', 'rsi', 'macd_diff', 'signal_strength']
available_cols = [c for c in feature_cols if c in df_train.columns]

scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[available_cols].values)
X_val = scaler.transform(df_val[available_cols].values)
X_test = scaler.transform(df_test[available_cols].values)

# Direction label - NO look-ahead bias
y_train = (df_train['r'].shift(-1) > 0).astype(int).fillna(0).values
y_val = (df_val['r'].shift(-1) > 0).astype(int).fillna(0).values
y_test = (df_test['r'].shift(-1) > 0).astype(int).fillna(0).values

# Create sequences
lookback = 20
def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, lookback)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)

print(f"\n✓ Sequences created:")
print(f"  Train sequences: {X_train_seq.shape[0]}")
print(f"  Val sequences:   {X_val_seq.shape[0]}")
print(f"  Test sequences:  {X_test_seq.shape[0]}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_seq).to(device)
y_train_t = torch.FloatTensor(y_train_seq).unsqueeze(1).to(device)
X_val_t = torch.FloatTensor(X_val_seq).to(device)
y_val_t = torch.FloatTensor(y_val_seq).unsqueeze(1).to(device)
X_test_t = torch.FloatTensor(X_test_seq).to(device)
y_test_t = torch.FloatTensor(y_test_seq).unsqueeze(1).to(device)

# ============================================================
# LSTM MODEL
# ============================================================

class LSTMForecaster(nn.Module):
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
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output, lstm_out

# ============================================================
# TRAIN LSTM
# ============================================================

print(f"\n{'='*70}")
print(f"TRAINING LSTM FORECASTER")
print(f"{'='*70}")

model = LSTMForecaster(len(available_cols), hidden_dim=64, num_layers=2, dropout=0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
criterion = nn.BCELoss()

best_val_loss = float('inf')
patience_counter = 0
best_state = None

for epoch in range(100):
    model.train()
    train_loss = 0
    bs = 32
    
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
    
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= 20:
            print(f"Early stop at epoch {epoch+1}")
            model.load_state_dict(best_state)
            break
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# ============================================================
# EVALUATE FORECAST QUALITY
# ============================================================

print(f"\n{'='*70}")
print(f"LSTM FORECAST QUALITY ASSESSMENT")
print(f"{'='*70}")

model.eval()
with torch.no_grad():
    train_pred_probs = torch.sigmoid(model(X_train_t)[0]).cpu().numpy().flatten()
    val_pred_probs = torch.sigmoid(model(X_val_t)[0]).cpu().numpy().flatten()
    test_pred_probs = torch.sigmoid(model(X_test_t)[0]).cpu().numpy().flatten()

def calculate_accuracy(y_true, y_pred_probs):
    y_pred = (y_pred_probs > 0.5).astype(int)
    return (y_pred == y_true).mean()

def calculate_balanced_accuracy(y_true, y_pred_probs):
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Sensitivity (True Positive Rate)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    sensitivity = tp / (tp + fn + 1e-8)
    
    # Specificity (True Negative Rate)
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    specificity = tn / (tn + fp + 1e-8)
    
    return (sensitivity + specificity) / 2

train_acc = calculate_accuracy(y_train_seq, train_pred_probs)
val_acc = calculate_accuracy(y_val_seq, val_pred_probs)
test_acc = calculate_accuracy(y_test_seq, test_pred_probs)

train_bacc = calculate_balanced_accuracy(y_train_seq, train_pred_probs)
val_bacc = calculate_balanced_accuracy(y_val_seq, val_pred_probs)
test_bacc = calculate_balanced_accuracy(y_test_seq, test_pred_probs)

# Class distribution
train_up_ratio = y_train_seq.mean()
val_up_ratio = y_val_seq.mean()
test_up_ratio = y_test_seq.mean()

print(f"\nCLASS DISTRIBUTION:")
print(f"  Train: UP={train_up_ratio*100:.1f}%  DOWN={(1-train_up_ratio)*100:.1f}%")
print(f"  Val:   UP={val_up_ratio*100:.1f}%  DOWN={(1-val_up_ratio)*100:.1f}%")
print(f"  Test:  UP={test_up_ratio*100:.1f}%  DOWN={(1-test_up_ratio)*100:.1f}%")

print(f"\nACCURACY:")
print(f"  Train: {train_acc:.4f} (baseline: {max(train_up_ratio, 1-train_up_ratio):.4f})")
print(f"  Val:   {val_acc:.4f} (baseline: {max(val_up_ratio, 1-val_up_ratio):.4f})")
print(f"  Test:  {test_acc:.4f} (baseline: {max(test_up_ratio, 1-test_up_ratio):.4f})")

print(f"\nBALANCED ACCURACY (Sensitivity + Specificity / 2):")
print(f"  Train: {train_bacc:.4f}")
print(f"  Val:   {val_bacc:.4f}")
print(f"  Test:  {test_bacc:.4f}")

# Additional metrics
print(f"\nFORECAST QUALITY VERDICT:")

if test_acc > max(test_up_ratio, 1-test_up_ratio) + 0.05:
    print(f"  ✓ GOOD: Forecast beats baseline by 5%+")
    forecast_quality = "GOOD"
elif test_acc > max(test_up_ratio, 1-test_up_ratio) + 0.02:
    print(f"  ⚠ MARGINAL: Forecast beats baseline but <5%")
    forecast_quality = "MARGINAL"
else:
    print(f"  ✗ POOR: Forecast is NOT better than baseline!")
    forecast_quality = "POOR"

print(f"\nRECOMMENDATION:")
if forecast_quality == "GOOD":
    print(f"  → LSTM is good enough, but Check:")
    print(f"    1. Is forecast properly integrated in reward?")
    print(f"    2. Is there a look-ahead bias issue?")
    print(f"    3. Does agent actually USE the forecast signal?")
elif forecast_quality == "MARGINAL":
    print(f"  → LSTM barely beats random chance")
    print(f"  → Consider: Better features, longer lookback, or different model")
else:
    print(f"  → LSTM is WORSE than random guessing!")
    print(f"  → DO NOT USE FORECAST - Agent will learn wrong signals")
    print(f"  → Try: Different features or skip forecast altogether")

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Test Accuracy:         {test_acc:.4f}")
print(f"Baseline (always UP):  {max(test_up_ratio, 1-test_up_ratio):.4f}")
print(f"Advantage:             {(test_acc - max(test_up_ratio, 1-test_up_ratio))*100:+.2f}%")
print(f"Forecast Quality:      {forecast_quality}")
print(f"{'='*70}\n")

