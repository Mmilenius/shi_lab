#!/usr/bin/env python3
# lab5_variant7_full.py
"""
Лабораторна робота №5 — варіант 7
Повний скрипт: генерація даних, ANFIS-like модель (PyTorch), навчання (RMSprop, batch_size=75, epochs=150),
пермутаційна важливість, збереження графіків/CSV та створення Word-звіту.

Автор: автоматично згенерований код для М. Лукомського
Дисципліна: Системи штучного інтелекту
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

# Optional fuzzy demo
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except Exception:
    SKFUZZY_AVAILABLE = False

# For Word report
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

# -------------------------
# Config / Hyperparameters
# -------------------------
OUT_DIR = "./lab5_variant7_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_SAMPLES = 1200
TEST_SIZE = 0.2

BATCH_SIZE = 75          # варіант 7
EPOCHS = 150             # варіант 7
LR = 0.01
N_RULES = 8

# -------------------------
# 1) Генерація даних
# -------------------------
def generate_data(n_samples=N_SAMPLES, seed=SEED):
    rng = np.random.RandomState(seed)
    temp = rng.uniform(20, 100, n_samples)    # Temperature °C
    press = rng.uniform(1, 10, n_samples)     # Pressure bar
    humid = rng.uniform(10, 90, n_samples)    # Humidity %
    # Формула target + шум (та обмеження 0..1)
    risk = (0.5 * (temp / 100.0) +
            0.3 * (1 - press / 10.0) +
            0.2 * (humid / 100.0) +
            0.1 * np.sin(temp / 10.0) +
            rng.normal(0, 0.03, n_samples))
    risk = np.clip(risk, 0.0, 1.0)
    X = np.vstack([temp, press, humid]).T
    return X, risk

# -------------------------
# 2) Простий ANFIS-like модель (PyTorch)
# -------------------------
class SimpleANFIS(nn.Module):
    def __init__(self, n_inputs=3, n_rules=N_RULES):
        super().__init__()
        # premise: лінійна проєкція входів в простір правил (можна інтерпретувати як центри)
        self.premise = nn.Linear(n_inputs, n_rules, bias=True)
        # параметр масштабів для правил (буде застосовано softplus -> додатний)
        self.scale_param = nn.Parameter(torch.rand(n_rules))
        # змішування: перетворює нормалізовані ваги правил у скалярний вихід
        self.mix = nn.Linear(n_rules, 1, bias=True)

    def forward(self, x):
        # x: [batch, n_inputs]
        a = self.premise(x)  # [batch, n_rules]
        scale = torch.nn.functional.softplus(self.scale_param) + 1e-9  # >0
        # gaussian-подібні активації відносно a та scale
        activations = torch.exp(-0.5 * (a / scale)**2)  # [batch, n_rules]
        weights = activations / (activations.sum(dim=1, keepdim=True) + 1e-9)
        y = self.mix(weights)  # [batch, 1]
        return torch.sigmoid(y)  # привести в діапазон 0..1

# -------------------------
# 3) Навчання та оцінка
# -------------------------
def train_and_evaluate(X, y,
                       test_size=TEST_SIZE,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       lr=LR,
                       out_dir=OUT_DIR):
    # нормалізація
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=test_size, random_state=SEED)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SimpleANFIS(n_inputs=X.shape[1], n_rules=N_RULES)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_test_t)
            val_loss = loss_fn(y_val_pred, y_test_t).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 25 == 0 or epoch == 0 or (epoch+1)==epochs:
            print(f"Epoch {epoch+1}/{epochs} — Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

    # фінальна оцінка
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t).numpy().flatten()
    y_true = y_test

    rmse = mean_squared_error(y_true, y_pred_test, squared=False)
    mae = mean_absolute_error(y_true, y_pred_test)
    r2 = r2_score(y_true, y_pred_test)

    results = {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

    # save metrics to CSV
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R2"],
        "Value": [rmse, mae, r2]
    })
    metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    return results

# -------------------------
# 4) Пермутаційна важливість
# -------------------------
def permutation_importance(model, X_test_np, y_true_np, metric_fn, n_rounds=20, seed=SEED):
    rng = np.random.RandomState(seed)
    device = next(model.parameters()).device if len(list(model.parameters()))>0 else "cpu"
    # baseline
    with torch.no_grad():
        base_pred = model(torch.tensor(X_test_np, dtype=torch.float32)).numpy().flatten()
    baseline = metric_fn(y_true_np, base_pred)
    importances = np.zeros(X_test_np.shape[1])
    for j in range(X_test_np.shape[1]):
        scores = []
        X_permuted = X_test_np.copy()
        for r in range(n_rounds):
            idx = rng.permutation(len(X_permuted))
            X_permuted[:, j] = X_test_np[idx, j]
            with torch.no_grad():
                pred = model(torch.tensor(X_permuted, dtype=torch.float32)).numpy().flatten()
            score = metric_fn(y_true_np, pred)
            scores.append(score)
        importances[j] = np.mean(scores) - baseline  # збільшення метрики (RMSE) -> показник погіршення
    # нормалізація відносних важливостей
    rel = importances / (np.sum(importances) + 1e-12)
    return baseline, importances, rel

# -------------------------
# 5) Візуалізації та збереження
# -------------------------
def save_plots_and_tables(results, importances, rel_importances, out_dir=OUT_DIR):
    X_test = results["X_test"]
    y_true = results["y_test"]
    y_pred_test = results["y_pred_test"]

    # Loss curve
    plt.figure(figsize=(8,5))
    plt.plot(results["train_losses"], label='train_loss')
    plt.plot(results["val_losses"], label='val_loss')
    plt.title('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(out_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # True vs Predicted (first up to 100)
    Nplot = min(100, len(y_true))
    plt.figure(figsize=(10,4))
    plt.plot(y_true[:Nplot], 'o-', label='True', alpha=0.7)
    plt.plot(y_pred_test[:Nplot], 'o-', label='Predicted', alpha=0.7)
    plt.title('True vs Predicted (first samples)')
    plt.xlabel('Sample index')
    plt.ylabel('Risk')
    plt.legend()
    plt.grid(True)
    comp_path = os.path.join(out_dir, "true_vs_predicted.png")
    plt.tight_layout()
    plt.savefig(comp_path)
    plt.close()

    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred_test, s=15, alpha=0.6)
    plt.plot([0,1],[0,1], 'r--')
    plt.title('Scatter: True vs Predicted')
    plt.xlabel('True Risk')
    plt.ylabel('Predicted Risk')
    plt.grid(True)
    scatter_path = os.path.join(out_dir, "scatter_true_pred.png")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    # Heatmap of importances (1 x n_features)
    plt.figure(figsize=(6,2.5))
    imp_matrix = rel_importances.reshape(1, -1)
    im = plt.imshow(imp_matrix, aspect='auto')
    plt.yticks([])
    plt.xticks(range(X_test.shape[1]), ['Temperature', 'Pressure', 'Humidity'])
    plt.colorbar(im, orientation='vertical', label='Relative importance')
    plt.title('Heatmap of Permutation Importances (relative)')
    heatmap_path = os.path.join(out_dir, "heatmap_importances.png")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    # Save importances table
    importances_df = pd.DataFrame({
        "Feature": ["Temperature", "Pressure", "Humidity"],
        "Absolute importance (RMSE increase)": importances,
        "Relative importance": rel_importances
    })
    importances_df.to_csv(os.path.join(out_dir, "importances.csv"), index=False)

    return {
        "loss": loss_path,
        "comp": comp_path,
        "scatter": scatter_path,
        "heatmap": heatmap_path,
        "metrics": os.path.join(out_dir, "metrics.csv"),
        "importances": os.path.join(out_dir, "importances.csv")
    }