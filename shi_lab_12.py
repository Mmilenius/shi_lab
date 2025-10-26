"""
anfis_scoring_improved.py
Оновлена версія: виправлено AUC/ROC, стабілізовано навчання (BCEWithLogits),
навчання на більшому піднаборі (default 500), покращена ініціалізація.
Вимоги: numpy pandas scikit-fuzzy scikit-learn torch matplotlib
pip install numpy pandas scikit-fuzzy scikit-learn torch matplotlib
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------------- генерація синтетичних даних ----------------
def generate_synthetic_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    df = pd.DataFrame()
    df['revenue'] = np.abs(np.random.normal(loc=500, scale=200, size=n_samples))
    df['profit'] = df['revenue'] * np.random.uniform(-0.1, 0.3, size=n_samples)
    df['working_capital'] = np.abs(np.random.normal(loc=50, scale=30, size=n_samples))
    df['company_age'] = np.random.exponential(scale=5, size=n_samples)
    df['industry_code'] = np.random.randint(0, 6, size=n_samples)
    df['employees'] = np.random.poisson(lam=25, size=n_samples)
    df['past_defaults'] = np.random.poisson(lam=0.2, size=n_samples)
    df['num_loans'] = np.random.poisson(lam=1.2, size=n_samples)
    df['avg_delay_days'] = np.abs(np.random.normal(loc=5, scale=20, size=n_samples))
    df['management_quality'] = np.clip(np.random.beta(2,2,size=n_samples),0,1)
    df['market_position'] = np.clip(np.random.beta(2,3,size=n_samples),0,1)
    df['liquidity_ratio'] = np.clip(df['working_capital'] / (df['revenue'] + 1e-6), 0, 5)

    scaler = MinMaxScaler()
    numeric_cols = ['revenue','profit','working_capital','company_age','employees',
                    'past_defaults','num_loans','avg_delay_days','management_quality',
                    'market_position','liquidity_ratio']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Правило для "ground truth" risk
    risk_score = (
        0.45 * df['past_defaults'] +
        0.25 * df['avg_delay_days'] +
        0.2 * (1 - df['management_quality']) +
        0.1 * (1 - df['market_position'])
    )
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    risk_score = np.clip(risk_score + np.random.normal(0, 0.04, size=n_samples), 0, 1)
    df['risk'] = risk_score
    return df

# ---------------- Gaussian MF з кращою ініціалізацією ----------------
class GaussianMF(nn.Module):
    def __init__(self, init_c, init_sigma):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(float(init_c), dtype=torch.float32))
        # log_sigma з ініціалізацією трохи більшою, щоб не бути занадто вузькими
        self.log_sigma = nn.Parameter(torch.tensor(np.log(max(1e-3, init_sigma)), dtype=torch.float32))

    def forward(self, x):
        sigma = torch.exp(self.log_sigma) + 1e-6
        return torch.exp(-0.5 * ((x - self.c) / sigma) ** 2)

# ---------------- ANFIS-like модель (повертає логіт, без sigmoid) ----------------
class NeuroFuzzyANFIS(nn.Module):
    def __init__(self, input_names, num_mfs=3, rules=None):
        super().__init__()
        self.input_names = input_names
        self.M = len(input_names)
        self.K = num_mfs

        centers_init = np.linspace(0.15, 0.85, self.K)
        self.mfs = nn.ModuleDict()
        for name in input_names:
            mfs_for_var = nn.ModuleList()
            for c in centers_init:
                # трохи більша sigma на старті
                mfs_for_var.append(GaussianMF(init_c=c, init_sigma=0.18))
            self.mfs[name] = mfs_for_var

        if rules is None:
            np.random.seed(0)
            R = max(5, self.K ** self.M // 10)
            rules = []
            for _ in range(R):
                rule = list(np.random.randint(0, self.K, size=self.M))
                rules.append(rule)
        self.rules = rules
        self.R = len(rules)

        # consequents ініціалізуємо трохи ширше (щоб не застрягати на 0)
        self.consequents = nn.Parameter(torch.randn(self.R, self.M + 1) * 0.5)

    def forward(self, x):
        batch = x.shape[0]
        mu_vars = []
        for i, name in enumerate(self.input_names):
            xi = x[:, i].unsqueeze(1)
            mus = [mf(xi) for mf in self.mfs[name]]
            mus = torch.cat(mus, dim=1)
            mu_vars.append(mus)

        firing_raw = []
        for r, rule in enumerate(self.rules):
            mu_selected = [mu_vars[i][:, rule[i]].unsqueeze(1) for i in range(self.M)]
            mu_stack = torch.cat(mu_selected, dim=1)
            w_r = torch.prod(mu_stack, dim=1)
            firing_raw.append(w_r.unsqueeze(1))
        firing_raw = torch.cat(firing_raw, dim=1)

        w_sum = torch.sum(firing_raw, dim=1, keepdim=True) + 1e-8
        w_norm = firing_raw / w_sum

        x_with_bias = torch.cat([torch.ones(batch,1, device=x.device), x], dim=1)
        y_r = torch.matmul(x_with_bias, self.consequents.t())

        y = torch.sum(w_norm * y_r, dim=1)  # логіт без sigmoid
        return y, w_norm, y_r

# ---------------- підготовка, навчання (за замовчуванням більше прикладів) ----------------
def prepare_and_train(df, small_n=500, epochs=400, lr=0.01):
    input_names = ['revenue','profit','past_defaults','management_quality','avg_delay_days']
    X = df[input_names].values.astype(np.float32)
    y = df['risk'].values.astype(np.float32)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    small_n = min(small_n, X_train_full.shape[0])
    idx = np.random.RandomState(1).choice(X_train_full.shape[0], small_n, replace=False)
    X_train = X_train_full[idx]
    y_train = y_train_full[idx]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Правила (ті самі інтуїтивні 5 правил)
    rules = [
        [2,2,0,2,0],
        [1,1,1,1,1],
        [0,0,2,0,2],
        [2,0,2,1,2],
        [0,1,1,0,1],
    ]
    model = NeuroFuzzyANFIS(input_names=input_names, num_mfs=3, rules=rules)

    # BCEWithLogitsLoss працює на логітах і стабільніший для бінарної задачі
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Тренування
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} — loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits_test, rule_w, y_r = model(X_test_t)
        probs_test = torch.sigmoid(logits_test).numpy()

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_pred': probs_test,
        'input_names': input_names,
        'rules': rules,
        'rule_weights': rule_w.numpy()
    }

# ---------------- метрики (тепер AUC обчислюється коректно) ----------------
def evaluate_results(y_true_cont, y_pred_prob, threshold=0.5):
    y_true_bin = (y_true_cont >= threshold).astype(int)
    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    try:
        auc = roc_auc_score(y_true_bin, y_pred_prob)
    except Exception:
        auc = float('nan')

    print("=== Оцінка моделі ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    return {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'auc':auc}

# ---------------- порівняння клієнтів ----------------
def compare_clients(model, df, clients_idx=None, n_show=6):
    if clients_idx is None:
        clients_idx = np.random.choice(df.shape[0], n_show, replace=False)
    selected = df.iloc[clients_idx]
    input_names = model.input_names
    X_sel = torch.tensor(selected[input_names].values.astype(np.float32))
    with torch.no_grad():
        logits, w_norm, y_r = model(X_sel)
        probs = torch.sigmoid(logits).numpy()
    res = selected.copy()
    res['predicted_risk'] = probs
    display_cols = input_names + ['predicted_risk','risk']
    print("\n=== Порівняння ризику для вибраних клієнтів ===")
    print(res[display_cols].round(3).to_string(index=True))
    return res

# ---------------- приклад функцій належності ----------------
def demonstrate_fuzzy_memberships(var='revenue'):
    x = np.linspace(0,1,200)
    low = fuzz.trimf(x, [0, 0, 0.4])
    med = fuzz.trimf(x, [0.2, 0.5, 0.8])
    high = fuzz.trimf(x, [0.6, 1.0, 1.0])
    plt.figure(figsize=(6,3))
    plt.plot(x, low, label='low')
    plt.plot(x, med, label='medium')
    plt.plot(x, high, label='high')
    plt.title(f"Функції належності для '{var}'")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------- main ----------------
if __name__ == "__main__":
    df = generate_synthetic_data(n_samples=2000)
    print("Приклад датасету (перші 5):")
    print(df.head().round(3).to_string(index=False))
    demonstrate_fuzzy_memberships('revenue')

    # Тренуємо на bigger small_n (default 500). Змінюйте small_n для експериментів.
    results = prepare_and_train(df, small_n=500, epochs=400, lr=0.01)

    metrics = evaluate_results(results['y_test'], results['y_test_pred'])

    # ROC — потребує бінарних y_true
    y_true_bin = (results['y_test'] >= 0.5).astype(int)
    try:
        fpr, tpr, _ = roc_curve(y_true_bin, results['y_test_pred'])
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--')
        plt.title("ROC curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Не вдалося побудувати ROC:", e)

    compare_clients(results['model'], df, n_show=6)
    print("\n--- Готово ---")
