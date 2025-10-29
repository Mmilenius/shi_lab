import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Try import shap, give clear instruction if missing
try:
    import shap
except Exception as e:
    raise ImportError(
        "Бібліотека 'shap' не знайдена. Встановіть її:\n"
        "pip install shap\n\n"
        f"Точна помилка: {e}"
    )


# ---------------------------
# 1) Load or generate data
# ---------------------------
def generate_synthetic_energy_data(n_samples=3000, random_state=42):
    np.random.seed(random_state)
    timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="H")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.random.normal(15, 10, n_samples),
        "humidity": np.random.uniform(30, 90, n_samples),
        "wind_speed": np.random.uniform(0, 10, n_samples),
        "building_area": np.random.normal(250, 100, n_samples),
        "building_age": np.random.randint(1, 50, n_samples),
        "electric_usage": np.random.normal(15, 5, n_samples),
        "heating_usage": np.random.normal(8, 3, n_samples),
        "cooling_usage": np.random.normal(5, 2, n_samples),
        "occupancy": np.random.randint(1, 100, n_samples),
        "region_factor": np.random.uniform(0.8, 1.2, n_samples)
    })
    df["total_energy"] = (
            df["electric_usage"] * df["region_factor"] +
            0.6 * df["heating_usage"] +
            0.4 * df["cooling_usage"] +
            (df["building_area"] / 500) +
            np.random.normal(0, 1, n_samples)
    )
    # simple categorical example
    df["insulation_type"] = np.random.choice(["A", "B", "C"], size=len(df))
    df = pd.get_dummies(df, columns=["insulation_type"], drop_first=True)
    return df


DATA_CSV = "output/synthetic_energy_full.csv"
if os.path.exists(DATA_CSV):
    data = pd.read_csv(DATA_CSV)
else:
    data = generate_synthetic_energy_data()
    # Створимо папку, якщо генеруємо дані
    os.makedirs("output", exist_ok=True)
    # data.to_csv(DATA_CSV, index=False) # Можна розкоментувати, щоб зберегти

# ---------------------------
# 2) Prepare features/target
# ---------------------------
target = "total_energy"
X = data.drop(columns=[target, "timestamp"], errors="ignore")
y = data[target].values

# train/test split (time not strictly preserved here since synthetic; adapt if needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features for model stability / SHAP interpretability not strictly required for trees
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ---------------------------
# 3) Train a tree-based model
# ---------------------------
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------------------
# 4) SHAP analysis (TreeExplainer)
# ---------------------------
print("[INFO] Розрахунок SHAP... Це може зайняти деякий час.")
explainer = shap.TreeExplainer(model)  # good for tree-based models
# For large datasets, compute SHAP on a sample
sample_for_shap = X_train_scaled.sample(n=min(500, len(X_train_scaled)), random_state=42)
shap_values = explainer.shap_values(sample_for_shap)  # old API
# New unified API:
# shap_values = explainer(sample_for_shap)

# ---------------------------
# 5) Visualizations (save to files)
# ---------------------------
print("[INFO] Збереження візуалізацій SHAP...")
os.makedirs("output/shap_plots", exist_ok=True)

# Summary plot (bee swarm)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, sample_for_shap, show=False)
plt.title("SHAP summary (bee-swarm) — важливість та вплив ознак")
plt.savefig("output/shap_plots/shap_summary_beeswarm.png", bbox_inches="tight", dpi=150)
plt.close()

# Bar plot (mean absolute SHAP values)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, sample_for_shap, plot_type="bar", show=False)
plt.title("SHAP bar — середня абсолютна важливість ознак")
plt.savefig("output/shap_plots/shap_summary_bar.png", bbox_inches="tight", dpi=150)
plt.close()

# Dependence plot for the top feature
# Identify top feature by mean(|shap|)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = int(np.argmax(mean_abs_shap))
top_feature = sample_for_shap.columns[top_idx]

plt.figure(figsize=(8, 6))
# Використовуємо interaction_index=None для уникнення попереджень
shap.dependence_plot(top_feature, shap_values, sample_for_shap, show=False, interaction_index=None)
plt.title(f"SHAP dependence plot — {top_feature}")
plt.savefig(f"output/shap_plots/shap_dependence_{top_feature}.png", bbox_inches="tight", dpi=150)
plt.close()


# ---------------------------
# 6) Textual explanation for a single instance
# ---------------------------
def explain_instance(idx, X_df, model, explainer, scaler=None, top_k=5):
    """
    Повертає текстове пояснення для рядка з індексом idx (в X_df).
    X_df — original unscaled dataframe used for predictions (or scaled accordingly).
    """
    # Prepare instance
    x = X_df.loc[idx:idx] if isinstance(idx, (int, np.integer)) else X_df.loc[[idx]]
    # If scaler was applied externally, assume X_df is scaled. If not, user should pass scaled X.
    x_arr = x.values

    # SHAP values for this instance
    shap_vals = explainer.shap_values(x)  # shape (1, n_features)
    shap_vals = np.array(shap_vals).reshape(-1)  # flatten
    feature_names = X_df.columns.tolist()

    base_value = explainer.expected_value if hasattr(explainer, "expected_value") else explainer.base_value
    pred = model.predict(x)[0]

    # --- ПОМИЛКА БУЛА ТУТ ---
    # explainer.expected_value може повертати array([value]), а не float
    # Нам потрібно витягнути саме число (float)
    base_value_float = base_value
    if isinstance(base_value_float, np.ndarray):
        base_value_float = base_value_float[0]
        # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---

    # Pair feature, value, shap
    feat_info = []
    for fname, fval, sval in zip(feature_names, x_arr.flatten(), shap_vals):
        feat_info.append((fname, float(fval), float(sval), abs(float(sval))))
    # sort by absolute shap
    feat_info_sorted = sorted(feat_info, key=lambda t: t[3], reverse=True)

    # Build explanation strings
    explanation_lines = []
    # Використовуємо виправлене значення base_value_float
    explanation_lines.append(f"Прогноз (вклад базового значення {base_value_float:.3f}): {pred:.3f}")
    explanation_lines.append("Топ фічі, що вплинули на прогноз (в порядку значущості):")

    for fname, fval, sval, _ in feat_info_sorted[:top_k]:
        sign = "підвищують" if sval > 0 else "знижують"
        explanation_lines.append(
            f" • {fname}: значення={fval:.3f}, SHAP={sval:.3f} → {sign} прогноз на {abs(sval):.3f}")

    # Also list small contributors
    small_contributors = [(f, v, s) for f, v, s, _ in feat_info_sorted[top_k:top_k + 5]]
    if small_contributors:
        explanation_lines.append("\nІнші (менш значущі) впливи:")
        for fname, fval, sval in small_contributors:
            explanation_lines.append(f"   - {fname}: {sval:.3f}")

    return "\n".join(explanation_lines)


# Choose an instance from test set (use scaled X_test_scaled)
print("[INFO] Генерація текстового пояснення для одного прикладу...")
example_idx = X_test_scaled.index[0]
text_explanation = explain_instance(example_idx, X_test_scaled, model, explainer, scaler=scaler, top_k=6)

# Save textual explanation to file and print
with open("output/shap_plots/explanation_example.txt", "w", encoding="utf-8") as f:
    f.write(text_explanation)

print("\n====================================================================")
print("SHAP analysis виконано. Файли з візуалізаціями та поясненням збережені в 'output/shap_plots/'.")
print("====================================================================")
print("\nКоротке текстове пояснення для прикладу (з файлу explanation_example.txt):\n")
print(text_explanation)
