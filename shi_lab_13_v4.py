import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- Завантаження синтетичних даних ---
df = pd.read_csv("output/synthetic_energy_full.csv")

# Вибір ознак і цільової змінної
X = df.drop(columns=['timestamp', 'total_energy'])
y = df['total_energy']

# Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Навчання моделі
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_scaled, y)

# SHAP аналіз
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# Графік топ-5 ознак за впливом
shap.summary_plot(shap_values, X, plot_type="bar", max_display=5, show=True)

# Якщо хочеш індивідуальний приклад (перший рядок)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=True)
plt.show()
