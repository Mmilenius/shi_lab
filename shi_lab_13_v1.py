import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models

# --- 1. Дані ---
np.random.seed(42)
normal_data = np.random.normal(50, 5, 500)
anomaly_data = np.random.normal(80, 3, 20)
data = np.concatenate([normal_data, anomaly_data])
df = pd.DataFrame(data, columns=["consumption"])

# --- 2. Масштабування ---
scaler = MinMaxScaler()
df["scaled"] = scaler.fit_transform(df[["consumption"]])

# --- 3. Isolation Forest ---
iso = IsolationForest(contamination=0.05, random_state=42)
df["iforest_pred"] = iso.fit_predict(df[["scaled"]])
df["iforest_anomaly"] = df["iforest_pred"] == -1

# --- 4. Autoencoder ---
# тут виправлено: беремо форму (n_samples, 1)
X = df[["scaled"]].values

autoencoder = models.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=30, batch_size=16, verbose=0)

# --- 5. Відновлення та пошук аномалій ---
reconstructed = autoencoder.predict(X)
mse = np.mean(np.power(X - reconstructed, 2), axis=1)
threshold = np.percentile(mse, 95)
df["autoenc_anomaly"] = mse > threshold

# --- 6. Сповіщення ---
anomalies_iforest = df[df["iforest_anomaly"]]
anomalies_autoenc = df[df["autoenc_anomaly"]]

if len(anomalies_iforest) > 0 or len(anomalies_autoenc) > 0:
    print("⚠️ Виявлено аномалії у споживанні!")
    print(f"- IsolationForest: {len(anomalies_iforest)} підозрілих точок")
    print(f"- Autoencoder: {len(anomalies_autoenc)} підозрілих точок")
else:
    print("✅ Аномалії не виявлено")

print(df.head(10))
