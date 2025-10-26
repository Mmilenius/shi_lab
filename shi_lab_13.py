# === Практична робота №13 ===
# Огляд інтеграції AI у хмарні сервіси: Google Vertex AI, AWS SageMaker, Azure ML
# Основне завдання: прогнозування енергоспоживання будівель
# Автор: студент кафедри САПР, НУ "Львівська політехніка"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === XGBoost (опціонально) ===
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    print("Увага: xgboost не встановлений. Встановіть 'xgboost' для кращих результатів.")
    xgb_available = False

# === TensorFlow для LSTM ===
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# === 1) Генерація синтетичного датасету ===
def generate_synthetic_energy_data(n_samples=3000, random_state=42):
    np.random.seed(random_state)
    timestamps = [datetime(2020,1,1) + timedelta(hours=i) for i in range(n_samples)]

    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.random.normal(15, 10, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'wind_speed': np.random.uniform(0, 10, n_samples),
        'building_area': np.random.normal(250, 100, n_samples),
        'building_age': np.random.randint(1, 50, n_samples),
        'insulation_type': np.random.choice(['A', 'B', 'C'], n_samples),
        'hour': [t.hour for t in timestamps],
        'day_of_week': [t.weekday() for t in timestamps],
        'season': [((t.month%12)//3)+1 for t in timestamps],
        'electric_usage': np.random.normal(15, 5, n_samples),
        'heating_usage': np.random.normal(8, 3, n_samples),
        'cooling_usage': np.random.normal(5, 2, n_samples),
        'occupancy': np.random.randint(1, 100, n_samples),
        'region_factor': np.random.uniform(0.8, 1.2, n_samples)
    })

    # Ціль: сумарне енергоспоживання
    data['total_energy'] = (
        data['electric_usage'] * data['region_factor'] +
        0.6 * data['heating_usage'] +
        0.4 * data['cooling_usage'] +
        (data['building_area'] / 500) +
        np.random.normal(0, 1, n_samples)
    )

    # Кодування insulation_type
    data = pd.get_dummies(data, columns=['insulation_type'], drop_first=True)

    data.to_csv("output/synthetic_energy_full.csv", index=False)
    print(f"Rows: {len(data)}\nCSV saved: output/synthetic_energy_full.csv")
    return data

# === 2) Feature engineering ===
def feature_engineering(df):
    df = df.copy()
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    df['energy_ratio'] = df['electric_usage'] / (df['heating_usage'] + df['cooling_usage'] + 1)
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    return df

# === 3) Train/test split by time ===
def time_split(df, target='total_energy', split_ratio=0.8):
    df = df.sort_values('timestamp')
    split_index = int(len(df)*split_ratio)
    train, test = df.iloc[:split_index], df.iloc[split_index:]

    X_train = train.drop(columns=[target, 'timestamp'])
    y_train = train[target]
    X_test = test.drop(columns=[target, 'timestamp'])
    y_test = test[target]

    return X_train, X_test, y_train, y_test

# === 4) Feature selection via RandomForest ===
def feature_selection(X, y, n_features=13):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = importances.head(n_features).index.tolist()
    return X[selected], selected, importances

# === 5) Ensemble of models ===
def train_tree_models(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=150, random_state=42)
    xgb = XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=42) if xgb_available else None

    base_models = [('rf', rf), ('gb', gb)]
    if xgb:
        base_models.append(('xgb', xgb))

    stack = StackingRegressor(estimators=base_models, final_estimator=GradientBoostingRegressor())

    models = {'rf': rf, 'gb': gb, 'stack': stack}
    if xgb:
        models['xgb'] = xgb

    for name, model in models.items():
        model.fit(X_scaled, y_train)

    return models, scaler

# === 6) Optional XGBoost hyperparameter tuning ===
def tune_xgb(X, y):
    if not xgb_available:
        print("XGBoost недоступний, пропускаємо tuning.")
        return None
    xgb = XGBRegressor(random_state=42)
    params = {
        'max_depth': [3,5,7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100,200,300],
        'subsample': [0.8,1.0]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(xgb, params, cv=tscv, scoring='r2', n_iter=10, random_state=42)
    search.fit(X, y)
    print("Best XGB params:", search.best_params_)
    return search.best_estimator_

# === 7) LSTM model ===
def train_lstm(series, epochs=12):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(series.reshape(-1,1))

    X, y = [], []
    for i in range(20, len(data_scaled)):
        X.append(data_scaled[i-20:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)

    next_pred = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
    print("LSTM next-step aggregate prediction (approx):", float(scaler.inverse_transform(next_pred)))
    return model

# === 8) Evaluation ===
def evaluate_models(models, scaler, X_test, y_test):
    print("8) Evaluation on test set...")
    preds = {}
    for name, model in models.items():
        X_scaled = scaler.transform(X_test)
        preds[name] = model.predict(X_scaled)

        mae = mean_absolute_error(y_test, preds[name])
        mse_val = mean_squared_error(y_test, preds[name])
        rmse = np.sqrt(mse_val)
        r2 = r2_score(y_test, preds[name])

        print(f"{name.upper()} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return preds

# === Основний конвеєр ===
def main_pipeline():
    print("1) Генерація синтетичного датасету...")
    data = generate_synthetic_energy_data()

    print("2) Feature engineering...")
    data = feature_engineering(data)

    print("3) Train/test split by time (80/20 by timestamp)...")
    X_train, X_test, y_train, y_test = time_split(data)

    print("4) Feature selection via RandomForest...")
    X_train_sel, selected_features, importances = feature_selection(X_train, y_train)
    X_test_sel = X_test[selected_features]
    print(f"Selected features count: {len(selected_features)}")

    print("5) Training tree-based models (RF, GB, XGB, Stacking)...")
    models, scaler = train_tree_models(X_train_sel, y_train)

    print("6) (Optional) Tuning XGBoost with RandomizedSearchCV (may take time)...")
    if xgb_available:
        best_xgb = tune_xgb(scaler.transform(X_train_sel), y_train)
        if best_xgb:
            models['xgb_tuned'] = best_xgb.fit(scaler.transform(X_train_sel), y_train)
    else:
        print("XGBoost недоступний, пропускаємо tuning.")

    print("7) Training LSTM on aggregate time series (global average) ...")
    agg_series = data['total_energy'].rolling(window=5).mean().fillna(method='bfill').values
    train_lstm(agg_series)

    preds = evaluate_models(models, scaler, X_test_sel, y_test)

    print("\n=== Завершено успішно ===")

if __name__ == "__main__":
    main_pipeline()
