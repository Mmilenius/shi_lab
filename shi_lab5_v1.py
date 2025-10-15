import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, GRU
import matplotlib.pyplot as plt

# --- 1. Підготовка даних ---
try:
    # ✅ ВИПРАВЛЕНО: Змінено 'time' на 'DateTime'
    data = pd.read_csv('traffic_data.csv', index_col='DateTime', parse_dates=True)
    print("✅ Файл 'traffic_data.csv' успішно завантажено.")
except FileNotFoundError:
    print("⚠️ Файл 'traffic_data.csv' не знайдено.")
    # ... (код для симуляції залишається без змін) ...

# Нормалізація даних
scaler = MinMaxScaler()
# ✅ ВИПРАВЛЕНО: Змінено 'traffic_volume' на 'Vehicles'
scaled_data = scaler.fit_transform(data['Vehicles'].values.reshape(-1, 1))


# Створення навчальних послідовностей
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


seq_length = 24  # Використовуємо дані за 24 години для прогнозу
X, y = create_sequences(scaled_data, seq_length)

# Розділення на навчальну та тестову вибірки (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"✅ Дані розділено. Навчальний набір: {len(X_train)}, тестовий: {len(X_test)}")


# --- 2. Побудова та навчання моделей ---
def create_and_train_model(model_type, X_train, y_train, X_test, y_test):
    print(f"\n--- Створення та навчання моделі: {model_type} ---")

    model = Sequential()
    if model_type == 'RNN':
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50))
    elif model_type == 'LSTM':
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
    elif model_type == 'GRU':
        model.add(GRU(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(50))

    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    # Зменшимо кількість епох для швидшого тестування
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    print(f"✅ Модель {model_type} навчено.")
    return model


# Навчання трьох моделей
model_rnn = create_and_train_model('RNN', X_train, y_train, X_test, y_test)
model_lstm = create_and_train_model('LSTM', X_train, y_train, X_test, y_test)
model_gru = create_and_train_model('GRU', X_train, y_train, X_test, y_test)

# --- 3. Оцінка моделей та візуалізація ---
print("\n--- Оцінка продуктивності моделей ---")

# Прогнозування для кожної моделі
predictions_rnn = scaler.inverse_transform(model_rnn.predict(X_test))
predictions_lstm = scaler.inverse_transform(model_lstm.predict(X_test))
predictions_gru = scaler.inverse_transform(model_gru.predict(X_test))
y_test_original = scaler.inverse_transform(y_test)


# Розрахунок метрик якості
def evaluate_model(predictions, y_actual):
    mae = np.mean(np.abs(predictions - y_actual))
    rmse = np.sqrt(np.mean((predictions - y_actual) ** 2))
    return mae, rmse


mae_rnn, rmse_rnn = evaluate_model(predictions_rnn, y_test_original)
mae_lstm, rmse_lstm = evaluate_model(predictions_lstm, y_test_original)
mae_gru, rmse_gru = evaluate_model(predictions_gru, y_test_original)

# Виведення результатів у вигляді таблиці
results = pd.DataFrame({
    'Модель': ['Simple RNN', 'LSTM', 'GRU'],
    'MAE': [mae_rnn, mae_lstm, mae_gru],
    'RMSE': [rmse_rnn, rmse_lstm, rmse_gru]
})
print(results)

# --- 4. Візуалізація результатів ---
plt.figure(figsize=(18, 8))
plt.plot(data.index[train_size + seq_length:], y_test_original, color='blue', label='Реальний трафік', linewidth=2)
plt.plot(data.index[train_size + seq_length:], predictions_rnn, color='orange', linestyle='--',
         label=f'Прогноз RNN (RMSE: {rmse_rnn:.2f})')
plt.plot(data.index[train_size + seq_length:], predictions_lstm, color='red', linestyle='--',
         label=f'Прогноз LSTM (RMSE: {rmse_lstm:.2f})')
plt.plot(data.index[train_size + seq_length:], predictions_gru, color='green', linestyle='--',
         label=f'Прогноз GRU (RMSE: {rmse_gru:.2f})')

plt.title('Порівняння моделей RNN, LSTM та GRU для прогнозу трафіку')
plt.xlabel('Час')
plt.ylabel('Кількість транспортних засобів (Vehicles)')
plt.legend()
plt.grid(True)
plt.show()