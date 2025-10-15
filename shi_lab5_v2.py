import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN, GRU  # Додано SimpleRNN та GRU
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Завантаження та підготовка даних
try:
    # Спробуємо завантажити дані з правильною назвою стовпця 'DateTime'
    data = pd.read_csv("traffic_data.csv", index_col='DateTime', parse_dates=True)
    traffic_volume = data['Vehicles'].values.reshape(-1, 1)
    print("✅ Файл traffic_data.csv успішно завантажено.")
except (FileNotFoundError, KeyError):
    print("⚠️ Файл traffic_data.csv не знайдено або має неправильні стовпці. Створення симуляційних даних.")
    time = np.arange(0, 2000, 1)
    # Створення даних з добовою та тижневою сезонністю та шумом
    daily_pattern = np.sin(2 * np.pi * time / 24)
    weekly_pattern = np.sin(2 * np.pi * time / (24 * 7))
    traffic_volume = 150 * daily_pattern + 80 * weekly_pattern + 300 + np.random.normal(0, 25, 2000)
    traffic_volume = traffic_volume.reshape(-1, 1)

# Нормалізація даних
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(traffic_volume)


# 2. Створення навчальних послідовностей
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


seq_length = 24  # Використовуємо дані за 24 попередні години
X, y = create_sequences(scaled_data, seq_length)

# Розділення на навчальний та тестовий набори
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"✅ Дані розділено. Навчальний набір: {len(X_train)}, тестовий: {len(X_test)}")


# 3. Функція для побудови та навчання моделей
def create_and_train_model(model_type, X_train_data, y_train_data):
    print(f"\n--- Побудова та навчання моделі: {model_type} ---")
    model = Sequential()
    if model_type == 'RNN':
        model.add(SimpleRNN(50, activation='relu', input_shape=(seq_length, 1)))
    elif model_type == 'LSTM':
        model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    elif model_type == 'GRU':
        model.add(GRU(50, activation='relu', input_shape=(seq_length, 1)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Навчання моделі (зменшимо епохи для швидшого порівняння)
    model.fit(X_train_data, y_train_data, epochs=20, batch_size=32, verbose=0)
    print(f"✅ Модель {model_type} навчено.")
    return model


# Навчання трьох різних моделей
model_rnn = create_and_train_model('RNN', X_train, y_train)
model_lstm = create_and_train_model('LSTM', X_train, y_train)
model_gru = create_and_train_model('GRU', X_train, y_train)

# 4. Прогноз та оцінка для кожної моделі
print("\n--- Оцінка продуктивності моделей ---")
y_test_orig = scaler.inverse_transform(y_test)

# Прогнози
predictions_rnn_scaled = model_rnn.predict(X_test)
predictions_lstm_scaled = model_lstm.predict(X_test)
predictions_gru_scaled = model_gru.predict(X_test)

# Повернення до початкового масштабу
predictions_rnn = scaler.inverse_transform(predictions_rnn_scaled)
predictions_lstm = scaler.inverse_transform(predictions_lstm_scaled)
predictions_gru = scaler.inverse_transform(predictions_gru_scaled)


# Розрахунок метрик
def evaluate(predictions, actual):
    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
    return mae, rmse


mae_rnn, rmse_rnn = evaluate(predictions_rnn, y_test_orig)
mae_lstm, rmse_lstm = evaluate(predictions_lstm, y_test_orig)
mae_gru, rmse_gru = evaluate(predictions_gru, y_test_orig)

# Виведення результатів у таблиці
results_df = pd.DataFrame({
    "Модель": ["Simple RNN", "LSTM", "GRU"],
    "MAE": [mae_rnn, mae_lstm, mae_gru],
    "RMSE": [rmse_rnn, rmse_lstm, rmse_gru]
})
print(results_df)

# 5. Візуалізація результатів
print("\nВізуалізація порівняльних результатів...")
plt.figure(figsize=(18, 8))

# Відображаємо тільки тестову частину реальних даних для чистоти графіка
plt.plot(y_test_orig, color='blue', label='Реальний трафік', linewidth=2)

# Прогнози
plt.plot(predictions_rnn, color='orange', linestyle='--', label=f'Прогноз RNN (RMSE: {rmse_rnn:.2f})')
plt.plot(predictions_lstm, color='red', linestyle='--', label=f'Прогноз LSTM (RMSE: {rmse_lstm:.2f})')
plt.plot(predictions_gru, color='green', linestyle='--', label=f'Прогноз GRU (RMSE: {rmse_gru:.2f})')

plt.title('Порівняння моделей RNN, LSTM та GRU для прогнозу трафіку')
plt.xlabel('Час (години, тестова вибірка)')
plt.ylabel('Обсяг трафіку')
plt.legend()
plt.grid(True)
plt.show()