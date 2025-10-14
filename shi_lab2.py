import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 1. ПІДГОТОВКА ДАНИХ ---

# Завантаження даних з CSV файлу
try:
    data = pd.read_csv('stock_prices.csv')
    # Перевірка, чи є в файлі колонка 'Close'
    if 'Close' not in data.columns:
        raise ValueError("У файлі 'stock_prices.csv' відсутня колонка 'Close'.")
except FileNotFoundError:
    print("Помилка: Файл 'stock_prices.csv' не знайдено.")
    exit() # Завершити виконання, якщо файл не знайдено

# Використання ціни закриття ('Close') як цільового показника
prices = data['Close'].values.reshape(-1, 1)

# Масштабування даних до діапазону [0, 1]
data_scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = data_scaler.fit_transform(prices)

# Функція для створення послідовностей (метод ковзного вікна)
def create_sequences(data, window_size):
    """Формує вхідні (X) та вихідні (y) послідовності для RNN/LSTM."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 20  # Період огляду: 20 попередніх днів
X, y = create_sequences(prices_scaled, window_size)

# Розбиття даних на навчальну (80%) та тестову (20%) вибірки
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Форматування даних для моделей
# Для LSTM: (зразки, часові кроки, ознаки)
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Для MLP: (зразки, ознаки)
X_train_mlp = X_train.reshape(X_train.shape[0], -1)
X_test_mlp = X_test.reshape(X_test.shape[0], -1)

print(f"Форма навчального набору LSTM: {X_train_lstm.shape}")
print(f"Форма навчального набору MLP: {X_train_mlp.shape}")

# --- 2. ПОБУДОВА ТА НАВЧАННЯ МОДЕЛЕЙ ---

# А. Модель Багатошарового Перцептрона (MLP)
print("\nНавчання MLP моделі...")
mlp_model = Sequential()
mlp_model.add(Dense(50, activation='relu', input_dim=window_size))
mlp_model.add(Dense(1)) # Вихідний шар для прогнозу ціни
mlp_model.compile(optimizer='adam', loss='mse')
mlp_model.fit(X_train_mlp, y_train, epochs=50, batch_size=32, verbose=0)

# Прогнозування за допомогою MLP
y_pred_mlp_scaled = mlp_model.predict(X_test_mlp)
y_pred_mlp = data_scaler.inverse_transform(y_pred_mlp_scaled)

# Б. Модель Довготривалої Рекурентної Мережі (LSTM)
print("Навчання LSTM моделі...")
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(window_size, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)

# Прогнозування за допомогою LSTM
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
y_pred_lstm = data_scaler.inverse_transform(y_pred_lstm_scaled)

# --- 3. ОЦІНКА ЯКОСТІ ТА ВІЗУАЛІЗАЦІЯ ---

# Перетворення фактичних тестових значень у початковий масштаб
y_test_inv = data_scaler.inverse_transform(y_test)

# Розрахунок метрик якості
print("\n--- Метрики якості прогнозу ---")
mse_mlp = mean_squared_error(y_test_inv, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test_inv, y_pred_mlp)
mse_lstm = mean_squared_error(y_test_inv, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test_inv, y_pred_lstm)
print(f"MLP - MSE: {mse_mlp:.2f}, MAE: {mae_mlp:.2f}")
print(f"LSTM - MSE: {mse_lstm:.2f}, MAE: {mae_lstm:.2f}")

# Візуалізація результатів
plt.figure(figsize=(15, 6))
test_index = data.index[train_size + window_size:]
actual_prices = prices[train_size + window_size:]

plt.plot(test_index, actual_prices, label='Фактичні ціни', color='blue')
plt.plot(test_index, y_pred_mlp, label='Прогноз MLP', color='red', linestyle='--')
plt.plot(test_index, y_pred_lstm, label='Прогноз LSTM', color='green', linestyle='-')

plt.title(f'Прогноз біржових котирувань ({window_size} днів lookback)')
plt.xlabel('Дата / Крок часу')
plt.ylabel('Ціна закриття')
plt.legend()
plt.grid(True)
plt.show()