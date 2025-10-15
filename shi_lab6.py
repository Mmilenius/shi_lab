# -*- coding: utf-8 -*-
"""
Повний та виправлений код для Лабораторної роботи №6:
SOM для сегментації клієнтів у маркетингу.
"""

# --- 1. Імпорт необхідних бібліотек ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
import matplotlib.pyplot as plt

print("Бібліотеки успішно імпортовано.")

# --- 2. Завантаження та підготовка даних ---
print("\nЕтап 1: Завантаження та підготовка даних...")

# Спроба завантажити CSV-файл.
try:
    data = pd.read_csv('mall_customers.csv')
    print("Файл 'mall_customers.csv' успішно завантажено.")
except FileNotFoundError:
    print("Файл 'mall_customers.csv' не знайдено. Створюю демонстраційний набір даних...")
    data_dict = {
        'CustomerID': range(1, 201),
        'Gender': np.random.choice(['Male', 'Female'], 200),
        'Age': np.random.randint(18, 71, 200),
        'Annual Income (k$)': np.random.randint(15, 138, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    }
    data = pd.DataFrame(data_dict)
    print("Демонстраційний набір даних створено.")

# ✨ ВИПРАВЛЕННЯ: Перейменування стовпців для зручності та уникнення помилок.
# Це вирішує проблему KeyError.
data.rename(columns={
    'Genre': 'Gender',
    'Annual Income (k$)': 'AnnualIncome',
    'Spending Score (1-100)': 'SpendingScore'
}, inplace=True)
print("Стовпці перейменовано для зручності.")

# Вибір ознак для аналізу за новими, зручними назвами
features = ['Age', 'AnnualIncome', 'SpendingScore']
X = data[features].values

# Нормалізація даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Розмірність підготовлених даних: {X_scaled.shape}")
print("Дані успішно нормалізовано.")

# --- 3. Побудова та навчання SOM ---
print("\nЕтап 2: Побудова та навчання моделі SOM...")

# Визначення розмірів карти
MAP_X_DIM = 10
MAP_Y_DIM = 10
INPUT_LEN = X_scaled.shape[1]

# Ініціалізація та навчання моделі SOM
som = MiniSom(
    x=MAP_X_DIM,
    y=MAP_Y_DIM,
    input_len=INPUT_LEN,
    sigma=1.5,
    learning_rate=0.5,
    random_seed=42  # Для відтворюваності результатів
)

# Ініціалізація ваг та навчання
som.random_weights_init(X_scaled)
print("Навчання моделі SOM...")
som.train_random(data=X_scaled, num_iteration=1000)
print("Навчання завершено.")

# --- 4. Візуалізація та аналіз результатів ---
print("\nЕтап 3: Візуалізація результатів...")

# 4.1. Візуалізація U-Matrix (Карта відстаней)
plt.figure(figsize=(10, 9))
plt.title('Карта відстаней (U-Matrix)', fontsize=16)
plt.pcolor(som.distance_map().T, cmap='viridis_r')
plt.colorbar(label='Відстань між нейронами')
plt.xticks(np.arange(MAP_X_DIM + 1))
plt.yticks(np.arange(MAP_Y_DIM + 1))
plt.grid()
plt.show()

# 4.2. Візуалізація компонентних карт
plt.figure(figsize=(20, 6))
plt.suptitle('Компонентні карти для кожної ознаки', fontsize=18)
for i, feature_name in enumerate(features):
    plt.subplot(1, 3, i + 1)
    plt.title(f'Ознака: {feature_name}')
    plt.pcolor(som.get_weights()[:, :, i].T, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(np.arange(MAP_X_DIM + 1))
    plt.yticks(np.arange(MAP_Y_DIM + 1))
    plt.grid()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 4.3. Візуалізація розподілу клієнтів на карті
plt.figure(figsize=(12, 11))
plt.title('Розподіл клієнтів на карті SOM', fontsize=16)
plt.pcolor(som.distance_map().T, cmap='viridis_r', alpha=0.7)
plt.colorbar(label='Відстань (межі кластерів)')

for i, x_vec in enumerate(X_scaled):
    winner_node = som.winner(x_vec)
    plt.plot(
        winner_node[0] + 0.5, winner_node[1] + 0.5,
        'o',
        markerfacecolor='None',
        markeredgecolor='crimson',
        markersize=10,
        markeredgewidth=1.5
    )
plt.xticks(np.arange(MAP_X_DIM + 1))
plt.yticks(np.arange(MAP_Y_DIM + 1))
plt.grid()
plt.show()

print("\nВсі етапи роботи завершено.")