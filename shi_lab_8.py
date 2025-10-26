import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time


# 1. Генерація синтетичних даних
def generate_medical_data(n_samples=1000):
    np.random.seed(42)
    # Симптоми: температура, кашель, головний біль, слабкість, нудота
    symptoms = np.random.randint(0, 2, (n_samples, 5))
    diagnoses = []
    for i in range(n_samples):
        temp, cough, headache, weakness, nausea = symptoms[i]
        if temp > 0 and cough > 0 and headache > 0:
            diagnoses.append('Грип')
        elif cough > 0 and weakness > 0:
            diagnoses.append('Застуда')
        elif temp > 0 and nausea > 0 and headache > 0:
            diagnoses.append('Кишкова інфекція')
        elif headache > 0 and weakness > 0:
            diagnoses.append('Мігрень')
        else:
            diagnoses.append('Здоровий')
    return symptoms, diagnoses


X, y = generate_medical_data(1000)

# 2. Кодування міток
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Нормалізація даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(float))
X_test_scaled = scaler.transform(X_test.astype(float))

print(f"Розмірність даних (тренувальна): {X_train_scaled.shape}")
print(f"Кількість класів: {len(np.unique(y_encoded))}")

def create_model(optimizer='adam'):
    """Створення моделі нейронної мережі з заданим оптимізатором"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(5,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(5, activation='softmax')  # 5 класів діагнозів
    ])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate(optimizer, optimizer_name, X_train, y_train, X_test, y_test):
    """Навчання та оцінка моделі з заданим оптимізатором"""
    print(f"\n--- Навчання з {optimizer_name} ---")
    model = create_model(optimizer)

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0  # Вимикаємо вивід логів навчання
    )
    end_time = time.time()

    print(f"Час навчання: {end_time - start_time:.2f} секунд")

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Точність на тестових даних: {test_accuracy:.4f}")
    print(f"Функція втрат на тестових даних: {test_loss:.4f}")
    return history, test_accuracy, test_loss, (end_time - start_time)


# Визначення оптимізаторів для порівняння
optimizers = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'RMSProp': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': keras.optimizers.Adam(learning_rate=0.001)
}

# Словник для зберігання результатів
results = {}

# Навчання моделей з різними оптимізаторами
for opt_name, optimizer in optimizers.items():
    history, accuracy, loss, train_time = train_and_evaluate(
        optimizer, opt_name, X_train_scaled, y_train, X_test_scaled, y_test
    )
    results[opt_name] = {
        'history': history,
        'accuracy': accuracy,
        'loss': loss,
        'time': train_time
    }

# Створення графіків для порівняння
plt.figure(figsize=(15, 12))

# 1. Графік функції втрат
plt.subplot(2, 2, 1)
for opt_name, result in results.items():
    plt.plot(result['history'].history['loss'], label=f'{opt_name} Train')
    plt.plot(result['history'].history['val_loss'], label=f'{opt_name} Val', linestyle='--')
plt.title('Функція втрат під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.legend()
plt.grid(True)

# 2. Графік точності
plt.subplot(2, 2, 2)
for opt_name, result in results.items():
    plt.plot(result['history'].history['accuracy'], label=f'{opt_name} Train')
    plt.plot(result['history'].history['val_accuracy'], label=f'{opt_name} Val', linestyle='--')
plt.title('Точність під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.grid(True)

# 3. Порівняння фінальної точності
plt.subplot(2, 2, 3)
accuracies = [result['accuracy'] for result in results.values()]
optimizer_names = list(results.keys())
bars = plt.bar(optimizer_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Фінальна точність на тестових даних')
plt.ylabel('Точність')
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{accuracy:.4f}', ha='center', va='bottom')

# 4. Порівняння швидкості збіжності (епохи до досягнення 80% точності)
plt.subplot(2, 2, 4)
convergence_epochs = {}
for opt_name, result in results.items():
    val_accuracy = result['history'].history['val_accuracy']
    convergence_epoch = next((i for i, acc in enumerate(val_accuracy) if acc > 0.8), 100)  # 100 - max епох
    convergence_epochs[opt_name] = convergence_epoch
bars = plt.bar(convergence_epochs.keys(), convergence_epochs.values(),
               color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Швидкість збіжності (епохи до 80% точності)')
plt.ylabel('Кількість епох')
for bar, epochs in zip(bars, convergence_epochs.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f'{epochs}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 5. Детальний аналіз стабільності
print("\n" + "=" * 50)
print("ДЕТАЛЬНИЙ АНАЛІЗ РЕЗУЛЬТАТІВ")
print("=" * 50)
for opt_name, result in results.items():
    val_loss = result['history'].history['val_loss']
    stability = np.std(val_loss[-20:])  # Стабільність = ст. відхилення за останні 20 епох
    final_val_accuracy = result['history'].history['val_accuracy'][-1]
    final_train_accuracy = result['history'].history['accuracy'][-1]

    print(f"\n{opt_name}:")
    print(f" - Фінальна точність (val): {final_val_accuracy:.4f}")
    print(f" - Різниця train/val (ознака перенавчання): {abs(final_train_accuracy - final_val_accuracy):.4f}")
    print(f" - Стабільність (ст. відх. val_loss): {stability:.6f}")
    print(f" - Загальний час навчання: {result['time']:.2f} сек.")