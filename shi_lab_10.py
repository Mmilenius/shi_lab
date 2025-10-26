import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import time
import pandas as pd

# --- 1. ЗАВАНТАЖЕННЯ ДАНИХ (ЗМІНЕНО) ---

# Шляхи до CSV файлів та базової директорії
# (Вони коректні, виходячи з вашого скріншоту)
train_csv_path = "datasets/Train.csv"
test_csv_path = "datasets/Test.csv"
# Базова папка, де лежать папки 'Train' і 'Test' з зображеннями
base_dir = "datasets/"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Завантажуємо CSV
try:
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
except FileNotFoundError:
    print(f"Помилка: Не знайдено CSV файли за шляхами {train_csv_path} або {test_csv_path}")
    print("Переконайтеся, що файли лежать у папці 'datasets'")
    exit()


# Keras вимагає, щоб 'y_col' (ClassId) був рядком (string) для 'categorical' режиму
train_df['ClassId'] = train_df['ClassId'].astype(str)
test_df['ClassId'] = test_df['ClassId'].astype(str)

# Генератор даних з аугментацією та нормалізацією (без змін)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Валідаційний генератор - тільки нормалізація (без змін)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Створюємо генератори з DataFrame (ЗАМІСТЬ flow_from_directory)
print("Створення генератора для навчання (flow_from_dataframe)...")
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=base_dir,  # Папка, від якої будуються шляхи в 'x_col'
    x_col='Path',        # Назва колонки зі шляхами до зображень
    y_col='ClassId',     # Назва колонки з класами
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Створення генератора для валідації (flow_from_dataframe)...")
val_gen = val_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=base_dir,
    x_col='Path',
    y_col='ClassId',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Отримання кількості класів (напр., 43 для GTSRB)
num_classes = len(train_df['ClassId'].unique())
print(f"Знайдено {num_classes} класів.")


# --- 2. ПОБУДОВА МОДЕЛІ (БЕЗ ЗМІН) ---

# Функція для побудови моделі на основі transfer learning
def build_model(base_model_fn, input_shape=(128, 128, 3)):
    # Завантажуємо базову модель з вагами ImageNet
    base_model = base_model_fn(weights='imagenet', include_top=False,
                               input_shape=input_shape)

    # Заморожуємо ваги базової моделі
    base_model.trainable = False

    # Створюємо нову модель
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Компіляція моделі
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# --- 3. НАВЧАННЯ (БЕЗ ЗМІН) ---

# --- Навчання ResNet50 ---
print("--- Навчання ResNet50 ---")
model_resnet = build_model(ResNet50, input_shape=(128, 128, 3))
start_time_res = time.time()
history_res = model_resnet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE
)
end_time_res = time.time()
print(f"Час навчання ResNet50: {end_time_res - start_time_res:.2f} сек.")

# --- Навчання MobileNetV2 ---
print("\n--- Навчання MobileNetV2 ---")
model_mobilenet = build_model(MobileNetV2, input_shape=(128, 128, 3))
start_time_mob = time.time()
history_mob = model_mobilenet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE
)
end_time_mob = time.time()
print(f"Час навчання MobileNetV2: {end_time_mob - start_time_mob:.2f} сек.")


# --- 4. ВІЗУАЛІЗАЦІЯ (БЕЗ ЗМІН) ---

# Функція для візуалізації результатів
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))

    # Графік точності
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точність (навчання)')
    plt.plot(history.history['val_accuracy'], label='Точність (валідація)')
    plt.title(f'Точність {model_name}')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.legend()

    # Графік втрат
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Втрати (навчання)')
    plt.plot(history.history['val_loss'], label='Втрати (валідація)')
    plt.title(f'Втрати {model_name}')
    plt.xlabel('Епохи')
    plt.ylabel('Втрати')
    plt.legend()

    plt.show()


# Візуалізація результатів
plot_history(history_res, "ResNet50")
plot_history(history_mob, "MobileNetV2")

# Порівняння параметрів
print(f"\nПараметри ResNet50 (включаючи базову модель): {model_resnet.count_params()}")
print(f"Параметри MobileNetV2 (включаючи базову модель): {model_mobilenet.count_params()}")