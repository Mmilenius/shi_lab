import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print(f"TensorFlow версія: {tf.__version__}")

# --- 1. Налаштування гіперпараметрів ---
IMAGE_SIZE = 180
BATCH_SIZE = 32
EPOCHS = 10

# --- 2. Завантаження та підготовка локального датасету ---
print("Використання локального датасету 'chest_xray'.")

base_project_dir = r"D:\workspace\shi_lab"
base_data_dir = os.path.join(base_project_dir, 'chest_xray')

train_dir = os.path.join(base_data_dir, 'train')
val_dir = os.path.join(base_data_dir, 'val')
test_dir = os.path.join(base_data_dir, 'test')

print(f"Шлях до тренувальних даних: {train_dir}")

if not os.path.exists(train_dir) or not os.path.exists(val_dir) or not os.path.exists(test_dir):
    print(f"ПОМИЛКА: Директорії 'train', 'val' або 'test' не знайдено за шляхом: {base_data_dir}")
    print("Будь ласка, переконайтеся, що папка 'chest_xray' лежить у 'D:\\workspace\\shi_lab'")
    exit()
else:
    print("Директорії 'train', 'val' та 'test' успішно знайдено.")

# Створення наборів даних
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print(f"Класи: {class_names}")

# Візуалізація прикладів
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.suptitle("Приклади зображень з тренувального набору")
plt.show()

# --- 3. Нормалізація та оптимізація ---
rescale_layer = layers.Rescaling(1. / 255)

train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y))
val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y))
test_ds = test_ds.map(lambda x, y: (rescale_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Нормалізацію та оптимізацію завантаження завершено.")

# --- 4. Створення моделі CNN ---
model = keras.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# --- 5. Компіляція та навчання ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nПочаток навчання моделі...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
)

print("Навчання завершено.")

# --- 6. Оцінка ---
print("\nОцінка моделі на тестових даних...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Загальна Втрата на тесті: {test_loss:.4f}")
print(f"Загальна Точність на тесті: {test_acc * 100:.2f}%")

# --- 6.1. Графіки ---
print("\nПобудова графіків історії навчання...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точність (Тренування)')
plt.plot(epochs_range, val_acc, label='Точність (Валідація)')
plt.legend(loc='lower right')
plt.title('Точність Навчання та Валідації')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Втрати (Тренування)')
plt.plot(epochs_range, val_loss, label='Втрати (Валідація)')
plt.legend(loc='upper right')
plt.title('Втрати Навчання та Валідації')
plt.show()

# --- 6.2. Precision, Recall, F1-score ---
print("\nОтримання прогнозів для тестового набору...")
y_true = []
for images, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true).astype(int)

y_pred_probs = model.predict(test_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\n--- Звіт про Класифікацію ---")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# --- 6.3. Confusion Matrix ---
print("\nПобудова Матриці Помилок...")
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Матриця Помилок')
plt.show()

# --- 6.4. Аналіз помилкових класифікацій ---
print("\nПошук прикладів помилкової класифікації...")

all_test_images = []
all_test_labels = []
for images, labels in test_ds.unbatch():
    all_test_images.append(images.numpy())
    all_test_labels.append(labels.numpy())

all_test_images = np.array(all_test_images)
all_test_labels = np.array(all_test_labels).astype(int).flatten()
misclassified_indices = np.where(y_pred != all_test_labels)[0]

if len(misclassified_indices) > 0:
    print(f"Знайдено {len(misclassified_indices)} помилкових класифікацій.")
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(misclassified_indices[:9]):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(all_test_images[idx])
        true_label = class_names[all_test_labels[idx]]
        pred_label = class_names[y_pred[idx]]
        plt.title(f"Реально: {true_label}\nПрогноз: {pred_label}", color='red')
        plt.axis("off")

    plt.suptitle("Приклади Помилкових Класифікацій", fontsize=16, color='red')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("Помилкових класифікацій не знайдено!")
