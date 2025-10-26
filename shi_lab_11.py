import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Налаштування ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'datasets/real_and_fake_face'
CLASSES = ['training_real', 'training_fake']

# --- Перевірка шляху ---
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ Не знайдено шлях: {DATA_DIR}")

print(f"✅ Знайдено папку з даними: {DATA_DIR}")

# --- Побудова моделі ---
base = MobileNetV2(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
base.trainable = False  # заморожуємо базову частину

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation='sigmoid')(x)
model = Model(base.input, out)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Підготовка даних ---
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASSES,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASSES,
    subset='validation'
)

# --- Навчання ---
print("\n🚀 Починаємо навчання...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE
)

# --- Збереження ---
model.save('deepfake_detector.h5')
print("✅ Модель збережено як 'deepfake_detector.h5'")

# --- Оцінка на валідації ---
val_loss, val_acc = model.evaluate(val_gen)
print(f"\n📊 Точність на валідаційних даних: {val_acc * 100:.2f}%")

# --- Графік точності і втрат ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Точність')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Втрата')
plt.legend()
plt.show()

# --- Тестування на кількох зображеннях ---
import random

# Вибираємо 6 випадкових зображень із валідаційного набору
sample_images, sample_labels = next(val_gen)
idxs = random.sample(range(len(sample_images)), 6)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(idxs):
    img = sample_images[idx]
    true_label = int(sample_labels[idx])
    pred = model.predict(img[np.newaxis, ...])[0][0]
    pred_label = 1 if pred >= 0.5 else 0

    color = 'green' if pred_label == true_label else 'red'
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Pred: {'Fake' if pred_label else 'Real'}\n({pred:.2f})", color=color)
    plt.axis('off')

plt.suptitle("🔍 Результати розпізнавання (зелений = правильно)")
plt.show()
s