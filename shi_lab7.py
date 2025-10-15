# -*- coding: utf-8 -*-
"""
Повний код для Лабораторної роботи №7:
Застосування ART для розпізнавання рукописних цифр.
"""

# --- 1. Імпорт необхідних бібліотек ---
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

print("Бібліотеки успішно імпортовано.")

# --- 2. Завантаження та підготовка даних ---
print("\nЕтап 1: Завантаження та підготовка даних...")

# Завантаження вбудованого датасету рукописних цифр
digits = load_digits()
X_images = digits.images  # Зображення у форматі 8x8
y_labels = digits.target  # Справжні мітки (0-9)

# Перетворення зображень у бінарний формат (чорно-білий)
# Пікселі зі значенням > 7 стають 1 (чорний), інші - 0 (білий)
X_binary = (X_images > 7).astype(int)
print("Дані перетворено у бінарний формат.")

# Візуалізація кількох прикладів для перевірки
print("Приклади бінаризованих зображень:")
plt.figure(figsize=(8, 3))
plt.suptitle("Приклади бінаризованих цифр", fontsize=14)
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_binary[i], cmap='gray')
    plt.title(f'Цифра: {y_labels[i]}')
    plt.axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- 3. Реалізація алгоритму ART1 ---
print("\nЕтап 2: Реалізація класу для мережі ART1...")


class ART1:
    """
    Клас, що реалізує логіку нейронної мережі ART1.
    """

    def __init__(self, n_input, vigilance=0.75):
        self.n_input = n_input
        self.vigilance = vigilance  # Поріг пильності (rho)
        self.weights = np.array([])  # Ваги, що є прототипами класів
        self.n_categories = 0  # Лічильник створених класів

    def learn(self, input_pattern):
        # Перетворення вхідного зображення в одновимірний вектор
        input_vector = input_pattern.flatten()

        # Якщо це перший зразок, створюємо першу категорію
        if self.n_categories == 0:
            self.weights = input_vector.copy().reshape(1, -1)
            self.n_categories = 1
            return 0  # Повертаємо індекс створеної категорії

        # Розрахунок подібності вхідного вектора до всіх існуючих категорій
        similarities = np.sum(self.weights * input_vector, axis=1)

        # Сортування категорій за спаданням подібності
        sorted_indices = np.argsort(similarities)[::-1]

        for winner_idx in sorted_indices:
            # Перевірка умови резонансу (vigilance test)
            prototype = self.weights[winner_idx]
            intersection = np.sum(input_vector * prototype)

            # Якщо вектор нульовий, резонанс неможливий
            if np.sum(input_vector) == 0:
                return -1

            resonance_ratio = intersection / np.sum(input_vector)

            if resonance_ratio >= self.vigilance:
                # РЕЗОНАНС: оновлюємо ваги категорії-переможця
                self.weights[winner_idx] = input_vector * prototype
                return winner_idx  # Повертаємо індекс існуючої категорії

        # НЕМАЄ РЕЗОНАНСУ: створюємо нову категорію
        self.weights = np.vstack([self.weights, input_vector])
        self.n_categories += 1
        return self.n_categories - 1  # Повертаємо індекс нової категорії


print("Клас ART1 успішно реалізовано.")

# --- 4. Експерименти з різними параметрами пильності ---
print("\nЕтап 3: Дослідження впливу параметра пильності (ρ)...")

vigilance_values = [0.5, 0.75, 0.9]

for vigilance in vigilance_values:
    print(f"\n--- Експеримент з ρ = {vigilance} ---")

    # Створення та навчання нової мережі
    art_net = ART1(n_input=64, vigilance=vigilance)
    for i in range(len(X_binary)):  # Навчання на всіх даних
        art_net.learn(X_binary[i])

    print(f"Кількість створених класів (категорій): {art_net.n_categories}")

    # Відображення прототипів (вагових векторів) для перших 10 класів
    plt.figure(figsize=(12, 3))
    plt.suptitle(f'Прототипи класів (ρ = {vigilance})', fontsize=14)
    num_prototypes_to_show = min(10, art_net.n_categories)

    for j in range(num_prototypes_to_show):
        plt.subplot(1, num_prototypes_to_show, j + 1)
        prototype_image = art_net.weights[j].reshape(8, 8)
        plt.imshow(prototype_image, cmap='gray')
        plt.title(f'Клас {j + 1}')
        plt.axis('off')
    plt.show()

# --- 5. Аналіз стійкості до шуму ---
print("\nЕтап 4: Аналіз стійкості мережі до шуму...")


def add_noise(image, noise_level=0.1):
    """Функція для додавання шуму до бінарного зображення."""
    noisy_image = image.copy().flatten()
    n_pixels = len(noisy_image)
    n_noise = int(n_pixels * noise_level)

    # Випадково вибираємо пікселі для інверсії
    noise_indices = np.random.choice(n_pixels, n_noise, replace=False)
    noisy_image[noise_indices] = 1 - noisy_image[noise_indices]

    return noisy_image.reshape(image.shape)


# Створюємо мережу із середнім рівнем пильності
art_net_for_noise_test = ART1(n_input=64, vigilance=0.75)
# Спочатку навчаємо її на чистих даних
for image in X_binary:
    art_net_for_noise_test.learn(image)

print("Тестування на зашумлених даних:")
for noise_level in [0.0, 0.1, 0.2, 0.3]:
    correct_predictions = 0
    total_samples = len(X_binary)

    for i in range(total_samples):
        # Класифікація оригінального та зашумленого зображення
        original_category = art_net_for_noise_test.learn(X_binary[i])
        noisy_image = add_noise(X_binary[i], noise_level)
        noisy_category = art_net_for_noise_test.learn(noisy_image)

        if original_category == noisy_category:
            correct_predictions += 1

    accuracy = (correct_predictions / total_samples) * 100
    print(f"Рівень шуму {noise_level * 100:.0f}%: Стійкість класифікації {accuracy:.1f}%")

print("\nВсі етапи роботи завершено.")