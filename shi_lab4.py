import numpy as np
import matplotlib.pyplot as plt

print("Підготовка даних")
# Еталонний патерн для якісного продукту
prototype_quality = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
# Еталонний патерн для бракованого продукту (наприклад, з дефектом у центрі)
prototype_defective = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])

prototypes = {
    "Якісний": prototype_quality,
    "Бракований": prototype_defective
}
prototype_list = list(prototypes.values())
prototype_names = list(prototypes.keys())

print(f"Еталон 'Якісний':    {prototypes['Якісний']}")
print(f"Еталон 'Бракований': {prototypes['Бракований']}")

# Згенеруємо тестові дані, включаючи ідеальні зразки та зразки з шумом.
def add_noise(pattern, noise_level):
    """Інвертує 'noise_level' відсоток бітів у патерні."""
    noisy_pattern = pattern.copy()
    num_bits_to_flip = int(len(pattern) * noise_level)
    indices_to_flip = np.random.choice(len(pattern), num_bits_to_flip, replace=False)
    noisy_pattern[indices_to_flip] *= -1
    return noisy_pattern

# Створення тестового набору даних
test_data = {
    "Ідеальний якісний": prototypes["Якісний"],
    "Ідеальний бракований": prototypes["Бракований"],
    "Якісний з малим шумом (10%)": add_noise(prototypes["Якісний"], 0.1),
    "Бракований з малим шумом (10%)": add_noise(prototypes["Бракований"], 0.1),
    "Якісний з великим шумом (30%)": add_noise(prototypes["Якісний"], 0.3),
    "Невідомий патерн": np.array([-1, -1, -1, 1, 1, 1, 1, -1, -1, -1])
}
print("\nТестові дані згенеровано.\n")

# --- Мережа Хемінга ---
def hamming_network_classify(input_signal, prototypes):
    """
    Класифікує вхідний сигнал, знаходячи найближчий еталонний патерн.
    Для біполярних векторів (-1, 1) максимальна схожість відповідає
    максимальному скалярному добутку.
    """
    scores = [np.dot(input_signal, p) for p in prototypes]
    closest_pattern_idx = np.argmax(scores)
    return closest_pattern_idx, scores

# --- Мережа MAXNET ---
def maxnet(initial_activations, epsilon=0.15, max_iterations=100):
    """
    Імітує конкуренцію між нейронами, залишаючи активним лише один з них.
    """
    activations = np.array(initial_activations, dtype=float)
    num_neurons = len(activations)

    for _ in range(max_iterations):
        if np.sum(activations > 0) <= 1:
            break
        prev_activations = activations.copy()
        for i in range(num_neurons):
            inhibition_sum = np.sum(prev_activations) - prev_activations[i]
            activations[i] = max(0, prev_activations[i] - epsilon * inhibition_sum)
    return np.argmax(activations) if np.any(activations > 0) else -1

# --- Мережа Mexican Hat ---
def mexican_hat_filter(input_signal, radius=2):
    """
    Застосовує 1D фільтр "мексиканський капелюх" для виявлення локальних максимумів (дефектів).
    """
    size = len(input_signal)
    output_signal = np.zeros(size)
    for i in range(size):
        excitation_sum = 0
        for j in range(max(0, i - radius), min(size, i + radius + 1)):
            excitation_sum += input_signal[j]
        inhibition_sum = np.sum(input_signal) - excitation_sum
        output_signal[i] = excitation_sum - inhibition_sum
    return output_signal

print("Функції для мереж Хемінга, MAXNET та Mexican Hat реалізовано.\n")

# --- Тестування мережі Хемінга (Класифікація) ---
print("\n--- 1. Тест мережі Хемінга (Класифікація 'Якісний'/'Бракований') ---")

correct_classifications = 0
total_tests = 0

for name, data in test_data.items():
    predicted_idx, _ = hamming_network_classify(data, prototype_list)
    predicted_class = prototype_names[predicted_idx]

    # Визначаємо очікуваний (правильний) клас для поточного тесту
    expected_class = None
    if "Якісний" in name:
        expected_class = "Якісний"
    elif "Бракований" in name:
        expected_class = "Бракований"

    # Перевіряємо правильність тільки для тих даних, де результат відомий
    is_correct = False
    # Якщо для тесту визначено очікуваний клас
    if expected_class is not None:
        total_tests += 1
        if predicted_class == expected_class:
            is_correct = True
            correct_classifications += 1

    # Виводимо результат, N/A (не застосовується) для невідомих патернів
    print(f"Вхід: '{name}' -> Результат: '{predicted_class}' (Правильність: {is_correct if expected_class is not None else 'N/A'})")

# Розраховуємо точність
accuracy = (correct_classifications / total_tests) * 100 if total_tests > 0 else 0
print(f"\nТочність класифікації мережі Хемінга: {accuracy:.2f}%")



# --- Тестування мережі MAXNET (Вибір домінуючого сигналу) ---
print("\n--- 2. Тест мережі MAXNET (Вибір домінуючого сигналу) ---")
ambiguous_scores = [1.2, 1.0]
print(f"Початкові активації (схожість): {ambiguous_scores}")
winner_index = maxnet(ambiguous_scores)
winner_class = prototype_names[winner_index]
print(f"MAXNET визначив переможця: '{winner_class}' (індекс {winner_index})")

# --- Тестування мережі Mexican Hat (Виявлення дефектів) ---
print("\n--- 3. Тест мережі Mexican Hat (Виявлення локальних дефектів) ---")
surface_signal = np.array([0, 0, 0, 0, 0, 5, 6, 5, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0])
print(f"Вхідний сигнал з поверхні: {surface_signal}")
filtered_signal = mexican_hat_filter(surface_signal, radius=1)
print(f"Сигнал після фільтрації:   {[round(x, 1) for x in filtered_signal]}")

plt.figure(figsize=(10, 5))
plt.title("Виявлення дефектів за допомогою мережі Mexican Hat")
plt.plot(surface_signal, 'bo-', label='Початковий сигнал (поверхня)')
plt.plot(filtered_signal, 'ro-', label='Вихід мережі (виявлені дефекти)')
plt.xlabel("Позиція на поверхні")
plt.ylabel("Амплітуда сигналу")
plt.legend()
plt.grid(True)
plt.show()

