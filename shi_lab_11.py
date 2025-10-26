import torch
from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification
import warnings

# Ігнорувати попередження, які не впливають на результат
warnings.filterwarnings("ignore")

# --- 1. Налаштування ---
print("Завантаження моделі та процесора...")

model_name = "dima806/deepfake_vs_real_image_detection"

try:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
except Exception as e:
    print(f"Не вдалося завантажити модель: {e}")
    exit()

labels = model.config.id2label
print(f"Модель завантажена. Класи: {labels}")


# --- 2. Допоміжна функція для аналізу ---

def detect_deepfake(image_url):
    """
    Завантажує зображення з URL, обробляє його
    та повертає прогноз моделі.
    """
    try:
        # -------------------------------------------------------------------
        # ОНОВЛЕНО: Додаємо User-Agent, щоб імітувати запит браузера
        # -------------------------------------------------------------------
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        print(f"Обробка: {image_url[:50]}...")
        # Передаємо 'headers' у запит
        response = requests.get(image_url, stream=True, headers=headers)

        # Перевіряємо, чи успішний запит
        if response.status_code != 200:
            return f"Помилка: Не вдалося завантажити (Код: {response.status_code})", 0

        # Відкриваємо зображення з отриманих даних
        image = Image.open(response.raw).convert("RGB")

        # Обробка зображення для моделі
        inputs = processor(images=image, return_tensors="pt")

        # Вимкнення розрахунку градієнтів для пришвидшення
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        probability = torch.nn.functional.softmax(logits, dim=1)[0][predicted_class_idx].item()

        return labels[predicted_class_idx], probability

    except Exception as e:
        # Якщо помилка все одно виникає, виводимо її
        return f"Помилка обробки зображення: {e}", 0


# --- 3. Проведення тесту ---

print("\n--- Початок тестування ---")

# Зразок 1: Справжнє зображення
real_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Audrey_Hepburn_in_Breakfast_at_Tiffany%27s_1.jpg/800px-Audrey_Hepburn_in_Breakfast_at_Tiffany%27s_1.jpg"

print(f"\n🔍 Аналіз Зразка 1 (Справжнє фото):")
result_real, prob_real = detect_deepfake(real_image_url)
print(f"➡️ Вердикт моделі: {result_real} (Ймовірність: {prob_real * 100:.2f}%)")

# Зразок 2: Згенероване (фейкове) зображення
fake_image_url = "https://this-person-does-not-exist.com/img/avatar-gen112b0785c4906f360f0e30931d8c1c51.jpg"

print(f"\n🔍 Аналіз Зразка 2 (Згенероване фото):")
result_fake, prob_fake = detect_deepfake(fake_image_url)
print(f"➡️ Вердикт моделі: {result_fake} (Ймовірність: {prob_fake * 100:.2f}%)")

print("\n--- Тестування завершено ---")