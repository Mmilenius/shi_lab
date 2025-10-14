import google.generativeai as genai
import pandas as pd
import os

# Вставте ваш API ключ безпосередньо в лапки.
# Я взяв ключ із вашого скриншоту Google AI Studio.
genai.configure(api_key="AIzaSyBdF9lwYZg71njSjtOJQe5FsB_XWvXcECM")

# Створення моделі з найновішою стабільною назвою
model = genai.GenerativeModel('gemini-2.5-flash')

prompts = [
    'Переклади "The quick brown fox jumps over the lazy dog." на українську мову.',
    "Напиши функцію Python для прогнозування часових рядів за допомогою LSTM-моделі, використовуючи Keras.",
    "Хто отримав Нобелівську премію з фізики у 1900 році?",
    "Опиши детальний план подорожі з Києва до Марса на сучасному автономному автомобілі.",
    "Чи повинен автономний автомобіль у неминучій аварії рятувати пасажирів чи пішоходів, якщо вибір є обов'язковим?"
]

# --- Обробка та виведення результатів ---
print("--- Результати запитів до моделі Gemini ---")

for i, prompt in enumerate(prompts):
    try:
        # Відправляємо запит до моделі
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Помилка під час виконання запиту: {e}"

    # Виводимо запит та відповідь
    print(f"\n\n==================== ЗАПИТ {i + 1} ====================")
    print(f"ЗАПИТ:\n{prompt}")
    print("\nВІДПОВІДЬ:")
    print(answer)

print("\n\n==================================================")
print("Всі запити оброблено.")
