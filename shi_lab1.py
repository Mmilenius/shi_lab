import google.generativeai as genai
import os

# Конфігурація API-ключа
genai.configure(api_key="AIzaSyBdF9lwYZg71njSjtOJQe5FsB_XWvXcECM")

# Створення моделі
model = genai.GenerativeModel('gemini-2.5-flash')

prompts = [
    'Як би ти переклав англійський ідіоматичний вислів "Break a leg!" на українську?', # 1. Ідіома
    "Напиши функцію Python, яка використовує pandas для обчислення ковзного середнього (rolling average) для стовпця 'price' у DataFrame.", # 2. Код (інша бібліотека)
    'Хто написав відому оперу "Місячна соната"?', # 3. Питання з підступом
    'Опиши смак синього кольору.', # 4. Креатив/Галюцинація
    'Чи етично використовувати ШІ для створення повністю автономних систем озброєння?' # 5. Етика (інша дилема)
]

print("--- Результати запитів до моделі Gemini ---")
for i, prompt in enumerate(prompts):
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Помилка: {e}"

    print(f"\n==================== ЗАПИТ {i + 1} ====================")
    print(f"ЗАПИТ:\n{prompt}")
    print(f"\nВІДПОВІДЬ:\n{answer}")

print("\n==================================================")