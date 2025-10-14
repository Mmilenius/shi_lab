import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# --- 1. Ініціалізація NLTK-об'єктів (Після гарантованого завантаження) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- 2. Завантаження обраного датасету ---
try:
    # Завантаження з типовими параметрами для цього датасету
    df = pd.read_csv(
        'spam.csv',
        encoding='latin-1',
        usecols=['v1', 'v2']
    )
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    print("✅ Дані успішно завантажено та розмічено.")
except FileNotFoundError:
    print("⚠️ Файл 'spam.csv' не знайдено. Створення симуляції даних.")
    df = pd.DataFrame({
        'label': [0, 1, 0, 1, 0],
        'text': [
            'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got the money to pay an...'
            , 'URGENT! You have won a 1 week FREE membership in our $1000 prize. CALL NOW 09061701461'
            , 'Hey man, what you up to?'
            , 'WINNER! You just won a free iPhone. Click here: http://free.com. Claim it now!'
            , 'Sorry, I\'ll call you later.'
        ]
    })


# --- 3. Передобробка текстів: очищення, токенізація, нормалізація ---
def preprocess_text(text):
    # Очищення: URL та пунктуація
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Нормалізація: нижній регістр
    text = text.lower()

    # Токенізація, видалення стоп-слів та лемматизація
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


df['cleaned_text'] = df['text'].apply(preprocess_text)
print("✅ Передобробка тексту завершена.")

# --- 4. Розділення датасету та TF-IDF ---
X_train_data, X_test_data, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"✅ Датасет розділено: Навчальний набір: {len(X_train_data)}, Тестовий набір: {len(X_test_data)}")

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train_data)
X_test_vectorized = vectorizer.transform(X_test_data)
print(f"✅ Текст векторизовано (TF-IDF). Форма навчального набору: {X_train_vectorized.shape}")

# --- 5. Збереження готового датасету у вигляді JSON ---
final_dataset = pd.DataFrame({
    'text': df['cleaned_text'],
    'label': df['label']
})

final_dataset.to_json('spam_dataset_prepared.json', orient='records', indent=4)
print("✅ Готовий датасет збережено у файл 'spam_dataset_prepared.json' у форматі JSON.")