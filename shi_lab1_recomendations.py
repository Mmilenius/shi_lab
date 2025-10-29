import pandas as pd
import numpy as np

# Завантаження даних
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.data', sep='\t', names=ratings_cols)

movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
               'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

# Об'єднання даних
data = pd.merge(ratings, movies, on='movie_id')
print("Приклад даних після об'єднання:")
print(data.head())

# Створення матриці
user_item_matrix = data.pivot_table(index='user_id', columns='title', values='rating')
print("\nРозмір матриці користувач-фільм:", user_item_matrix.shape)

# --- Змінено фільм для аналізу ---
target_movie = 'Pulp Fiction (1994)'
print(f"\n--- Пошук рекомендацій для фільму '{target_movie}' ---")
# ------------------------------------

# Перевірка, чи фільм існує в матриці
if target_movie not in user_item_matrix.columns:
    print(f"Фільм '{target_movie}' не знайдено в датасеті.")
else:
    # Обчислення кореляції
    movie_ratings = user_item_matrix[target_movie]
    # Використовуємо 'pearson' кореляцію, мінімум 50 спільних оцінок для надійності
    similar_to_movie = user_item_matrix.corrwith(movie_ratings, method='pearson', min_periods=50)

    recommendations = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    recommendations.dropna(inplace=True)

    # Додавання кількості рейтингів для фільтрації
    ratings_count = pd.DataFrame(data.groupby('title')['rating'].count())
    ratings_count.rename(columns={'rating': 'ratings_count'}, inplace=True)

    recommendations = recommendations.join(ratings_count)

    # Фільтруємо за популярністю (понад 100 оцінок) та сортуємо
    # Змінив поріг на 100 для кращої фільтрації менш відомих фільмів
    top_recommendations = recommendations[recommendations['ratings_count'] > 100].sort_values('Correlation', ascending=False)

    print(f"\nТоп-10 рекомендованих фільмів (схожих на '{target_movie}'):")
    # Виключаємо сам фільм зі списку рекомендацій
    print(top_recommendations.drop(target_movie, errors='ignore').head(10))