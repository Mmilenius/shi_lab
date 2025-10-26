import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Список імен для всіх колонок файлу u.item
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
               'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Завантаження даних з правильними іменами колонок
movies = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.item',
    sep='|',
    names=movies_cols,
    encoding='latin-1'
)

# Створення єдиного рядка з назвами жанрів для кожного фільму
# Ми починаємо з 5-ї колонки ('unknown'), щоб захопити всі жанри
genre_columns = movies.columns[5:]
movies['genres_str'] = movies[genre_columns].apply(
    lambda row: ' '.join(genre_columns[row.astype(bool)].values),
    axis=1
)

# Векторизація жанрів за допомогою TF-IDF
# stop_words='english' тут не є критичним, але залишено для повноти
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_str'])

# Обчислення матриці косинусної схожості
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Створення індексів для швидкого пошуку фільмів за назвою
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    """
    Функція для отримання рекомендацій на основі схожості жанрів.
    """
    # Отримання індексу фільму за його назвою
    try:
        idx = indices[title]
    except KeyError:
        return "Фільм з такою назвою не знайдено."

    # Отримання пар (індекс, схожість) для всіх фільмів
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Сортування фільмів за спаданням схожості
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Вибір 10 найбыльш схожих фільмів (ігноруючи сам фільм, який є на першому місці)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Повернення назв рекомендованих фільмів
    return movies['title'].iloc[movie_indices]

# Приклад виклику для фільму 'Toy Story (1995)'
print("--- Рекомендації для фільму 'Toy Story (1995)' ---")
print(get_recommendations('Toy Story (1995)'))