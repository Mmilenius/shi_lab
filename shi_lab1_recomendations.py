import pandas as pd
import numpy as np

ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.data', sep='\t', names=ratings_cols)

movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
               'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

data = pd.merge(ratings, movies, on='movie_id')

print("Приклад даних після об'єднання:")
print(data.head())

user_item_matrix = data.pivot_table(index='user_id', columns='title', values='rating')

print("\nРозмір матриці користувач-фільм:", user_item_matrix.shape)


print("\n--- Пошук рекомендацій для фільму 'Star Wars (1977)' ---")

starwars_ratings = user_item_matrix['Star Wars (1977)']

similar_to_starwars = user_item_matrix.corrwith(starwars_ratings)

recommendations = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
recommendations.dropna(inplace=True)

ratings_count = pd.DataFrame(data.groupby('title')['rating'].count())
ratings_count.rename(columns={'rating': 'ratings_count'}, inplace=True)

recommendations = recommendations.join(ratings_count)

top_recommendations = recommendations[recommendations['ratings_count'] > 100].sort_values('Correlation', ascending=False)

print("\nТоп-10 рекомендованих фільмів (схожих на 'Star Wars (1977)'):")
print(top_recommendations.iloc[1:11])