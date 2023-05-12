import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf

netflix_df = pd.read_csv('netflix_titles.csv')


def recommend_movies(user_input, netflix_df, num_recommendations):
   
    tfidf_vectorizer = tfidf()

 
    tfidf_matrix = np.array(tfidf_vectorizer.fit_transform(
        netflix_df['description'].fillna('')).todense())
    
    item_inner_products = np.dot(tfidf_matrix, tfidf_matrix.T)

   
    indices = pd.Series(
        netflix_df.index, index=netflix_df['title']).drop_duplicates()
    idx = indices[user_input]


    item_norms = np.linalg.norm(tfidf_matrix, axis=1)
    user_norm = item_norms[idx]

    similarities = item_inner_products[idx] / (user_norm * item_norms)

    top_indices = np.argsort(similarities)[::-1][:num_recommendations]

    recommended_movies = list(netflix_df['title'].iloc[top_indices])

 
    output_str = f"Recommended movies for '{user_input}':\n\n"
    for i, movie_title in enumerate(recommended_movies):
        output_str += f"{i+1}. {movie_title}\n"

    return output_str


user_input = "Stranger Things"
num_recommendations = 5
output_str = recommend_movies(user_input, netflix_df, num_recommendations)
print(output_str)
