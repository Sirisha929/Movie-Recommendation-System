# Movie Recommendation System using Content-Based Filtering

# Step 1: Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization
from sklearn.metrics.pairwise import cosine_similarity  # For measuring similarity

# Step 2: Load movie dataset (make sure movies.csv is in same folder)
# This dataset should have columns: movieId, title, genres
movies = pd.read_csv("movies.csv")

# Step 3: Check for missing data in genres column and fill with empty string
movies['genres'] = movies['genres'].fillna('')

# Step 4: Combine the 'title' and 'genres' into a single column for content similarity
movies['combined'] = movies['title'] + " " + movies['genres']

# Step 5: Create a TF-IDF Vectorizer object (removes stop words like "the", "a", etc.)
tfidf = TfidfVectorizer(stop_words='english')

# Step 6: Convert the combined column into a TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movies['combined'])

# Step 7: Compute cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 8: Create a series that maps movie titles to their index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Step 9: Define the recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices.get(title)

    # If movie not found, return message
    if idx is None:
        return f"Movie '{title}' not found in the dataset."

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (excluding itself)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movie titles
    return movies['title'].iloc[movie_indices].tolist()

# Step 10: Example usage - Get recommendations
movie_name = input("Enter a movie name: ")
recommended_movies = get_recommendations(movie_name)

# Step 11: Print recommendations
print("\nRecommended Movies:")
if isinstance(recommended_movies, list):
    for i, movie in enumerate(recommended_movies, 1):
        print(f"{i}. {movie}")
else:
    print(recommended_movies)
