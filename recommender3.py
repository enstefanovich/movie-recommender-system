import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

def update_reviews(new_user_reviews):
    'This function updates the csv containing user reviews to include newly reviewed movies'

    with open('user_reviews_200moviesplus.csv', 'a') as review_file:
        for movie_id, user_id_rating in new_user_reviews.items():
            review_file.write(f'{user_id_rating[0]},{movie_id},{user_id_rating[1]}\n')

def create_user_review_matrix(user_reviews, user_id):
    "Creates a matrix out of the user review data frame"

    review_matrix = user_reviews.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    if user_id not in review_matrix.index:
        raise ValueError("User ID not found in review matrix")

    return review_matrix

def find_similar_movies(movie_vector, movies_df, movies_titles, num_recommendations=5):
    'Uses KNN to identify the most similar movies to a given movie'

    # Fitting KNN
    knn = NearestNeighbors(n_neighbors=num_recommendations + 1, metric='cosine', n_jobs=-1)
    knn.fit(movies_df)
    distances, indices = knn.kneighbors(movie_vector, n_neighbors=num_recommendations + 1)

    # Returns the movie ids of the most similar movies
    similar_movies = movies_titles.iloc[indices[0][1:]].index.tolist()  
    
    return similar_movies

def find_similar_users(review_matrix, user_vector, k=25):
    'Uses KNN to identify the most similar users to a given user'

    # Fitting KNN
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
    knn.fit(review_matrix)

    distances, indices = knn.kneighbors(user_vector, n_neighbors=k + 1)
    
    # Getting the user ids of the nearest neighbors
    nearest_neighbors = [review_matrix.index[indices[0, j]] for j in range(1, k + 1)]   
    
    #Getting the distances of the nearest neighbors
    neighbor_distances = distances[0,1:k+1] 

    return nearest_neighbors, neighbor_distances


def weighted_predict_ratings(movie_id, review_matrix, neighbors, distances):
    'Predicts a movie rating for a user based on the userids of their neighbor as well as their distances'
    if movie_id not in review_matrix.columns:
        return np.nan
    
    # Getting the indices of neighbors who have rated this movie
    valid_indices = [i for i, neighbor in enumerate(neighbors) if review_matrix.loc[neighbor, movie_id] > 0]
    
    if not valid_indices:
        return np.nan
    
    # Getting the ratings and distances for valid neighbors
    ratings = np.array([review_matrix.loc[neighbors[i], movie_id] for i in valid_indices])
    valid_distances = distances[valid_indices]
    
    # Convert distances to weights (smaller distance = larger weight)
    # Add small constant to avoid division by zero
    weights = 1 / (valid_distances ** 2 + 0.001)
    
    # Calculating the weighted average rating
    weighted_avg = np.sum(ratings * weights) / np.sum(weights)

   
    return weighted_avg
    
def recommend_movies(user_id, movies_df, movie_titles, review_matrix, top_n=10):
    'Returns the movie IDs of the recommended movies'

    # Checks for the user in the review matrix
    if user_id not in review_matrix.index:
        raise ValueError("User ID not found in review matrix")  
    
    # Finds the movies the user has reviewed
    user_ratings = review_matrix.loc[user_id]
    reviewed_movies = user_ratings.loc[user_ratings > 1]
    reviewed_movies = reviewed_movies.index     

    # Filtering for movies the user likes
    favorite_movies = user_ratings[user_ratings >= 4]
    if len(favorite_movies) == 0:
        return "Rate some movies first"
    
    # Creating a vector representation of the users average favorite movies
    movie_vectors = movies_df.loc[favorite_movies.index]
    composite_vector = movie_vectors.mean(axis=0)

    composite_vector = composite_vector.values.reshape(1,-1)
    user_ratings = user_ratings.to_frame().T
    
    # Finding the nearest users and movies
    similar_users, user_distances = find_similar_users(review_matrix, user_ratings) 
    similar_movies = find_similar_movies(composite_vector, movies_df, movie_titles, 40)

    # Excluding movies the user has already seen and rated
    similar_movies = [movie for movie in similar_movies if movie not in reviewed_movies]

    ratings = {}
    for movie in similar_movies:    
        rating = weighted_predict_ratings(movie, review_matrix, similar_users, user_distances) 
        
        # If there are no similar users who have rated the movie, returns the average rating for the movie from the dataset
        if pd.isna(rating):
            ratings[movie] = review_matrix[review_matrix[movie]>0][movie].mean()  
        else:
            ratings[movie] = rating

    # Sorting the list to return the highest predicted ratings
    recommended_movies = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for movie, rating in recommended_movies:
        print(f"Movie: {movie}, Rating: {rating}")
    
    return recommended_movies





