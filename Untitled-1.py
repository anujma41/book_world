# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
import os

# Load data with error handling
books_path = "C:/Users/anujm/Downloads/BOOK RECOMMDED SYSTEM/BX-Books.csv"
users_path = "C:/Users/anujm/Downloads/BOOK RECOMMDED SYSTEM/BX-Users.csv"
ratings_path = "C:/Users/anujm/Downloads/BOOK RECOMMDED SYSTEM/BX-Book-Ratings.csv"

try:
    books = pd.read_csv(books_path, sep=";", encoding='latin1', on_bad_lines='skip')
    users = pd.read_csv(users_path, sep=";", encoding='latin1', on_bad_lines='skip')
    ratings = pd.read_csv(ratings_path, sep=";", encoding='latin1', on_bad_lines='skip')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Preview books dataset
print(books.head())

# Renaming columns for clarity
books.rename(columns={
    "Book-Title": "Title",
    "Book-Author": "Author",
    "Year-Of-Publication": "Year",
    "Publisher": "Publisher",
    "Image-URL-S": "Image-URL",
    "ISBN": "isbn"
}, inplace=True)

ratings.rename(columns={"ISBN": "isbn"}, inplace=True)

# Filter users with more than 200 ratings
user_activity = ratings['User-ID'].value_counts()
active_users = user_activity[user_activity > 200].index
filtered_ratings = ratings[ratings['User-ID'].isin(active_users)]

# Merge the ratings with book details
rating_with_books = filtered_ratings.merge(books, on="isbn", how='inner')

# Count number of ratings per book
num_rating = rating_with_books.groupby('Title')['Book-Rating'].count().reset_index()
num_rating.rename(columns={"Book-Rating": "num_of_ratings"}, inplace=True)

# Merge to get only books with more than 50 ratings
final_rating = rating_with_books.merge(num_rating, on="Title")
final_rating = final_rating[final_rating['num_of_ratings'] >= 50]
final_rating.drop_duplicates(['User-ID', 'Title'], inplace=True)

# Create pivot table
book_pivot = final_rating.pivot_table(columns='User-ID', index='Title', values='Book-Rating')
book_pivot.fillna(0, inplace=True)

# Convert the pivot table to a sparse matrix for efficient processing
book_sparse = csr_matrix(book_pivot)

# Build the Nearest Neighbors model
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# Save model and data using pickle for reuse
os.makedirs('artifacts', exist_ok=True)
with open('artifacts/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('artifacts/books_name.pkl', 'wb') as books_name_file:
    pickle.dump(book_pivot.index.tolist(), books_name_file)
with open('artifacts/book_pivot.pkl', 'wb') as book_pivot_file:
    pickle.dump(book_pivot, book_pivot_file)

# Recommendation function
def recommend_book(book_name):
    # Check if the book exists
    if book_name not in book_pivot.index:
        print(f"Book '{book_name}' not found in the dataset.")
        return
    
    # Find the book index and get recommendations
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    # Print recommended books
    print(f"Books similar to '{book_name}':")
    for i in range(1, len(suggestions[0])):  # Skip the first as it's the same book
        print(book_pivot.index[suggestions[0][i]])

# Example usage: recommending books similar to "The Notebook"
selected_book = "The Notebook"
recommend_book(selected_book)
