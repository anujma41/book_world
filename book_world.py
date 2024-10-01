import pickle
import streamlit as st
import numpy as np
import os

# Set the header for the app
st.header("Book Recommendation System")

# Define the path for loading models and data
ARTIFACTS_DIR = "artifacts"  # Update this if you're organizing files differently

# Initialize variables
model = None
final_rating = None
book_pivot = None
books_name = []  # Initialize as an empty list

# Load the necessary data
try:
    model = pickle.load(open(os.path.join(ARTIFACTS_DIR, 'model.pkl'), 'rb'))
    books_name = pickle.load(open(os.path.join(ARTIFACTS_DIR, 'books_name.pkl'), 'rb'))
    final_rating = pickle.load(open(os.path.join(ARTIFACTS_DIR, 'final_rating.pkl'), 'rb'))
    book_pivot = pickle.load(open(os.path.join(ARTIFACTS_DIR, 'book_pivot.pkl'), 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Ensure books_name is iterable and a list
if isinstance(books_name, str):
    books_name = [books_name]  # Convert single string to a list
elif isinstance(books_name, np.ndarray):
    books_name = books_name.tolist()  # Convert numpy array to list

# Check if book_pivot is loaded correctly before defining books_name
if book_pivot is not None:
    # If books_name is not already set from the file, define it using book_pivot
    if not books_name:
        books_name = book_pivot.index.tolist()  # Define books_name from book_pivot index

# Ensure books_name is a list
if not isinstance(books_name, list):
    st.error("Error: books_name should be a list or numpy array of book titles.")
    st.stop()  # Stop further execution if there's an error

# Function to fetch poster URLs
def fetch_poster(suggestions):
    poster_urls = []
    for book_id in suggestions:
        book_title = book_pivot.index[book_id]
        # Get the book index in final_rating and fetch the URL
        idx = np.where(final_rating['Title'] == book_title)[0][0]
        poster_url = final_rating.iloc[idx]['Image-URL']
        poster_urls.append(poster_url)
    return poster_urls

# Function to recommend books based on selected book
def recommend_book(book_name):
    # Check if the selected book exists in the pivot table
    if book_name not in book_pivot.index:
        st.error(f"Book '{book_name}' not found in the dataset.")
        return [], []  # Return empty lists in case of failure

    # Get the book ID
    book_id = np.where(book_pivot.index == book_name)[0][0]
    # Get distances and suggestions from the model
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    # Skip the first suggestion because it's the selected book itself
    suggested_books = book_pivot.index[suggestions[0][1:]]
    # Fetch poster URLs for suggested books
    poster_urls = fetch_poster(suggestions[0][1:])

    return list(suggested_books), poster_urls

# Create a selectbox for book selection
selected_book = st.selectbox("Type or select a book:", books_name)

# Show recommendations when the button is clicked
if st.button('Show Recommendations'):
    recommended_books, poster_urls = recommend_book(selected_book)

    # Ensure the recommendations and posters are available
    if recommended_books and poster_urls:
        # Create columns to display book recommendations and their posters
        cols = st.columns(5)
        for i in range(len(recommended_books)):
            with cols[i]:
                st.text(recommended_books[i])
                st.image(poster_urls[i])
