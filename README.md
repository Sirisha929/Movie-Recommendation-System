# 🎬 Movie Recommendation System

This is a **Content-Based Movie Recommendation System** built using Python and Scikit-learn. It suggests movies based on the similarity of genres and titles using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 📌 Features

- Recommends similar movies based on a selected movie
- Uses **TF-IDF** to vectorize movie genres and titles
- Computes **cosine similarity** to rank and recommend movies
- Simple command-line interface for input/output
- Lightweight, fast, and easy to understand

---

## 🧠 How It Works

1. Loads a dataset of movies with `movieId`, `title`, and `genres`.
2. Combines the `title` and `genres` for each movie.
3. Converts the combined text into a TF-IDF matrix.
4. Calculates cosine similarity between all movies.
5. When the user inputs a movie name, it returns the **top 10 most similar movies**.

---

## 🗂️ Dataset

The dataset is a small version of the [MovieLens dataset](https://grouplens.org/datasets/movielens/).  
File: `movies.csv`

| movieId | title                         | genres                            |
|---------|-------------------------------|-----------------------------------|
| 1       | Toy Story (1995)              | Adventure\|Animation\|Comedy...   |
| 2       | Jumanji (1995)                | Adventure\|Fantasy                |
| ...     | ...                           | ...                               |

---

## 🚀 Getting Started

### ✅ Prerequisites
Make sure you have Python 3 installed with the following libraries:
```bash
pip install pandas scikit-learn
