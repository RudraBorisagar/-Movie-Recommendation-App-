"""
Train the Movie Recommendation Model

Run this script ONCE to preprocess the dataset, build features,
compute cosine similarity, and save the model files.

Usage:
    python train_model.py

Outputs:
    models/similarity.pkl  - Cosine similarity matrix
    models/movies.pkl      - Processed movies DataFrame
"""

import sys
import os

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recommendation_engine import train_and_save

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Movie Recommendation System - Model Training")
    print("=" * 60)
    print()
    print("This script will:")
    print("  1. Load the TMDB 5000 dataset from data/")
    print("  2. Preprocess and clean the data")
    print("  3. Engineer features (tags column)")
    print("  4. Vectorize text using CountVectorizer")
    print("  5. Compute cosine similarity matrix")
    print("  6. Save models to models/")
    print()
    print("Make sure you have placed these files in the data/ folder:")
    print("  - data/tmdb_5000_movies.csv")
    print("  - data/tmdb_5000_credits.csv")
    print()
    print("Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
    print()

    movies_path = os.path.join("data", "tmdb_5000_movies.csv")
    credits_path = os.path.join("data", "tmdb_5000_credits.csv")

    if not os.path.exists(movies_path):
        print(f"ERROR: Dataset file not found: {movies_path}")
        print("Please download the TMDB 5000 dataset from Kaggle and place")
        print("the CSV files in the data/ directory.")
        sys.exit(1)

    if not os.path.exists(credits_path):
        print(f"ERROR: Dataset file not found: {credits_path}")
        print("Please download the TMDB 5000 dataset from Kaggle and place")
        print("the CSV files in the data/ directory.")
        sys.exit(1)

    try:
        similarity, movies_df = train_and_save(
            movies_path=str(movies_path),
            credits_path=str(credits_path)
        )
        print()
        print("=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        print()
        print(f"  Movies in model : {len(movies_df)}")
        print(f"  Similarity shape: {similarity.shape}")
        print()
        print("Model files saved:")
        print("  - models/similarity.pkl")
        print("  - models/movies.pkl")
        print()
        print("Now run the Streamlit app:")
        print("  streamlit run app.py")
    except Exception as e:
        print()
        print(f"ERROR: Training failed — {e}")
        sys.exit(1)

