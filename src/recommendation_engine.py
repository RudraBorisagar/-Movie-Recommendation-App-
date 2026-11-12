"""
recommendation_engine.py
-------------------------
Core engine for training, persisting, loading, and querying a content-based
movie recommendation model.

The model consists of:
  - A cosine-similarity matrix computed from the bag-of-words feature vectors.
  - A movies DataFrame that maps matrix row indices to movie metadata.

Both artefacts are serialised with ``pickle`` for fast re-use without
retraining.
"""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    # When called as part of the src package (from app.py via src.recommendation_engine)
    from src.data_preprocessing import preprocess_all
    from src.feature_engineering import build_feature_matrix
except ImportError:
    # When run directly as a script (python src/recommendation_engine.py)
    from data_preprocessing import preprocess_all
    from feature_engineering import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_similarity(vectors):
    """
    Compute the pairwise cosine-similarity matrix from a feature vector matrix.

    Cosine similarity is chosen over Euclidean distance because it is invariant
    to document length: a movie with a longer overview would otherwise appear
    artificially dissimilar to a shorter one even if they share many themes.

    The result is an (n_movies × n_movies) matrix where entry [i, j] is the
    cosine similarity between movie i and movie j.  The diagonal is always 1.0
    (a movie is perfectly similar to itself).

    Parameters
    ----------
    vectors : scipy.sparse.csr_matrix or np.ndarray
        Term-frequency matrix of shape (n_movies, n_features) produced by
        ``feature_engineering.vectorize_tags``.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_movies, n_movies), dtype float32.

    Raises
    ------
    ValueError
        If ``vectors`` is empty or has an unexpected shape.
    Exception
        For any other computation error.
    """
    try:
        if vectors is None or vectors.shape[0] == 0:
            raise ValueError("Feature vector matrix is empty.")

        logger.info(
            "Computing cosine similarity for %d movies …", vectors.shape[0]
        )
        similarity = cosine_similarity(vectors)
        logger.info("Similarity matrix shape: %s", similarity.shape)
        return similarity

    except ValueError:
        raise
    except Exception as exc:
        logger.error("Error computing similarity: %s", exc)
        raise


def save_model(similarity, movies_df, path="models/"):
    """
    Persist the similarity matrix and movies DataFrame to disk as pickle files.

    Two separate files are saved:
    - ``similarity.pkl`` : the (n × n) numpy similarity matrix.
    - ``movies.pkl``     : the movies DataFrame (with 'tags' column).

    Separating them allows the DataFrame to be loaded independently for
    metadata lookups without loading the (potentially large) similarity matrix.

    Parameters
    ----------
    similarity : np.ndarray
        Cosine-similarity matrix from :func:`compute_similarity`.
    movies_df : pd.DataFrame
        DataFrame with movie metadata and 'tags' column.
    path : str or Path, optional
        Directory in which to save the pickle files.  Created automatically
        if it does not exist.  Default is ``'models/'``.

    Raises
    ------
    OSError
        If the directory cannot be created or files cannot be written.
    Exception
        For any other serialisation error.
    """
    try:
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)

        similarity_path = model_dir / "similarity.pkl"
        movies_path = model_dir / "movies.pkl"

        logger.info("Saving similarity matrix to %s", similarity_path)
        with open(similarity_path, "wb") as f:
            pickle.dump(similarity, f)

        logger.info("Saving movies DataFrame to %s", movies_path)
        with open(movies_path, "wb") as f:
            pickle.dump(movies_df, f)

        logger.info("Model artefacts saved successfully.")

    except OSError as exc:
        logger.error("I/O error while saving model: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error while saving model: %s", exc)
        raise


def load_model(path="models/"):
    """
    Load the similarity matrix and movies DataFrame from pickle files.

    This function is used at inference time to avoid re-training the model on
    every request.  The loaded objects are ready for use with
    :func:`get_recommendations`.

    Parameters
    ----------
    path : str or Path, optional
        Directory containing ``similarity.pkl`` and ``movies.pkl``.
        Default is ``'models/'``.

    Returns
    -------
    tuple
        - **similarity** (np.ndarray): Cosine-similarity matrix.
        - **movies_df** (pd.DataFrame): Movies metadata DataFrame.

    Raises
    ------
    FileNotFoundError
        If either pickle file is missing.
    Exception
        For any deserialisation error.
    """
    try:
        model_dir = Path(path)
        similarity_path = model_dir / "similarity.pkl"
        movies_path = model_dir / "movies.pkl"

        if not similarity_path.exists():
            raise FileNotFoundError(f"Similarity file not found: {similarity_path}")
        if not movies_path.exists():
            raise FileNotFoundError(f"Movies file not found: {movies_path}")

        logger.info("Loading similarity matrix from %s", similarity_path)
        with open(similarity_path, "rb") as f:
            similarity = pickle.load(f)

        logger.info("Loading movies DataFrame from %s", movies_path)
        with open(movies_path, "rb") as f:
            movies_df = pickle.load(f)

        logger.info(
            "Model loaded: %d movies, similarity shape %s",
            len(movies_df),
            similarity.shape,
        )
        return similarity, movies_df

    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.error("Error loading model: %s", exc)
        raise


def get_recommendations(movie_title, movies_df, similarity, top_n=10):
    """
    Return the top-N most similar movies for a given title.

    The function:
    1. Locates the row index of ``movie_title`` in ``movies_df``.
    2. Retrieves that row's similarity scores from the matrix.
    3. Sorts scores descending and returns the top-N (excluding itself).

    If the title is not found, an empty list is returned and a warning is
    logged rather than raising an exception, allowing the caller to decide
    how to handle the case.

    Parameters
    ----------
    movie_title : str
        Exact movie title to look up (case-insensitive match attempted).
    movies_df : pd.DataFrame
        DataFrame with at least 'title' and 'movie_id' columns.
    similarity : np.ndarray
        Cosine-similarity matrix aligned with ``movies_df`` rows.
    top_n : int, optional
        Number of recommendations to return.  Default is 10.

    Returns
    -------
    list of dict
        Each dict has keys:
        - ``movie_id`` (int)   : TMDB movie ID.
        - ``title`` (str)      : Movie title.
        - ``index`` (int)      : Row index in ``movies_df``.
        Returns an empty list if the movie is not found.

    Raises
    ------
    Exception
        For unexpected errors during lookup or sorting.
    """
    try:
        # Case-insensitive title search
        title_lower = movie_title.strip().lower()
        matched = movies_df[movies_df["title"].str.lower() == title_lower]

        if matched.empty:
            logger.warning("Movie '%s' not found in dataset.", movie_title)
            return []

        # Use the first match if there are duplicates
        movie_index = matched.index[0]
        logger.info(
            "Found '%s' at index %d. Computing top-%d recommendations …",
            movie_title,
            movie_index,
            top_n,
        )

        # Get similarity scores for this movie vs all others
        scores = list(enumerate(similarity[movie_index]))

        # Sort by score descending, skip the first entry (the movie itself)
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        top_scores = scores_sorted[1: top_n + 1]  # exclude self at position 0

        recommendations = []
        for idx, score in top_scores:
            row = movies_df.iloc[idx]
            recommendations.append(
                {
                    "movie_id": int(row.get("movie_id", -1)),
                    "title": str(row["title"]),
                    "index": int(idx),
                    "similarity_score": round(float(score), 4),
                }
            )

        logger.info("Recommendations generated: %d results.", len(recommendations))
        return recommendations

    except Exception as exc:
        logger.error("Error generating recommendations for '%s': %s", movie_title, exc)
        raise


def train_and_save(
    movies_path="data/tmdb_5000_movies.csv",
    credits_path="data/tmdb_5000_credits.csv",
    model_path="models/",
):
    """
    End-to-end training pipeline: preprocess → feature-engineer → compute
    similarity → save model artefacts.

    This is the single command to (re-)train the recommendation engine from
    raw CSV files.  It calls:
    1. ``data_preprocessing.preprocess_all`` to load and clean the data.
    2. ``feature_engineering.build_feature_matrix`` to build the feature vectors.
    3. :func:`compute_similarity` to build the similarity matrix.
    4. :func:`save_model` to persist the artefacts.

    Parameters
    ----------
    movies_path : str or Path, optional
        Path to tmdb_5000_movies.csv relative to the project root.
    credits_path : str or Path, optional
        Path to tmdb_5000_credits.csv relative to the project root.
    model_path : str or Path, optional
        Directory in which to save the trained model artefacts.

    Returns
    -------
    tuple
        - **similarity** (np.ndarray): Trained similarity matrix.
        - **movies_df** (pd.DataFrame): Movies DataFrame with tags.

    Raises
    ------
    Exception
        Propagates any exception from sub-steps.
    """
    try:
        logger.info("=== Starting full training pipeline ===")

        # Resolve paths relative to this file's directory (project-root agnostic)
        base_dir = Path(__file__).resolve().parent.parent
        movies_path = base_dir / movies_path
        credits_path = base_dir / credits_path
        model_path = base_dir / model_path

        # Step 1: Data preprocessing
        df = preprocess_all(movies_path, credits_path)

        # Step 2: Feature engineering
        df_with_tags, vectors, _ = build_feature_matrix(df)

        # Step 3: Similarity computation
        similarity = compute_similarity(vectors)

        # Step 4: Persist model
        save_model(similarity, df_with_tags, path=model_path)

        logger.info("=== Training pipeline complete ===")
        return similarity, df_with_tags

    except Exception as exc:
        logger.error("Training pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    similarity_matrix, movies = train_and_save()
    print(f"\nModel trained on {len(movies)} movies.")
    print("Sample recommendations for 'Avatar':")
    recs = get_recommendations("Avatar", movies, similarity_matrix, top_n=5)
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec['title']} (score: {rec['similarity_score']})")
