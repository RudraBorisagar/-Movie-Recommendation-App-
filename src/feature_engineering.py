"""
feature_engineering.py
-----------------------
Transforms preprocessed movie data into a numerical feature matrix suitable
for cosine-similarity-based content-based filtering.

Pipeline overview:
  1. Normalise names (remove whitespace so multi-word names are treated as
     single tokens by the vectoriser).
  2. Stem all tokens to conflate morphological variants (e.g. 'running' ->
     'run'), reducing the vocabulary size and improving recall.
  3. Concatenate all features into a single 'tags' string per movie.
  4. Vectorise with CountVectorizer (bag-of-words) to produce a sparse
     term-frequency matrix.
"""

import logging
from pathlib import Path

import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

try:
    from src.data_preprocessing import preprocess_all
except ImportError:
    try:
        from data_preprocessing import preprocess_all
    except ImportError:
        pass

# Download punkt tokeniser data if not already present
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Module-level stemmer instance (stateless, safe to share)
_stemmer = PorterStemmer()


def remove_spaces(lst):
    """
    Remove internal spaces from each string in a list.

    This prevents multi-word names like "Sam Worthington" or "Science Fiction"
    from being split into separate tokens ('Sam', 'Worthington') by the
    vectoriser.  After space removal they become single tokens
    ('SamWorthington', 'ScienceFiction') that correctly represent the entity.

    Parameters
    ----------
    lst : list of str
        Input list of strings, e.g. cast names or genre labels.

    Returns
    -------
    list of str
        List with all internal spaces removed from each element.

    Examples
    --------
    >>> remove_spaces(['Sam Worthington', 'Zoe Saldana'])
    ['SamWorthington', 'ZoeSaldana']
    """
    try:
        if not isinstance(lst, list):
            return []
        return [str(item).replace(" ", "") for item in lst]
    except Exception as exc:
        logger.warning("remove_spaces failed: %s", exc)
        return []


def stem_text(text):
    """
    Apply Porter stemming to every word in a whitespace-separated string.

    Stemming maps inflected forms to a common root (stem), e.g.:
        'loved', 'loving', 'loves'  ->  'love'
        'running', 'runner'         ->  'run'

    This reduces vocabulary size and prevents the model from treating
    synonymous forms as distinct tokens, improving the coverage of the
    similarity search.

    Parameters
    ----------
    text : str
        Space-separated string of tokens to stem.

    Returns
    -------
    str
        Space-joined string of stemmed tokens.

    Examples
    --------
    >>> stem_text('action adventure loved')
    'action adventur love'
    """
    try:
        if not isinstance(text, str):
            return ""
        tokens = text.split()
        stemmed = [_stemmer.stem(token) for token in tokens]
        return " ".join(stemmed)
    except Exception as exc:
        logger.warning("stem_text failed: %s", exc)
        return ""


def create_tags(df):
    """
    Combine all relevant text features into a single 'tags' string per movie.

    The 'tags' column is the input to the vectoriser.  It concatenates:
    - **overview**  : tokenised plot description (rich semantic content)
    - **genres**    : genre labels (Action, Drama, …)
    - **keywords**  : thematic keywords from TMDB editorial staff
    - **cast**      : top-5 principal cast members (space-free names)
    - **director**  : director's name (space-free)

    Each list field is first passed through :func:`remove_spaces` so that
    multi-word tokens are not fragmented.  The overview is split on whitespace
    into individual words before concatenation.

    The final 'tags' string is lowercased to ensure case-insensitive matching.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with columns: overview, genres, keywords,
        cast, director.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional 'tags' column (str).

    Raises
    ------
    KeyError
        If a required column is missing from ``df``.
    """
    try:
        required = ["overview", "genres", "keywords", "cast", "director"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in DataFrame.")

        def _build_tags(row):
            try:
                overview_tokens = str(row["overview"]).split()
                genres = remove_spaces(row["genres"])
                keywords = remove_spaces(row["keywords"])
                cast = remove_spaces(row["cast"])
                director = remove_spaces(row["director"])

                combined = overview_tokens + genres + keywords + cast + director
                return " ".join(combined).lower()
            except Exception as inner_exc:
                logger.debug("Tag build failed for row: %s", inner_exc)
                return ""

        df["tags"] = df.apply(_build_tags, axis=1)
        logger.info("Created 'tags' column for %d movies.", len(df))
        return df

    except KeyError:
        raise
    except Exception as exc:
        logger.error("Error in create_tags: %s", exc)
        raise


def vectorize_tags(df, max_features=5000):
    """
    Convert the 'tags' column into a sparse term-frequency matrix.

    We use ``CountVectorizer`` (bag-of-words) rather than TF-IDF because
    for content-based movie recommendation, raw term frequency is sufficient:
    we care whether a word appears, not how rare it is globally.  The
    ``max_features`` cap keeps the vocabulary manageable and filters very
    rare tokens that would add noise.

    English stop-words are removed to prevent common words ('the', 'a', 'is')
    from dominating similarity scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'tags' column produced by :func:`create_tags`.
    max_features : int, optional
        Maximum vocabulary size.  Default is 5000.

    Returns
    -------
    tuple
        - **vectors** (scipy.sparse.csr_matrix): Term-frequency matrix,
          shape (n_movies, max_features).
        - **vectorizer** (CountVectorizer): Fitted vectoriser instance,
          used for transforming new queries later.

    Raises
    ------
    KeyError
        If the 'tags' column is absent.
    """
    try:
        if "tags" not in df.columns:
            raise KeyError("'tags' column not found. Run create_tags() first.")

        logger.info(
            "Vectorising tags with max_features=%d, stop_words='english'", max_features
        )
        vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        vectors = vectorizer.fit_transform(df["tags"])

        logger.info(
            "Vectorisation complete: matrix shape %s", vectors.shape
        )
        return vectors, vectorizer

    except KeyError:
        raise
    except Exception as exc:
        logger.error("Error in vectorize_tags: %s", exc)
        raise


def build_feature_matrix(df):
    """
    Execute the full feature-engineering pipeline on a preprocessed DataFrame.

    Steps performed:
    1. :func:`create_tags`    — build concatenated feature string per movie
    2. Apply :func:`stem_text` on 'tags' — normalise tokens via stemming
    3. :func:`vectorize_tags` — produce sparse term-frequency matrix

    This function is the single entry point called by the recommendation
    training pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from ``data_preprocessing.preprocess_all``.

    Returns
    -------
    tuple
        - **df_with_tags** (pd.DataFrame): DataFrame with 'tags' column added.
        - **vectors** (scipy.sparse.csr_matrix): Feature matrix.
        - **vectorizer** (CountVectorizer): Fitted vectoriser.

    Raises
    ------
    Exception
        Propagates any exception raised by sub-steps.
    """
    try:
        logger.info("=== Starting feature engineering pipeline ===")

        df = create_tags(df)

        # Stem the tags to reduce vocabulary; apply row-wise
        logger.info("Applying Porter stemming to tags …")
        df["tags"] = df["tags"].apply(stem_text)

        vectors, vectorizer = vectorize_tags(df, max_features=5000)

        logger.info("=== Feature engineering complete ===")
        return df, vectors, vectorizer

    except Exception as exc:
        logger.error("Feature engineering pipeline failed: %s", exc)
        raise
