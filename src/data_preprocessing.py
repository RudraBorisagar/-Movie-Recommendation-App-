"""
data_preprocessing.py
----------------------
Handles loading, cleaning, and parsing of TMDB movie dataset files.
This module is the first stage in the movie recommendation pipeline.
"""

import ast
import logging
from pathlib import Path

import pandas as pd

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data(movies_path, credits_path):
    """
    Load the TMDB movies and credits CSV files and merge them into one DataFrame.

    We load two separate CSVs because the TMDB dataset is split:
    - tmdb_5000_movies.csv contains movie metadata (overview, genres, budget, etc.)
    - tmdb_5000_credits.csv contains cast and crew information.

    We merge on 'title' (a common unique-enough key across both files) so that
    every row in the resulting DataFrame contains all the information needed to
    build content-based recommendations without further joins at later stages.

    Parameters
    ----------
    movies_path : str or Path
        Path to the tmdb_5000_movies.csv file.
    credits_path : str or Path
        Path to the tmdb_5000_credits.csv file.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing both movie metadata and credit information.

    Raises
    ------
    FileNotFoundError
        If either CSV file does not exist at the given path.
    Exception
        For any other I/O or parsing error.
    """
    try:
        movies_path = Path(movies_path)
        credits_path = Path(credits_path)

        if not movies_path.exists():
            raise FileNotFoundError(f"Movies file not found: {movies_path}")
        if not credits_path.exists():
            raise FileNotFoundError(f"Credits file not found: {credits_path}")

        logger.info("Loading movies dataset from %s", movies_path)
        movies_df = pd.read_csv(movies_path)

        logger.info("Loading credits dataset from %s", credits_path)
        credits_df = pd.read_csv(credits_path)

        # Merge on 'title' — both CSVs share this column as a natural key.
        # Using an inner join ensures we only keep movies present in both files,
        # which avoids NaN values introduced by unmatched rows.
        logger.info("Merging datasets on 'title'")
        merged_df = movies_df.merge(credits_df, on="title")

        logger.info(
            "Merge complete: %d rows, %d columns", merged_df.shape[0], merged_df.shape[1]
        )
        return merged_df

    except FileNotFoundError:
        logger.error("One or both dataset files are missing.")
        raise
    except Exception as exc:
        logger.error("Unexpected error while loading data: %s", exc)
        raise


def clean_data(df):
    """
    Clean the merged DataFrame by removing nulls, duplicates, and irrelevant columns.

    Content-based filtering relies on textual and categorical features.  Rows
    missing an 'overview' cannot contribute meaningful text features, so we drop
    them rather than impute.  Keeping only the relevant columns reduces memory
    usage and speeds up downstream processing.

    Relevant columns retained:
        movie_id, title, overview, genres, keywords, cast, crew,
        vote_average, vote_count, release_date, runtime, popularity

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged DataFrame from :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with reset index.

    Raises
    ------
    KeyError
        If expected columns are absent from the input DataFrame.
    Exception
        For any other cleaning error.
    """
    try:
        # Standardize movie_id column if present as 'id'
        if "movie_id" not in df.columns and "id" in df.columns:
            df["movie_id"] = df["id"]

        relevant_columns = [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
            "vote_average",
            "vote_count",
            "release_date",
            "runtime",
            "popularity",
        ]

        # Keep only columns that actually exist in the DataFrame (defensive)
        existing_cols = [c for c in relevant_columns if c in df.columns]
        missing_cols = set(relevant_columns) - set(existing_cols)
        if missing_cols:
            logger.warning("Missing expected columns (will be skipped): %s", missing_cols)

        df = df[existing_cols].copy()

        initial_rows = len(df)

        # Drop rows where 'overview' is null — no text, no features
        df.dropna(subset=["overview"], inplace=True)
        logger.info(
            "Dropped %d rows with null overview", initial_rows - len(df)
        )

        # Remove exact duplicate rows to avoid skewed similarity scores
        before_dedup = len(df)
        df.drop_duplicates(inplace=True)
        logger.info("Dropped %d duplicate rows", before_dedup - len(df))

        df.reset_index(drop=True, inplace=True)
        logger.info("Clean data shape: %s", df.shape)
        return df

    except KeyError as exc:
        logger.error("Column not found during cleaning: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error during data cleaning: %s", exc)
        raise


def parse_json_column(df, column):
    """
    Parse a column whose values are stringified JSON / Python literals into
    actual Python lists.

    TMDB CSVs store structured data (genres, keywords, cast, crew) as JSON
    strings, e.g. ``"[{'id': 28, 'name': 'Action'}, ...]"``.  We use
    ``ast.literal_eval`` rather than ``json.loads`` because the strings use
    single-quote Python notation, which is not valid JSON.

    A safe wrapper is applied so that any malformed values are silently
    replaced with an empty list rather than raising an exception that would
    halt the whole pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column to parse.
    column : str
        Name of the column to parse in-place.

    Returns
    -------
    pd.DataFrame
        DataFrame with the specified column converted to lists.

    Raises
    ------
    KeyError
        If ``column`` does not exist in ``df``.
    """
    try:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        def safe_parse(val):
            """Return parsed literal or empty list on failure."""
            try:
                result = ast.literal_eval(val)
                return result if isinstance(result, list) else []
            except (ValueError, SyntaxError):
                return []

        df[column] = df[column].apply(safe_parse)
        logger.info("Parsed JSON column: '%s'", column)
        return df

    except KeyError:
        logger.error("Column '%s' not found.", column)
        raise
    except Exception as exc:
        logger.error("Error parsing column '%s': %s", column, exc)
        raise


def extract_genres(df):
    """
    Extract genre name strings from the parsed 'genres' column.

    After :func:`parse_json_column`, the 'genres' column contains lists of
    dicts such as ``[{'id': 28, 'name': 'Action'}, ...]``.  We extract only
    the 'name' field because the 'id' carries no semantic information useful
    for text-based similarity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a parsed 'genres' column (list of dicts).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'genres' column replaced by a list of genre name strings.
    """
    try:
        def _extract(genres_list):
            try:
                return [item["name"] for item in genres_list if "name" in item]
            except (TypeError, KeyError):
                return []

        df["genres"] = df["genres"].apply(_extract)
        logger.info("Extracted genre names.")
        return df

    except Exception as exc:
        logger.error("Error extracting genres: %s", exc)
        raise


def extract_keywords(df):
    """
    Extract keyword name strings from the parsed 'keywords' column.

    Similar to genres, the 'keywords' column contains lists of dicts.  We
    retain only the 'name' field.  Keywords provide thematic context (e.g.
    'based on novel', 'friendship') that improves recommendation relevance
    beyond genres alone.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a parsed 'keywords' column (list of dicts).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'keywords' column replaced by a list of keyword strings.
    """
    try:
        def _extract(keywords_list):
            try:
                return [item["name"] for item in keywords_list if "name" in item]
            except (TypeError, KeyError):
                return []

        df["keywords"] = df["keywords"].apply(_extract)
        logger.info("Extracted keyword names.")
        return df

    except Exception as exc:
        logger.error("Error extracting keywords: %s", exc)
        raise


def extract_cast(df, top_n=5):
    """
    Extract the top-N cast member names from the parsed 'cast' column.

    Cast information contributes significantly to content similarity: users who
    enjoyed an Actors's previous films often enjoy their new ones.  We limit to
    ``top_n`` (default 5) to avoid over-weighting movies with large ensembles
    and to focus on the principal cast whose names are most recognisable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a parsed 'cast' column (list of dicts with 'name' key).
    top_n : int, optional
        Maximum number of cast members to retain per movie.  Default is 5.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'cast' column replaced by a list of name strings.
    """
    try:
        def _extract(cast_list):
            try:
                return [item["name"] for item in cast_list[:top_n] if "name" in item]
            except (TypeError, KeyError):
                return []

        df["cast"] = df["cast"].apply(_extract)
        logger.info("Extracted top-%d cast members.", top_n)
        return df

    except Exception as exc:
        logger.error("Error extracting cast: %s", exc)
        raise


def extract_director(df):
    """
    Extract the director's name from the parsed 'crew' column.

    A film's director has an outsized influence on its style, tone, and
    thematic concerns.  We scan the crew list for the entry whose 'job' field
    equals 'Director' and store only that name.  If multiple directors are
    listed (rare), we take the first.  If none is found, we return an empty list
    so downstream processing handles it uniformly.

    The result is stored in a new column 'director' as a one-element list (or
    empty list) to be consistent with the other feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a parsed 'crew' column (list of dicts).

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'director' column (list of str).
    """
    try:
        def _extract(crew_list):
            try:
                for member in crew_list:
                    if member.get("job") == "Director":
                        return [member["name"]]
                return []
            except (TypeError, KeyError):
                return []

        df["director"] = df["crew"].apply(_extract)
        logger.info("Extracted director names.")
        return df

    except Exception as exc:
        logger.error("Error extracting directors: %s", exc)
        raise


def preprocess_all(movies_path, credits_path):
    """
    Orchestrate the full preprocessing pipeline and return a clean DataFrame.

    This function is the single entry point for data preparation.  It calls
    each preprocessing step in the correct order:

    1. :func:`load_data`         — load & merge CSVs
    2. :func:`clean_data`        — nulls, dedup, column selection
    3. :func:`parse_json_column` — parse genres, keywords, cast, crew
    4. :func:`extract_genres`    — extract genre name strings
    5. :func:`extract_keywords`  — extract keyword name strings
    6. :func:`extract_cast`      — extract top-5 cast names
    7. :func:`extract_director`  — extract director name

    Parameters
    ----------
    movies_path : str or Path
        Path to tmdb_5000_movies.csv.
    credits_path : str or Path
        Path to tmdb_5000_credits.csv.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed DataFrame ready for feature engineering.

    Raises
    ------
    Exception
        Propagates any exception raised by the sub-steps.
    """
    try:
        logger.info("=== Starting full preprocessing pipeline ===")

        df = load_data(movies_path, credits_path)
        df = clean_data(df)

        # Parse all stringified JSON columns before extracting named fields
        for col in ["genres", "keywords", "cast", "crew"]:
            df = parse_json_column(df, col)

        df = extract_genres(df)
        df = extract_keywords(df)
        df = extract_cast(df, top_n=5)
        df = extract_director(df)

        logger.info("=== Preprocessing complete: %d movies ready ===", len(df))
        return df

    except Exception as exc:
        logger.error("Pipeline failed during preprocessing: %s", exc)
        raise
