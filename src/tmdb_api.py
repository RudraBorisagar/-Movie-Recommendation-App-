"""TMDB API Integration Module

Handles all interactions with The Movie Database (TMDB) API.
Never hardcode API keys - read from environment or Streamlit secrets.
"""

import os
import logging
import requests

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

# Module-level constants
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
DEFAULT_POSTER = "https://via.placeholder.com/500x750?text=No+Poster+Available"

# Module-level session and logger
SESSION = requests.Session()
logger = logging.getLogger(__name__)


def get_api_key() -> str:
    """Retrieve the TMDB API key from Streamlit secrets or environment variables.

    Attempts to read the API key in the following order:
    1. Streamlit secrets (``st.secrets['tmdb_api_key']``)
    2. Environment variable ``TMDB_API_KEY``

    Returns:
        str or None: The API key string if found, or ``None`` if neither source
        provides a value. Callers are responsible for handling the ``None`` case.
    """
    if _STREAMLIT_AVAILABLE:
        try:
            key = st.secrets.get("tmdb_api_key")
            if key:
                return key
        except Exception:
            pass  # Streamlit secrets not configured; fall through to env var

    return os.environ.get("TMDB_API_KEY")


def fetch_movie_details(movie_id: int, api_key: str = None) -> dict:
    """Fetch detailed information about a specific movie from TMDB.

    Sends a GET request to ``/movie/{movie_id}`` and returns a normalised
    dictionary of the most useful fields.

    Args:
        movie_id (int): The TMDB movie identifier.
        api_key (str, optional): TMDB API key. Resolved via :func:`get_api_key`
            when not supplied.

    Returns:
        dict: A dictionary containing:
            - ``title`` (str)
            - ``overview`` (str)
            - ``release_date`` (str)
            - ``runtime`` (int, minutes)
            - ``vote_average`` (float)
            - ``vote_count`` (int)
            - ``genres`` (list[str]) – genre names only
            - ``poster_path`` (str)
            - ``backdrop_path`` (str)
            - ``popularity`` (float)
            - ``status`` (str)
        Returns an empty dict on any error.
    """
    if api_key is None:
        api_key = get_api_key()

    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {"api_key": api_key, "language": "en-US"}
        response = SESSION.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        return {
            "title": data.get("title", ""),
            "overview": data.get("overview", ""),
            "release_date": data.get("release_date", ""),
            "runtime": data.get("runtime", 0),
            "vote_average": data.get("vote_average", 0.0),
            "vote_count": data.get("vote_count", 0),
            "genres": [g["name"] for g in data.get("genres", [])],
            "poster_path": data.get("poster_path", ""),
            "backdrop_path": data.get("backdrop_path", ""),
            "popularity": data.get("popularity", 0.0),
            "status": data.get("status", ""),
        }
    except Exception as exc:
        logger.warning("fetch_movie_details failed for movie_id=%s: %s", movie_id, exc)
        return {}


def get_poster_url(poster_path: str, size: str = "w500") -> str:
    """Construct a full TMDB image URL from a relative poster path.

    Args:
        poster_path (str): The relative path returned by the TMDB API
            (e.g. ``"/abc123.jpg"``). Pass ``None`` or an empty string to
            receive the default placeholder URL.
        size (str, optional): TMDB image size identifier (e.g. ``"w500"``,
            ``"original"``). Defaults to ``"w500"``.

    Returns:
        str: The fully-qualified image URL, or :data:`DEFAULT_POSTER` when
        ``poster_path`` is ``None`` or empty.
    """
    if not poster_path:
        return DEFAULT_POSTER
    return f"{IMAGE_BASE_URL}/{size}{poster_path}"


def fetch_movie_poster(movie_id: int, api_key: str = None) -> str:
    """Retrieve the poster URL for a given movie.

    Internally calls :func:`fetch_movie_details` and passes the resulting
    ``poster_path`` through :func:`get_poster_url`.

    Args:
        movie_id (int): The TMDB movie identifier.
        api_key (str, optional): TMDB API key. Resolved via :func:`get_api_key`
            when not supplied.

    Returns:
        str: Full poster URL, or :data:`DEFAULT_POSTER` if the movie could not
        be fetched or has no poster.
    """
    if api_key is None:
        api_key = get_api_key()

    try:
        details = fetch_movie_details(movie_id, api_key=api_key)
        return get_poster_url(details.get("poster_path", ""))
    except Exception as exc:
        logger.warning("fetch_movie_poster failed for movie_id=%s: %s", movie_id, exc)
        return DEFAULT_POSTER


def fetch_trending_movies(time_window: str = "week", api_key: str = None) -> list:
    """Fetch the list of trending movies from TMDB.

    Sends a GET request to ``/trending/movie/{time_window}`` and returns up to
    20 results with normalised fields.

    Args:
        time_window (str, optional): Either ``"day"`` or ``"week"``. Defaults
            to ``"week"``.
        api_key (str, optional): TMDB API key. Resolved via :func:`get_api_key`
            when not supplied.

    Returns:
        list[dict]: Each dict contains:
            - ``id`` (int)
            - ``title`` (str)
            - ``poster_path`` (str) – full URL
            - ``vote_average`` (float)
            - ``release_date`` (str)
            - ``overview`` (str)
        Returns an empty list on any error.
    """
    if api_key is None:
        api_key = get_api_key()

    try:
        url = f"{TMDB_BASE_URL}/trending/movie/{time_window}"
        params = {"api_key": api_key}
        response = SESSION.get(url, params=params, timeout=5)
        response.raise_for_status()
        results = response.json().get("results", [])[:20]

        return [
            {
                "id": movie.get("id"),
                "title": movie.get("title", ""),
                "poster_path": get_poster_url(movie.get("poster_path", "")),
                "vote_average": movie.get("vote_average", 0.0),
                "release_date": movie.get("release_date", ""),
                "overview": movie.get("overview", ""),
            }
            for movie in results
        ]
    except Exception as exc:
        logger.warning("fetch_trending_movies failed (time_window=%s): %s", time_window, exc)
        return []


def fetch_popular_movies(page: int = 1, api_key: str = None) -> list:
    """Fetch the list of currently popular movies from TMDB.

    Sends a GET request to ``/movie/popular`` and returns results with
    normalised fields (same schema as :func:`fetch_trending_movies`).

    Args:
        page (int, optional): Page number for pagination. Defaults to ``1``.
        api_key (str, optional): TMDB API key. Resolved via :func:`get_api_key`
            when not supplied.

    Returns:
        list[dict]: Each dict contains:
            - ``id`` (int)
            - ``title`` (str)
            - ``poster_path`` (str) – full URL
            - ``vote_average`` (float)
            - ``release_date`` (str)
            - ``overview`` (str)
        Returns an empty list on any error.
    """
    if api_key is None:
        api_key = get_api_key()

    try:
        url = f"{TMDB_BASE_URL}/movie/popular"
        params = {"api_key": api_key, "language": "en-US", "page": page}
        response = SESSION.get(url, params=params, timeout=5)
        response.raise_for_status()
        results = response.json().get("results", [])

        return [
            {
                "id": movie.get("id"),
                "title": movie.get("title", ""),
                "poster_path": get_poster_url(movie.get("poster_path", "")),
                "vote_average": movie.get("vote_average", 0.0),
                "release_date": movie.get("release_date", ""),
                "overview": movie.get("overview", ""),
            }
            for movie in results
        ]
    except Exception as exc:
        logger.warning("fetch_popular_movies failed (page=%s): %s", page, exc)
        return []


def search_movies(query: str, api_key: str = None) -> list:
    """Search for movies on TMDB by title or keyword.

    Sends a GET request to ``/search/movie`` and returns up to 10 results.

    Args:
        query (str): The search string to send to TMDB.
        api_key (str, optional): TMDB API key. Resolved via :func:`get_api_key`
            when not supplied.

    Returns:
        list[dict]: Up to 10 dicts, each containing:
            - ``id`` (int)
            - ``title`` (str)
            - ``poster_path`` (str) – full URL
            - ``vote_average`` (float)
            - ``release_date`` (str)
            - ``overview`` (str)
        Returns an empty list on any error.
    """
    if api_key is None:
        api_key = get_api_key()

    try:
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {"api_key": api_key, "language": "en-US", "query": query}
        response = SESSION.get(url, params=params, timeout=5)
        response.raise_for_status()
        results = response.json().get("results", [])[:10]

        return [
            {
                "id": movie.get("id"),
                "title": movie.get("title", ""),
                "poster_path": get_poster_url(movie.get("poster_path", "")),
                "vote_average": movie.get("vote_average", 0.0),
                "release_date": movie.get("release_date", ""),
                "overview": movie.get("overview", ""),
            }
            for movie in results
        ]
    except Exception as exc:
        logger.warning("search_movies failed (query=%r): %s", query, exc)
        return []


def fetch_multiple_movie_details(movie_ids: list, api_key: str = None) -> list:
    """Fetch detailed information for multiple movies by their TMDB IDs.

    Iterates over ``movie_ids``, calls :func:`fetch_movie_details` for each,
    skips any that return an empty dict (i.e. failed), and augments successful
    results with a ``poster_url`` field constructed via :func:`get_poster_url`.

    Args:
        movie_ids (list[int]): A list of TMDB movie identifiers.
        api_key (str, optional): TMDB API key. Resolved via :func:`get_api_key`
            when not supplied. The same key is reused for every request.

    Returns:
        list[dict]: Each dict is the output of :func:`fetch_movie_details` with
        an additional ``poster_url`` (str) field. Movies whose fetch failed are
        omitted from the result.
    """
    if api_key is None:
        api_key = get_api_key()

    movies = []
    for movie_id in movie_ids:
        try:
            details = fetch_movie_details(movie_id, api_key=api_key)
            if not details:
                continue
            details["poster_url"] = get_poster_url(details.get("poster_path", ""))
            movies.append(details)
        except Exception as exc:
            logger.warning(
                "fetch_multiple_movie_details skipping movie_id=%s: %s", movie_id, exc
            )

    return movies
