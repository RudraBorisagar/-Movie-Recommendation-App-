"""
utils.py
--------
Shared utility functions and constants used across the movie recommendation
system.  These helpers deal with common text-normalisation, safe parsing,
date extraction, and display-formatting tasks.

Keeping them here (rather than scattered across modules) makes them easy to
test in isolation and prevents code duplication.
"""

import ast
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POSTER_URL = "/assets/default_poster.png"
"""
Fallback poster image URL used when a movie has no poster available in TMDB.
The front-end (Streamlit app) reads this constant to display a placeholder
image rather than a broken link.
"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def format_movie_title(title):
    """
    Clean and normalise a raw movie title string.

    Normalisation steps:
    - Strip leading/trailing whitespace.
    - Collapse multiple internal spaces into one.
    - Title-case the result for consistent display.

    This is used both when querying the recommendation engine (to match the
    stored titles) and when displaying results to the user.

    Parameters
    ----------
    title : str
        Raw movie title, possibly with extra whitespace or inconsistent casing.

    Returns
    -------
    str
        Cleaned, title-cased movie title.  Returns an empty string if the
        input is not a string or is empty.

    Examples
    --------
    >>> format_movie_title("  the dark   knight  ")
    'The Dark Knight'
    """
    try:
        if not isinstance(title, str):
            return ""
        # Collapse runs of whitespace, then strip edges
        cleaned = re.sub(r"\s+", " ", title).strip()
        return cleaned.title()
    except Exception as exc:
        logger.warning("format_movie_title failed for input '%s': %s", title, exc)
        return ""


def safe_literal_eval(val):
    """
    Safely parse a stringified Python literal (list, dict, etc.) using
    ``ast.literal_eval``.

    TMDB CSV columns (genres, keywords, cast, crew) store data as Python-
    notation string representations of lists of dicts.  ``json.loads`` rejects
    these because they use single quotes.  ``ast.literal_eval`` handles them
    correctly and is safe (unlike ``eval``) because it does not execute
    arbitrary code.

    If parsing fails for any reason (malformed string, NaN, non-string), the
    function returns an empty list so the calling code does not need to handle
    exceptions.

    Parameters
    ----------
    val : str or any
        Value to parse, typically a string from a CSV cell.

    Returns
    -------
    list
        Parsed Python object (expected to be a list), or ``[]`` on failure.

    Examples
    --------
    >>> safe_literal_eval("[{'id': 28, 'name': 'Action'}]")
    [{'id': 28, 'name': 'Action'}]
    >>> safe_literal_eval("bad input")
    []
    """
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def get_release_year(date_str):
    """
    Extract the four-digit year from a date string.

    TMDB stores release dates in ISO 8601 format (``YYYY-MM-DD``).  This
    function extracts the year component for display, filtering, and EDA
    purposes.  It uses a regex rather than ``datetime.strptime`` so that
    malformed or partial dates (e.g. '1995', '1995-00') are handled
    gracefully.

    Parameters
    ----------
    date_str : str or any
        Date string, typically in ``YYYY-MM-DD`` format.

    Returns
    -------
    int or None
        Four-digit year as an integer, or ``None`` if extraction fails.

    Examples
    --------
    >>> get_release_year('2009-12-18')
    2009
    >>> get_release_year('unknown')
    None
    """
    try:
        if not isinstance(date_str, str):
            return None
        match = re.search(r"\b(\d{4})\b", date_str)
        if match:
            return int(match.group(1))
        return None
    except Exception as exc:
        logger.warning("get_release_year failed for '%s': %s", date_str, exc)
        return None


def truncate_text(text, max_chars=200):
    """
    Truncate a string to at most ``max_chars`` characters, appending an
    ellipsis if the text was shortened.

    Used to display movie overviews in recommendation result cards without
    overwhelming the UI.  The ellipsis signals to the user that more text
    is available.

    Parameters
    ----------
    text : str
        Input text, typically a movie overview.
    max_chars : int, optional
        Maximum number of characters to retain (not counting the ellipsis).
        Default is 200.

    Returns
    -------
    str
        Original text if it fits within ``max_chars``, otherwise the first
        ``max_chars`` characters followed by ``'…'``.  Returns an empty string
        for non-string inputs.

    Examples
    --------
    >>> truncate_text("A long movie overview …", max_chars=10)
    'A long mov…'
    """
    try:
        if not isinstance(text, str):
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "…"
    except Exception as exc:
        logger.warning("truncate_text failed: %s", exc)
        return ""


def format_rating(rating):
    """
    Format a numeric rating as a string with one decimal place.

    TMDB vote averages are floats in [0, 10].  This helper standardises their
    display across the UI (e.g. ``7.3`` instead of ``7.300000000000001``).

    Parameters
    ----------
    rating : float or int or str
        Numeric rating value.

    Returns
    -------
    str
        Rating formatted to one decimal place (e.g. ``'7.3'``), or ``'N/A'``
        if the input cannot be converted to a float.

    Examples
    --------
    >>> format_rating(7.3456)
    '7.3'
    >>> format_rating('bad')
    'N/A'
    """
    try:
        return f"{float(rating):.1f}"
    except (ValueError, TypeError):
        return "N/A"
