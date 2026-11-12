"""
AI-Powered Movie Recommendation System
Main Streamlit Application

A content-based movie recommendation engine using cosine similarity
on TMDB 5000 dataset features (genres, keywords, cast, director, overview).
"""

import sys
import os

# Ensure src/ imports work
sys.path.insert(0, '.')

import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pickle
import pathlib
import time

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# Optional src imports (graceful degradation)
# ─────────────────────────────────────────────
try:
    from src.recommendation_engine import get_recommendations
    RECO_ENGINE_AVAILABLE = True
except ImportError:
    RECO_ENGINE_AVAILABLE = False

try:
    from src.tmdb_api import (
        fetch_movie_details,
        fetch_trending_movies,
        get_poster_url,
        fetch_movie_poster,
        fetch_multiple_movie_details,
    )
    TMDB_MODULE_AVAILABLE = True
except ImportError:
    TMDB_MODULE_AVAILABLE = False

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODELS_DIR = pathlib.Path("models")
SIMILARITY_PATH = MODELS_DIR / "similarity.pkl"
MOVIES_PATH = MODELS_DIR / "movies.pkl"
DEFAULT_POSTER = "https://via.placeholder.com/300x450/1a1a2e/f5c518?text=No+Poster"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ─────────────────────────────────────────────
# CUSTOM CSS – Dark Cinematic Theme
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111111 0%, #1a1a2e 100%);
        border-right: 1px solid #2a2a3e;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #cccccc !important;
        font-size: 15px;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #f5c518 !important;
    }

    /* ── Hero title ── */
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f5c518, #ff8c00, #f5c518);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #aaaaaa;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .hero-desc {
        font-size: 0.95rem;
        color: #888888;
        text-align: center;
        max-width: 700px;
        margin: 0 auto 2rem auto;
    }

    /* ── Movie Card ── */
    .movie-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }

    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(245,197,24,0.25);
    }

    .movie-title-card {
        font-weight: 700;
        font-size: 0.88rem;
        color: #f0f0f0;
        margin-top: 8px;
        line-height: 1.3;
        min-height: 2.6rem;
    }

    .rating-badge {
        display: inline-block;
        background: #f5c518;
        color: #0a0a0a;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 7px;
        border-radius: 4px;
        margin-top: 4px;
    }

    .genre-tag {
        display: inline-block;
        background: #2a2a4a;
        color: #bbbbff;
        font-size: 0.7rem;
        padding: 2px 6px;
        border-radius: 10px;
        margin: 2px 2px 0 0;
    }

    /* ── Stats card ── */
    .stats-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .stats-card h3 {
        color: #f5c518;
        margin-bottom: 12px;
    }

    .stats-row {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 10px;
    }

    .stat-item {
        text-align: center;
    }

    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f5c518;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #888888;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f5c518;
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #f5c518;
        padding-left: 12px;
    }

    /* ── Tech badge ── */
    .tech-badge {
        display: inline-block;
        background: #16213e;
        border: 1px solid #0f3460;
        color: #e94560;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px;
    }

    /* ── Gold Button override ── */
    div.stButton > button {
        background: linear-gradient(90deg, #f5c518, #ff8c00);
        color: #0a0a0a;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        transition: opacity 0.2s;
        width: 100%;
    }

    div.stButton > button:hover {
        opacity: 0.85;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: #1a1a2e !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2a4e !important;
        border-radius: 8px !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        color: #aaaaaa !important;
        font-size: 0.85rem !important;
    }

    /* ── Horizontal divider ── */
    hr {
        border-color: #2a2a3e;
    }

    /* ── Poster image radius ── */
    .stImage > img {
        border-radius: 8px !important;
    }

    /* ── Info / warning boxes ── */
    .stAlert {
        background: #1a1a2e !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
    }

    /* ── Plotly chart background ── */
    .js-plotly-plot {
        background: transparent !important;
    }

    /* Sidebar footer */
    .sidebar-footer {
        font-size: 0.72rem;
        color: #555555;
        text-align: center;
        margin-top: 2rem;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load pre-trained similarity matrix and movies DataFrame from pickle files.

    Returns:
        tuple: (similarity_matrix, movies_df) if successful,
               (None, None) if files are not found.
    """
    if not SIMILARITY_PATH.exists() or not MOVIES_PATH.exists():
        return None, None
    try:
        with open(SIMILARITY_PATH, "rb") as f:
            similarity = pickle.load(f)
        with open(MOVIES_PATH, "rb") as f:
            movies_df = pickle.load(f)
        return similarity, movies_df
    except Exception as exc:
        st.error(f"Error loading model files: {exc}")
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def get_trending_cached(api_key: str):
    """
    Fetch trending movies from TMDB API, cached for 1 hour.

    Args:
        api_key: TMDB API key string.

    Returns:
        list[dict]: Trending movies data or empty list on failure.
    """
    if not TMDB_MODULE_AVAILABLE or not api_key:
        return []
    try:
        return fetch_trending_movies(api_key)
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_details_cached(movie_id: int, api_key: str):
    """
    Fetch individual movie details from TMDB, cached for 24 hours.

    Args:
        movie_id: TMDB movie id.
        api_key:  TMDB API key string.

    Returns:
        dict: Movie details or empty dict on failure.
    """
    if not TMDB_MODULE_AVAILABLE or not api_key or not movie_id:
        return {}
    try:
        return fetch_movie_details(movie_id, api_key)
    except Exception:
        return {}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_api_key() -> str:
    """Return TMDB API key from Streamlit secrets or environment variable."""
    try:
        return st.secrets["tmdb_api_key"]
    except Exception:
        return os.environ.get("TMDB_API_KEY", "")


def poster_url(path: str | None) -> str:
    """Build a full TMDB poster URL or return the placeholder."""
    if path:
        return f"{TMDB_IMAGE_BASE}{path}"
    return DEFAULT_POSTER


def safe_genres(genres, max_n: int = 2) -> list[str]:
    """Extract up to *max_n* genre names from various input formats."""
    if isinstance(genres, list):
        result = []
        for g in genres[:max_n]:
            if isinstance(g, dict):
                result.append(g.get("name", ""))
            elif isinstance(g, str):
                result.append(g)
        return [x for x in result if x]
    if isinstance(genres, str) and genres:
        return [genres]
    return []


def safe_year(date_str) -> str:
    """Extract the 4-digit year from a date string."""
    try:
        return str(date_str)[:4]
    except Exception:
        return "N/A"


def render_movie_card(title: str, poster: str, rating=None, year=None,
                      genres=None, overview: str = "", key_prefix: str = "card"):
    """Render a styled movie card using HTML + st.image + st.expander."""
    st.image(poster, use_column_width=True)
    st.markdown(f'<div class="movie-title-card">{title}</div>', unsafe_allow_html=True)

    meta_parts = []
    if rating:
        meta_parts.append(f'<span class="rating-badge">⭐ {rating:.1f}</span>' if isinstance(rating, float) else f'<span class="rating-badge">⭐ {rating}</span>')
    if year:
        meta_parts.append(f'<span style="color:#888;font-size:0.78rem">📅 {year}</span>')
    if meta_parts:
        st.markdown(" &nbsp;".join(meta_parts), unsafe_allow_html=True)

    if genres:
        tags_html = "".join(f'<span class="genre-tag">🎭 {g}</span>' for g in genres[:2])
        st.markdown(tags_html, unsafe_allow_html=True)

    if overview:
        with st.expander("📖 Overview", expanded=False):
            st.write(overview[:300] + ("…" if len(overview) > 300 else ""))


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(movies_df) -> str:
    """Draw the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;font-size:3rem;margin-bottom:0.25rem">🎬</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="text-align:center;font-weight:700;font-size:1.1rem;'
            'color:#f5c518;margin-bottom:1.5rem">AI Movie Recommender</div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            ["🏠 Home", "🎯 Recommendations", "ℹ️ About", "🧠 ML Explained"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Dataset info
        total = len(movies_df) if movies_df is not None else 0
        st.markdown(
            f"""
            <div style="font-size:0.82rem;color:#aaaaaa;line-height:2">
            📦 <b style="color:#f5c518">Dataset</b><br>
            &nbsp;&nbsp;🎥 {total:,} movies loaded<br>
            &nbsp;&nbsp;🗃️ TMDB 5000 Dataset<br>
            &nbsp;&nbsp;🤖 Content-Based Filtering<br>
            &nbsp;&nbsp;📐 Cosine Similarity
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            '<a href="https://github.com/your-username/movie-recommendation-system" '
            'target="_blank" style="color:#f5c518;text-decoration:none;">'
            '🔗 GitHub Repository</a>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-footer">'
            'Built with ❤️ using Streamlit<br>'
            '© 2024 AI Movie Recommender<br>'
            'Student Portfolio Project'
            '</div>',
            unsafe_allow_html=True,
        )

    return page


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
def page_home(similarity, movies_df, api_key: str):
    """Render the Home page with hero section, movie selector and trending grid."""

    # ── Hero ──
    st.markdown('<div class="hero-title">🎬 AI-Powered Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Discover movies you\'ll love — powered by machine learning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-desc">'
        'This app uses <b>Content-Based Filtering</b> and <b>Cosine Similarity</b> '
        'to recommend movies similar to the one you pick, based on genres, keywords, '
        'cast, director and plot overview.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Model not found warning ──
    if similarity is None or movies_df is None:
        st.warning(
            "⚠️ **Model files not found.**  \n"
            "Run the training script first:\n"
            "```bash\n"
            "cd movie-recommendation-system\n"
            "python src/recommendation_engine.py\n"
            "```\n"
            "This will generate `models/similarity.pkl` and `models/movies.pkl`."
        )
        st.stop()

    # ── Selector + Stats ──
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-header">🎥 Choose a Movie</div>', unsafe_allow_html=True)
        all_titles = sorted(movies_df["title"].dropna().tolist())
        default_idx = all_titles.index("Avatar") if "Avatar" in all_titles else 0

        selected_movie = st.selectbox(
            "Type to search…",
            options=all_titles,
            index=default_idx,
            help="Start typing to filter the list of 4 807 movies.",
        )
        st.session_state["selected_movie"] = selected_movie

        if st.button("🎯 Get Recommendations", key="get_reco_btn"):
            st.session_state["run_reco"] = True
            st.session_state["reco_movie"] = selected_movie

    with col_right:
        st.markdown(
            f"""
            <div class="stats-card">
                <h3>📊 Quick Stats</h3>
                <div class="stats-row">
                    <div class="stat-item">
                        <div class="stat-number">{len(movies_df):,}</div>
                        <div class="stat-label">Movies</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">5,000</div>
                        <div class="stat-label">Feature Dims</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">10</div>
                        <div class="stat-label">Recs / Query</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">~98%</div>
                        <div class="stat-label">Coverage</div>
                    </div>
                </div>
                <br>
                <div style="color:#888;font-size:0.8rem">
                    Algorithm: <b style="color:#f5c518">Cosine Similarity</b><br>
                    Vectorizer: <b style="color:#f5c518">CountVectorizer</b><br>
                    Dataset: <b style="color:#f5c518">TMDB 5000</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Trending Movies ──
    st.markdown("---")
    st.markdown('<div class="section-header">🔥 Trending Movies</div>', unsafe_allow_html=True)

    with st.spinner("Loading trending movies…"):
        trending = get_trending_cached(api_key)

    if not trending:
        if not api_key:
            st.info(
                "ℹ️ **No TMDB API key found.**  \n"
                "Add your key to `.streamlit/secrets.toml`:\n"
                "```toml\ntmdb_api_key = \"your_key_here\"\n```\n"
                "Get a free key at [themoviedb.org](https://www.themoviedb.org/settings/api)."
            )
        else:
            st.info("Could not load trending movies. Check your API key or internet connection.")
    else:
        display_trending_grid(trending[:8])


def display_trending_grid(trending: list):
    """Render trending movies in a 4-column grid."""
    rows = [trending[i:i+4] for i in range(0, len(trending), 4)]
    for row in rows:
        cols = st.columns(4, gap="medium")
        for col, movie in zip(cols, row):
            with col:
                p = poster_url(movie.get("poster_path"))
                title = movie.get("title", movie.get("name", "Unknown"))
                rating = movie.get("vote_average", 0)
                date = movie.get("release_date", movie.get("first_air_date", ""))
                year = safe_year(date)
                render_movie_card(
                    title=title,
                    poster=p,
                    rating=rating,
                    year=year,
                    key_prefix=f"trend_{title[:8]}",
                )


# ─────────────────────────────────────────────
# PAGE: RECOMMENDATIONS
# ─────────────────────────────────────────────
def page_recommendations(similarity, movies_df, api_key: str):
    """Render the Recommendations page."""
    st.markdown('<div class="hero-title">🎯 Movie Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Movies similar to your pick</div>', unsafe_allow_html=True)
    st.markdown("")

    if similarity is None or movies_df is None:
        st.warning(
            "⚠️ **Model files not found.**  \n"
            "Run `python src/recommendation_engine.py` to train the model first."
        )
        return

    # ── Movie picker on this page too ──
    all_titles = sorted(movies_df["title"].dropna().tolist())
    prev = st.session_state.get("selected_movie", all_titles[0] if all_titles else "")
    default_idx = all_titles.index(prev) if prev in all_titles else 0

    col1, col2 = st.columns([4, 1])
    with col1:
        movie_choice = st.selectbox("Pick a movie", all_titles, index=default_idx)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🎯 Recommend", key="reco_page_btn")

    # Trigger from home page button too
    if st.session_state.get("run_reco") and st.session_state.get("reco_movie"):
        movie_choice = st.session_state["reco_movie"]
        run = True
        st.session_state["run_reco"] = False

    if not run:
        st.info("👆 Select a movie and click **Recommend** to see similar titles.")
        return

    # ── Run engine ──
    if not RECO_ENGINE_AVAILABLE:
        st.error(
            "❌ Recommendation engine not available.  \n"
            "Make sure `src/recommendation_engine.py` exists and dependencies are installed."
        )
        return

    with st.spinner(f"🔍 Finding similar movies to **{movie_choice}**…"):
        try:
            recs = get_recommendations(movie_choice, movies_df, similarity, top_n=10)
        except ValueError as e:
            st.error(f"❌ {e}  \nTry a different movie title.")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return

    if not recs:
        st.error(f"No recommendations found for **{movie_choice}**. Try another title.")
        return

    st.success(f"✅ Found {len(recs)} movies similar to **{movie_choice}**!")
    st.markdown('<div class="section-header">🎬 Recommended For You</div>', unsafe_allow_html=True)

    # ── 2 rows × 5 cols ──
    for row_start in range(0, min(10, len(recs)), 5):
        row_recs = recs[row_start : row_start + 5]
        cols = st.columns(5, gap="medium")
        for col, rec in zip(cols, row_recs):
            with col:
                details = {}
                if api_key:
                    details = get_movie_details_cached(rec.get("movie_id"), api_key)

                p = poster_url(details.get("poster_path"))
                title = rec.get("title", "Unknown")
                rating = details.get("vote_average")
                date = details.get("release_date", "")
                year = safe_year(date)
                genres = safe_genres(details.get("genres", []))
                overview = details.get("overview", "")

                render_movie_card(
                    title=title,
                    poster=p,
                    rating=rating,
                    year=year,
                    genres=genres,
                    overview=overview,
                    key_prefix=f"rec_{title[:8]}",
                )


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
def page_about():
    """Render the About page."""
    st.markdown('<div class="hero-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── Project description ──
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        ## 📽️ Project Description

        The **AI-Powered Movie Recommendation System** is a portfolio project that demonstrates
        how machine learning can be applied to build a real-world recommendation engine.

        It uses **Content-Based Filtering** — analysing the intrinsic properties of movies
        (genres, cast, director, keywords, overview) to find titles that are most similar
        to the one you select.

        ### 🎯 Key Features
        - 🤖 ML-powered recommendations using Cosine Similarity
        - 🌐 Live movie posters and metadata via TMDB API
        - 📊 Interactive visualisations with Plotly
        - 🔥 Trending movies from TMDB
        - 📱 Responsive dark cinematic UI
        - ⚡ Cached results for performance

        ### 📂 Project Structure
        ```
        movie-recommendation-system/
        ├── app.py                      ← You are here (Streamlit frontend)
        ├── src/
        │   ├── data_preprocessing.py   ← Data loading & cleaning
        │   ├── feature_engineering.py  ← Tag creation & vectorization
        │   ├── recommendation_engine.py← Cosine similarity & model save/load
        │   ├── tmdb_api.py             ← TMDB API integration
        │   └── utils.py                ← Shared utilities
        ├── models/
        │   ├── similarity.pkl          ← Pre-computed similarity matrix
        │   └── movies.pkl              ← Processed movies DataFrame
        ├── data/
        │   ├── tmdb_5000_movies.csv
        │   └── tmdb_5000_credits.csv
        └── notebooks/
            └── movie_recommendation_analysis.ipynb
        ```
        """)

    with col2:
        # Technologies
        st.markdown("## 🛠️ Technologies Used")
        tech_badges = [
            ("Python 3.10+", "#3776ab"),
            ("Streamlit", "#ff4b4b"),
            ("Pandas", "#150458"),
            ("NumPy", "#013243"),
            ("Scikit-learn", "#f89939"),
            ("NLTK", "#154360"),
            ("Plotly", "#3d4f7c"),
            ("TMDB API", "#01b4e4"),
            ("Pickle", "#ffdd57"),
            ("Pathlib", "#6c757d"),
        ]
        badges_html = "".join(
            f'<span class="tech-badge">{name}</span>'
            for name, _ in tech_badges
        )
        st.markdown(badges_html, unsafe_allow_html=True)

        st.markdown("---")

        # Dataset info
        st.markdown("""
        ## 📦 Dataset

        **TMDB 5000 Movie Dataset**
        - Source: [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
        - `tmdb_5000_movies.csv` — 4 807 rows, 20 columns
        - `tmdb_5000_credits.csv` — Cast & crew JSON
        - Features used: genres, keywords, cast (top 5), director, overview
        """)

        st.markdown("""
        ## 🏗️ Architecture

        ```
        Raw CSV Data
              ↓
        data_preprocessing.py  (load, clean, parse)
              ↓
        feature_engineering.py (tags, CountVectorizer)
              ↓
        recommendation_engine.py (cosine similarity → pickle)
              ↓
        app.py + tmdb_api.py  (Streamlit UI + live metadata)
        ```
        """)

    st.markdown("---")

    # ── Streamlit Secrets instructions ──
    st.markdown("""
    ## 🔑 Setting Up TMDB API Key

    1. Create a free account at [themoviedb.org](https://www.themoviedb.org/)
    2. Go to **Settings → API** and request an API key
    3. Create the file `.streamlit/secrets.toml` in the project root:

    ```toml
    # .streamlit/secrets.toml
    tmdb_api_key = "your_api_key_here"
    ```

    > **Note:** Never commit `secrets.toml` to Git. It is already listed in `.gitignore`.
    """)

    # ── Run instructions ──
    st.markdown("""
    ## 🚀 Getting Started

    ```bash
    # 1. Install dependencies
    pip install streamlit pandas numpy scikit-learn nltk plotly requests

    # 2. Download TMDB 5000 dataset and place CSVs in data/

    # 3. Train the model  (~30 seconds)
    python src/recommendation_engine.py

    # 4. Launch the app
    streamlit run app.py
    ```
    """)


# ─────────────────────────────────────────────
# PAGE: ML EXPLAINED
# ─────────────────────────────────────────────
def page_ml_explained(similarity, movies_df):
    """Render the ML Explained page with beginner-friendly explanations."""
    st.markdown('<div class="hero-title">🧠 ML Explained</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">How does the AI actually work?</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── Model stats ──
    col1, col2, col3, col4 = st.columns(4)
    total_movies = len(movies_df) if movies_df is not None else 0
    similarity_shape = similarity.shape[0] if similarity is not None else 0

    col1.metric("🎬 Movies", f"{total_movies:,}", help="Total movies in dataset")
    col2.metric("📐 Matrix Size", f"{similarity_shape:,}×{similarity_shape:,}", help="Similarity matrix dimensions")
    col3.metric("🔢 Feature Dims", "5,000", help="CountVectorizer max_features")
    col4.metric("⚡ Algorithm", "Cosine Sim", help="Similarity metric used")

    st.markdown("---")

    # ── 1. Content-Based Filtering ──
    st.markdown('<div class="section-header">1️⃣ What is Content-Based Filtering?</div>', unsafe_allow_html=True)
    st.markdown("""
    **Content-Based Filtering** recommends items similar to what you have already liked,
    based on the *properties* (content) of the items themselves — not what other users thought.

    > Think of it this way: if you loved *The Dark Knight* (a crime thriller with Christian Bale
    > directed by Christopher Nolan), the system looks for other movies that share those same
    > traits — same director, similar genre, overlapping cast.

    **Pros:**
    - Works even with no user history (cold-start safe for items)
    - Transparent — you can explain *why* a movie was recommended

    **Cons:**
    - Limited to discovering movies "more of the same"
    - Does not leverage crowd wisdom (that's Collaborative Filtering's strength)
    """)

    # ── 2. CountVectorizer ──
    st.markdown('<div class="section-header">2️⃣ What is CountVectorizer?</div>', unsafe_allow_html=True)
    st.markdown("""
    We first build a **tags string** for every movie by combining:
    - Plot overview words (stemmed)
    - Genre names
    - Top-5 cast member names (spaces removed → `ChristianBale`)
    - Director name
    - Keywords

    **CountVectorizer** then converts these tags into a numerical vector.
    Each unique word across all movies becomes a dimension, and the value is
    *how many times* that word appears in a movie's tags.
    """)

    st.code("""
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(movies_df['tags'])

# vectors.shape → (4807, 5000)
# Each row is a movie; each column is a word's count
print(vectors.shape)
    """, language="python")

    # ── 3. Cosine Similarity ──
    st.markdown('<div class="section-header">3️⃣ What is Cosine Similarity?</div>', unsafe_allow_html=True)
    st.markdown("""
    Given two movies represented as vectors **A** and **B**, Cosine Similarity measures
    the angle between them — how "aligned" their content is.
    A score of **1.0** means identical content; **0.0** means completely different.
    """)

    st.latex(r"""
    \text{cosine\_similarity}(\mathbf{A}, \mathbf{B})
    = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|}
    = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}
    """)

    st.code("""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# vectors shape: (4807, 5000)
similarity = cosine_similarity(vectors)

# similarity shape: (4807, 4807)
# similarity[i][j] → score between movie i and movie j (0.0 – 1.0)
print(similarity.shape)      # (4807, 4807)
print(similarity[0][:5])     # scores for first movie vs first 5 movies
    """, language="python")

    # ── 4. Pipeline ──
    st.markdown('<div class="section-header">4️⃣ How the Pipeline Works</div>', unsafe_allow_html=True)

    steps = [
        ("📥 Step 1: Load Data", "Read `tmdb_5000_movies.csv` + `tmdb_5000_credits.csv`, merge on title."),
        ("🧹 Step 2: Clean Data", "Drop null overviews, remove duplicates, keep relevant columns."),
        ("🔧 Step 3: Parse JSON Columns", "Convert stringified JSON (genres, keywords, cast, crew) to Python lists using `ast.literal_eval`."),
        ("🏷️ Step 4: Build Tags", "Concatenate stemmed overview words + genres + cast + director + keywords into one `tags` string per movie."),
        ("🔢 Step 5: Vectorize", "Apply `CountVectorizer(max_features=5000)` to convert tags into a 4807 × 5000 sparse matrix."),
        ("📐 Step 6: Compute Similarity", "Run `cosine_similarity(vectors)` to get a 4807 × 4807 matrix — takes ~10 s."),
        ("💾 Step 7: Save Model", "Pickle both the similarity matrix and the movies DataFrame for fast loading in Streamlit."),
        ("🎯 Step 8: Recommend", "Given a movie title, find its row index, sort that row descending, return top-10 (excluding itself)."),
    ]

    for title_step, desc in steps:
        with st.expander(title_step, expanded=False):
            st.markdown(desc)

    st.code("""
def get_recommendations(movie_title, movies_df, similarity, top_n=10):
    # 1. Find the index of the chosen movie
    idx = movies_df[movies_df['title'] == movie_title].index[0]

    # 2. Get similarity scores for that movie vs all others
    scores = list(enumerate(similarity[idx]))

    # 3. Sort by score descending
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # 4. Take top_n (skip index 0 — that's the movie itself)
    top = scores[1 : top_n + 1]

    # 5. Build result list
    recommendations = []
    for i, score in top:
        row = movies_df.iloc[i]
        recommendations.append({
            'movie_id': int(row['movie_id']),
            'title':    row['title'],
            'index':    i,
        })
    return recommendations
    """, language="python")

    # ── 5. Feature vector visualisation ──
    st.markdown('<div class="section-header">5️⃣ Visualising Feature Vectors</div>', unsafe_allow_html=True)
    st.markdown("""
    Below is a simplified 2-D illustration of how movies sit in feature space.
    Movies that are close together (small angle) will have high cosine similarity.
    """)

    # Synthetic demo chart
    np.random.seed(42)
    n = 30
    labels = [
        "Avatar", "Titanic", "Inception", "The Dark Knight",
        "Interstellar", "Iron Man", "Thor", "Captain America",
        "The Avengers", "Guardians of the Galaxy",
        "Toy Story", "Finding Nemo", "Up", "Frozen",
        "The Lion King", "Pulp Fiction", "Goodfellas",
        "The Godfather", "Forrest Gump", "Schindler's List",
        "The Matrix", "The Terminator", "Alien", "Predator",
        "Die Hard", "Speed", "Jaws", "Jurassic Park",
        "E.T.", "Home Alone",
    ]
    # Group movies into clusters so the chart is meaningful
    genres_demo = ["Sci-Fi"] * 5 + ["Superhero"] * 5 + ["Animation"] * 5 + ["Drama"] * 5 + ["Action"] * 5 + ["Family"] * 5
    x_base = [2, 2.5, 3, 2.8, 3.2,   7, 7.5, 8, 7.2, 8.1,   1, 1.2, 0.8, 1.5, 1,   5, 5.5, 4.8, 5.2, 4.5,   6, 6.5, 7, 6.2, 5.8,   2, 2.2, 1.8, 2.5, 1.5]
    y_base = [7, 7.5, 8, 6.8, 7.2,   5, 5.5, 4.8, 5.2, 4.5,   3, 2.8, 3.5, 2.5, 3.2,   7, 6.5, 7.8, 6.2, 7.5,   2, 1.8, 2.5, 1.5, 2.2,   1, 1.2, 0.8, 1.5, 0.5]

    color_map = {
        "Sci-Fi": "#00bcd4", "Superhero": "#f5c518", "Animation": "#66bb6a",
        "Drama": "#ef5350", "Action": "#ab47bc", "Family": "#ffa726",
    }
    colors = [color_map[g] for g in genres_demo]

    fig = go.Figure()
    for genre, color in color_map.items():
        indices = [i for i, g in enumerate(genres_demo) if g == genre]
        fig.add_trace(go.Scatter(
            x=[x_base[i] for i in indices],
            y=[y_base[i] for i in indices],
            mode="markers+text",
            name=genre,
            text=[labels[i] for i in indices],
            textposition="top center",
            textfont=dict(size=8, color="#cccccc"),
            marker=dict(size=12, color=color, line=dict(width=1, color="#000000")),
        ))

    fig.update_layout(
        title="🎯 Simplified 2-D Movie Feature Space",
        title_font=dict(color="#f5c518", size=16),
        paper_bgcolor="#111111",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#cccccc"),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#2a2a4e", borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor="#2a2a4e", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#2a2a4e", zeroline=False, showticklabels=False),
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Movies of the same genre cluster together. High cosine similarity = small angular distance = good recommendation.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    inject_css()

    # Initialise session state
    if "selected_movie" not in st.session_state:
        st.session_state["selected_movie"] = ""
    if "run_reco" not in st.session_state:
        st.session_state["run_reco"] = False
    if "reco_movie" not in st.session_state:
        st.session_state["reco_movie"] = ""

    # Load models once
    similarity, movies_df = load_models()

    # API key
    api_key = get_api_key()

    # Sidebar + routing
    page = render_sidebar(movies_df)

    if page == "🏠 Home":
        page_home(similarity, movies_df, api_key)
    elif page == "🎯 Recommendations":
        page_recommendations(similarity, movies_df, api_key)
    elif page == "ℹ️ About":
        page_about()
    elif page == "🧠 ML Explained":
        page_ml_explained(similarity, movies_df)


if __name__ == "__main__":
    main()
