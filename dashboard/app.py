"""
GenAI Diffusion Lens Dashboard

A comprehensive dashboard for analyzing user friction and session costs
in image generation workflows.

Run with:
    uv run streamlit run dashboard/app.py
"""

import streamlit as st
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import random

# =============================================================================
# Plotly Theme - StockPeers Style
# =============================================================================
CHART_COLORS = {
    'primary': '#615fff',      # Purple accent
    'secondary': '#818cf8',    # Lighter purple
    'success': '#4ade80',      # Green
    'warning': '#fbbf24',      # Amber
    'danger': '#f87171',       # Red
    'info': '#60a5fa',         # Blue
    'text': '#e2e8f0',         # Slate light
    'muted': '#94a3b8',        # Slate muted
    'border': '#314158',       # Border color
}

CHART_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': 'Space Grotesk, sans-serif', 'color': '#e2e8f0', 'size': 12},
        'title': {'font': {'size': 14, 'color': '#e2e8f0', 'weight': 400}},
        'xaxis': {
            'gridcolor': 'rgba(49, 65, 88, 0.5)',
            'linecolor': '#314158',
            'tickfont': {'size': 11, 'color': '#94a3b8'},
            'title': {'font': {'size': 12, 'color': '#94a3b8'}},
            'zeroline': False,
        },
        'yaxis': {
            'gridcolor': 'rgba(49, 65, 88, 0.5)',
            'linecolor': '#314158',
            'tickfont': {'size': 11, 'color': '#94a3b8'},
            'title': {'font': {'size': 12, 'color': '#94a3b8'}},
            'zeroline': False,
        },
        'legend': {
            'font': {'size': 11, 'color': '#e2e8f0'},
            'bgcolor': 'rgba(0,0,0,0)',
        },
        'margin': {'l': 40, 'r': 20, 't': 40, 'b': 40},
        'hoverlabel': {
            'bgcolor': '#1d293d',
            'bordercolor': '#314158',
            'font': {'color': '#e2e8f0', 'size': 12}
        }
    }
}

pio.templates['custom_dark'] = go.layout.Template(CHART_TEMPLATE)
pio.templates.default = 'custom_dark'

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Diffusion Lens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS - StockPeers Style
# =============================================================================
st.markdown("""
<style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Increase base font size */
    .main .block-container {
        font-size: 18px !important;
    }

    .main .block-container p,
    .main .block-container li,
    .main .block-container td,
    .main .block-container th,
    .main .block-container span,
    .main .block-container label,
    .main .block-container div {
        font-size: 18px !important;
        line-height: 1.7 !important;
    }

    .main .block-container h1 { font-size: 42px !important; }
    .main .block-container h2 { font-size: 32px !important; }
    .main .block-container h3 { font-size: 24px !important; }

    /* Make material icons in h1 match heading size */
    h1 span[data-testid="stIconMaterial"],
    h1 .material-symbols-rounded,
    h1 span.icon {
        font-size: 42px !important;
        vertical-align: -6px !important;
        line-height: 1 !important;
    }

    /* Target all possible icon selectors in headings */
    [data-testid="stMarkdownContainer"] h1 span {
        font-size: 42px !important;
        vertical-align: -6px !important;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        font-size: 18px !important;
    }

    /* Container content - match base font size */
    [data-testid="stVerticalBlock"] p,
    [data-testid="stVerticalBlock"] li,
    [data-testid="stVerticalBlock"] td,
    [data-testid="stVerticalBlock"] th,
    [data-testid="stVerticalBlock"] span,
    [data-testid="element-container"] p,
    [data-testid="element-container"] li,
    [data-testid="element-container"] td,
    [data-testid="element-container"] th {
        font-size: 18px !important;
        line-height: 1.7 !important;
    }

    /* Make containers in same row equal height */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] > div > [data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1 !important;
        height: 100% !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: transparent;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 400 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 0.75rem 1.25rem;
        font-weight: 400;
        border-bottom: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border-bottom: 2px solid #615fff !important;
        color: #e2e8f0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #615fff;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 400;
        padding: 0.5rem 1rem;
    }

    .stButton > button:hover {
        background: #5248e6;
    }

    /* Image containers */
    [data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        border-color: #314158 !important;
        margin: 1.5rem 0 !important;
    }

    /* Plotly charts */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* Code blocks */
    code {
        background: #0f172b !important;
        border-radius: 4px;
        padding: 0.15rem 0.35rem;
        font-size: 0.85rem;
        border: 1px solid #314158;
    }

    pre {
        background: #0f172b !important;
        border-radius: 8px;
        border: 1px solid #314158;
    }

    /* Sidebar nav styling */
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 20px !important;
    }

    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label span {
        font-size: 20px !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(97, 95, 255, 0.1);
    }

    /* Sidebar title and caption */
    [data-testid="stSidebar"] h1 {
        font-size: 28px !important;
    }

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        font-size: 16px !important;
    }

</style>
""", unsafe_allow_html=True)

# =============================================================================
# Database & Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"
IMAGES_PATH = PROJECT_ROOT / "data" / "blob" / "images" / "generations"


@st.cache_resource
def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)


def query(sql: str):
    conn = get_connection()
    return conn.execute(sql).fetchdf()


# =============================================================================
# Sidebar Navigation
# =============================================================================
PAGES = [":material/query_stats: Overview", ":material/build: Architecture", ":material/monitoring: Analytics", ":material/search: Session Explorer", ":material/smart_toy: SQL Copilot"]

# Initialize session state for page
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

st.sidebar.title("Diffusion Lens")
st.sidebar.caption("built on **DiffusionDB**")
st.sidebar.markdown("---")

def on_sidebar_change():
    st.session_state.page = st.session_state.sidebar_nav

st.sidebar.radio(
    "Navigate",
    PAGES,
    index=PAGES.index(st.session_state.page),
    label_visibility="collapsed",
    key="sidebar_nav",
    on_change=on_sidebar_change
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: DiffusionDB + Simulated Telemetry")


# =============================================================================
# PAGE: Overview
# =============================================================================
if st.session_state.page == ":material/query_stats: Overview":
    """
    # :material/query_stats: Diffusion Lens

    Analyze user friction, session costs, and churn risk across image generation workflows.
    """

    ""  # spacing

    # Dashboard Navigation Guide - Clickable cards with hover effect
    st.markdown("""
    <style>
    /* Nav card buttons - full container clickable */
    .nav-card-btn button {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        padding: 20px !important;
        height: 120px !important;
        width: 100% !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
    }
    .nav-card-btn button:hover {
        background: rgba(97, 95, 255, 0.15) !important;
        border-color: rgba(97, 95, 255, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    .nav-card-btn button p {
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    def nav_to(page_name):
        st.session_state.page = page_name

    nav_cols = st.columns(4)
    with nav_cols[0]:
        with st.container(key="nav_card_1"):
            st.markdown('<div class="nav-card-btn">', unsafe_allow_html=True)
            if st.button(":material/build: **Architecture**\n\nTech stack and pipeline design.", key="nav_arch", use_container_width=True):
                nav_to(":material/build: Architecture")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    with nav_cols[1]:
        with st.container(key="nav_card_2"):
            st.markdown('<div class="nav-card-btn">', unsafe_allow_html=True)
            if st.button(":material/monitoring: **Analytics**\n\nMetrics, trends, and friction patterns.", key="nav_analytics", use_container_width=True):
                nav_to(":material/monitoring: Analytics")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    with nav_cols[2]:
        with st.container(key="nav_card_3"):
            st.markdown('<div class="nav-card-btn">', unsafe_allow_html=True)
            if st.button(":material/search: **Session Explorer**\n\nSemantic search + image preview.", key="nav_explorer", use_container_width=True):
                nav_to(":material/search: Session Explorer")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    with nav_cols[3]:
        with st.container(key="nav_card_4"):
            st.markdown('<div class="nav-card-btn">', unsafe_allow_html=True)
            if st.button(":material/smart_toy: **SQL Copilot**\n\nNatural language to SQL queries.", key="nav_copilot", use_container_width=True):
                nav_to(":material/smart_toy: SQL Copilot")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    ""  # spacing

    # About Section
    with st.container(border=True):
        st.markdown("**What is Diffusion Lens?**")
        st.markdown("""
        Diffusion Lens is a **portfolio project** demonstrating a production-grade data pipeline
        for analyzing user behavior in image generation workflows. It answers critical business questions:

        - **Are users finding value?** Track engagement signals like downloads, feedback, and repeat sessions by tier
        - **What drives session costs?** Understand the relationship between prompt complexity, latency, and compute costs
        - **Where is quality degrading?** Identify error patterns, timeouts, and rate-limiting across user segments
        - **What content are users creating?** Use LLM-extracted features to categorize prompts by domain, style, and complexity

        Built as a "clean room" demonstration of data engineering instincts—ingesting real data,
        enriching with ML, and exposing actionable analytics through a modern stack.
        """)

    ""  # spacing

    # Data Source Section
    """
    ## Data Source
    """

    cols = st.columns([2, 1])

    with cols[0]:
        st.markdown("""
        [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb) is a large-scale
        text-to-image prompt dataset containing **14 million images** generated by Stable Diffusion.
        For this project, a **10K sample** is used to demonstrate the analytics pipeline.

        See the **Data Dictionary** below for field details.
        """)

        ""  # spacing

        # Quick Stats Section (inside left column)
        with st.container(border=True):
            st.markdown("**Quick Stats**")
            try:
                stats = query("""
                    WITH domain_stats AS (
                        SELECT llm_domain, COUNT(*) as cnt,
                               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
                        FROM ftr_llm_analysis WHERE llm_domain IS NOT NULL
                        GROUP BY llm_domain ORDER BY cnt DESC LIMIT 1
                    ),
                    style_stats AS (
                        SELECT llm_art_style, COUNT(*) as cnt,
                               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
                        FROM ftr_llm_analysis WHERE llm_art_style IS NOT NULL
                        GROUP BY llm_art_style ORDER BY cnt DESC LIMIT 1
                    )
                    SELECT
                        (SELECT ROUND(1.0 * COUNT(DISTINCT session_id) / COUNT(DISTINCT user_id), 1) FROM fct_sessions) as sessions_per_user,
                        (SELECT COUNT(*) FROM dim_users) as users,
                        (SELECT COUNT(*) FROM fct_sessions) as sessions,
                        (SELECT COUNT(*) FROM fct_generations) as generations,
                        (SELECT ROUND(AVG(success_rate_pct), 1) FROM fct_sessions) as success_rate,
                        (SELECT ROUND(SUM(total_cost_credits), 0) FROM fct_sessions) as total_cost,
                        (SELECT ROUND(AVG(avg_latency_ms), 0) FROM fct_sessions) as avg_latency,
                        (SELECT ROUND(AVG(friction_score), 1) FROM fct_sessions) as avg_friction,
                        (SELECT llm_domain FROM domain_stats) as top_domain,
                        (SELECT pct FROM domain_stats) as top_domain_pct,
                        (SELECT llm_art_style FROM style_stats) as top_style,
                        (SELECT pct FROM style_stats) as top_style_pct,
                        (SELECT ROUND(100.0 * SUM(CASE WHEN downloaded THEN 1 ELSE 0 END) / COUNT(*), 1) FROM fct_generations) as download_rate,
                        (SELECT ROUND(1.0 * COUNT(*) / COUNT(DISTINCT session_id), 1) FROM fct_generations) as gens_per_session,
                        (SELECT model_version FROM fct_generations GROUP BY model_version ORDER BY COUNT(*) DESC LIMIT 1) as top_model,
                        (SELECT ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fct_generations), 0) FROM fct_generations GROUP BY model_version ORDER BY COUNT(*) DESC LIMIT 1) as top_model_pct
                """)
                if stats is not None and len(stats) > 0:
                    row = stats.iloc[0]
                    # Row 1: Volume metrics
                    stat_cols = st.columns(4)
                    stat_cols[0].metric("Generations", f"{row['generations']:,}")
                    stat_cols[1].metric("Users", f"{row['users']:,}")
                    stat_cols[2].metric("Sessions", f"{row['sessions']:,}")
                    stat_cols[3].metric("Sessions/User", f"{row['sessions_per_user']}")
                    # Row 2: Quality & cost metrics
                    stat_cols2 = st.columns(4)
                    stat_cols2[0].metric("Gen Success Rate", f"{row['success_rate']}%")
                    stat_cols2[1].metric("Avg Latency", f"{int(row['avg_latency']):,}ms")
                    stat_cols2[2].metric("Total Cost", f"${row['total_cost']:,.0f}")
                    stat_cols2[3].metric(
                        "Avg Friction Score",
                        f"{row['avg_friction']:.1f}",
                        help="0-100 scale. Formula: (error_rate × 50) + (latency_norm × 33) + (retry_norm × 17). Higher = worse experience."
                    )
                    # Row 3: Content & enrichment metrics
                    stat_cols3 = st.columns(4)
                    domain_str = f"{row['top_domain']} ({row['top_domain_pct']:.0f}%)" if row['top_domain'] else "N/A"
                    style_str = f"{row['top_style']} ({row['top_style_pct']:.0f}%)" if row['top_style'] else "N/A"
                    stat_cols3[0].metric("Top Domain", domain_str)
                    stat_cols3[1].metric("Top Art Style", style_str)
                    stat_cols3[2].metric("Download Rate", f"{row['download_rate']:.1f}%")
                    model_str = f"{row['top_model']} ({row['top_model_pct']:.0f}%)" if row['top_model'] else "N/A"
                    stat_cols3[3].metric("Top Model", model_str, help="Most used model version")
                else:
                    st.caption("Run the pipeline to populate metrics.")
            except Exception as e:
                st.caption(f"Run the pipeline to populate metrics.")

    with cols[1]:
        st.markdown("<p style='text-align:center; font-weight:bold; margin-bottom:8px;'>Sample Image</p>", unsafe_allow_html=True)

        @st.fragment(run_every=3)
        def rotating_sample_image():
            if IMAGES_PATH.exists():
                images = list(IMAGES_PATH.glob("*.png"))
                if images:
                    selected_img = random.choice(images)
                    import base64
                    with open(selected_img, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    st.markdown(f"""
                        <div style="display:flex; justify-content:center; align-items:center; flex:1; min-height:320px;">
                            <div style="width:300px; height:300px; overflow:hidden; border-radius:8px;">
                                <img src="data:image/png;base64,{img_data}"
                                     style="width:100%; height:100%; object-fit:cover;">
                            </div>
                        </div>
                        <p style="text-align:center; color:rgba(250,250,250,0.6); font-size:14px; margin-top:8px;">Random sample from DiffusionDB</p>
                    """, unsafe_allow_html=True)

        rotating_sample_image()

    ""  # spacing

    # Feature Engineering Section
    """
    ## Feature Engineering
    """

    st.markdown("""
    The dataset starts with **real prompt data from DiffusionDB**, then layers on **simulated telemetry**
    (generation performance, user profiles, feedback signals) to create realistic usage patterns.
    **Semantic features** are extracted using LLMs and sentence transformers to categorize
    prompts by domain, style, and complexity. Finally, **composite KPIs** like friction score
    transform raw signals into actionable metrics for product analytics.
    """)

    with st.container(border=True):
        st.markdown("**Data Dictionary**")

        with st.expander("**DiffusionDB** — Real prompt data from HuggingFace", expanded=False):
            st.markdown("""
| Field | Type | Description |
|-------|------|-------------|
| `prompt` | text | The text prompt used for image generation |
| `seed` | int | Random seed for reproducibility |
| `cfg` | float | Classifier-free guidance scale (prompt adherence) |
| `sampler` | string | Sampling method (DDIM, PLMS, etc.) |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `image` | blob | Generated image file (stored in `/data/blob/images/`) |
            """)

        with st.expander("**Simulated Users** — Synthetic user profiles", expanded=False):
            st.markdown("""
| Field | Type | Description | Logic |
|-------|------|-------------|-------|
| `user_id` | int | Unique user identifier | Sequential ID |
| `user_tier` | string | Subscription tier | 70% free, 25% pro, 5% enterprise |
| `signup_date` | timestamp | When user joined | Exponential growth toward end of month |
| `cohort_week` | string | Weekly cohort (e.g., "2025-W49") | Derived from signup_date |
| `region` | string | Geographic region | 30% us-west, 25% us-east, 20% europe, 15% asia, 10% other |
| `device_type` | string | Device used | 70% desktop, 25% mobile, 5% tablet |
            """)

        with st.expander("**Simulated Telemetry** — Generation-level metrics", expanded=False):
            st.markdown("""
**Status values:** `success` (completed), `timeout` (>30s), `safety_violation` (blocked), `rate_limited` (free tier throttled), `model_error` (infra failure)

| Field | Type | Description | Logic |
|-------|------|-------------|-------|
| `status` | string | Generation outcome | NSFW keywords → 85% `safety_violation`; >75 tokens → 15% `timeout`; free tier → 8% `rate_limited`; baseline 2% `model_error`; else `success` |
| `latency_ms` | int | Response time | Timeout=30s; safety rejection=100-500ms; success=~50ms/token with log-normal noise |
| `cost_credits` | float | Compute cost | `(tokens × 0.01) + (latency_sec × 0.05)` with tier discounts: pro -10%, enterprise -20% |
| `retry_count` | int | Retries before final status | Success: 85% zero, 12% one, 3% two; rate_limited: more retries |
| `feedback` | string | User rating | 10% free, 20% pro, 25% enterprise leave feedback; success → 80% thumbs_up |
| `downloaded` | bool | Did user download? | Only on success; base rate: 40% free, 65% pro, 80% enterprise; +15% if >30 tokens |
| `model_version` | string | Model used | v1.4 (5%), v1.5 (15%), v2.0 (30%), v2.1 (50%) |
| `token_count` | int | Prompt length | Word count of prompt text |
            """)

        with st.expander("**LLM Extraction** — Qwen2.5-1.5B via MLX", expanded=False):
            st.markdown("""
| Field | Type | Description | Values |
|-------|------|-------------|--------|
| `llm_domain` | string | Subject category | portrait, character, animal, environment, object, fantasy, scifi, fanart |
| `llm_art_style` | string | Visual style | photography, digital art, oil painting, watercolor, sketch, 3d render, anime, pixel art, concept art, pop art |
| `llm_complexity_score` | int | Prompt detail level | 1 (simple) to 5 (highly detailed) |
            """)

        with st.expander("**Text Embeddings** — Semantic search vectors", expanded=False):
            st.markdown("""
Embeddings are computed locally using sentence-transformers, stored in DuckDB, and indexed with HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search via the DuckDB VSS extension.

| Field | Type | Description | Details |
|-------|------|-------------|---------|
| `text_embedding` | float[384] | Semantic vector for prompt text | all-MiniLM-L6-v2 model |
| HNSW Index | — | Fast similarity search | O(log n) vs O(n) brute force |
| Similarity metric | — | Cosine similarity | `array_cosine_similarity(a, b)` |
            """)

        with st.expander("**Computed Metrics** — Derived scores", expanded=False):
            st.markdown("""
| Metric | Formula | Description |
|--------|---------|-------------|
| `friction_score` | `(error_rate × 50) + (latency_norm × 33) + (retry_norm × 17)` | Weighted frustration score (0-100) |
| `friction_category` | Based on weighted sum thresholds | low (<10%), medium (<30%), high (≥30%) |
| `success_rate_pct` | `successful / total × 100` | Percentage of successful generations |
| `feedback_rate_pct` | `(thumbs_up + thumbs_down) / total × 100` | Percentage leaving feedback |
| `download_rate_pct` | `downloads / successful × 100` | Download rate among successful generations |
            """)



# =============================================================================
# PAGE: Architecture
# =============================================================================
elif st.session_state.page == ":material/build: Architecture":
    """
    # :material/build: Architecture
    """

    st.markdown("""
    This project is built on a modern analytics stack optimized for **local development** and **fast iteration**.
    The pipeline follows a medallion architecture (raw → staging → marts) managed by dbt, with Dagster orchestrating
    asset materialization and Streamlit serving the analytics layer.
    """)

    ""  # spacing

    # Tech Stack Section (no heading, no container)
    tech_cols = st.columns(6)

    # Helper to convert local files to base64
    import base64
    def get_image_src(path_or_url):
        if path_or_url.startswith("http"):
            return path_or_url
        # Local file - convert to base64
        file_path = Path(path_or_url)
        if file_path.exists():
            with open(file_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = file_path.suffix.lower()
            mime = "image/svg+xml" if ext == ".svg" else "image/png"
            return f"data:{mime};base64,{data}"
        return path_or_url

    tech_stack = [
        ("https://cdn.simpleicons.org/duckdb/FFF000", "DuckDB", "Storage"),
        (str(Path(__file__).parent / "assets" / "dagster.svg"), "Dagster", "Orchestration"),
        (str(Path(__file__).parent / "assets" / "dbt.png"), "dbt", "Transforms"),
        ("https://cdn.simpleicons.org/apple/FFFFFF", "MLX", "ML Inference"),
        ("https://cdn.simpleicons.org/huggingface/FFD21E", "HuggingFace", "Embeddings"),
        ("https://cdn.simpleicons.org/streamlit/FF4B4B", "Streamlit", "Dashboard"),
    ]
    for col, (logo_url, name, layer) in zip(tech_cols, tech_stack):
        with col:
            img_src = get_image_src(logo_url)
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="{img_src}" width="40" style="margin-bottom: 8px;">
                <p style="margin: 0; font-weight: bold;">{name}</p>
                <p style="margin: 0; font-size: 14px; color: rgba(250,250,250,0.6);">{layer}</p>
            </div>
            """, unsafe_allow_html=True)

    ""  # spacing

    # Design Decisions
    with st.expander("**Design Decisions** — Why these tools? Can they scale?", expanded=False):
        st.markdown("""
| Tool | Why | Can it scale? |
|------|-----|-------|
| **DuckDB** | Embedded OLAP, no server setup, native Parquet | Migrate to MotherDuck (hosted DuckDB), or swap to Snowflake/Databricks |
| **Dagster** | Asset-based orchestration, not task-based | Better mental model for data pipelines; Airflow has larger community but task-centric |
| **dbt** | SQL transforms, testable, tracks lineage | Covers most transform logic; doesn't handle non-SQL steps. Infra-agnostic — works with any warehouse |
| **MLX** | Fast local inference on Apple Silicon | Demo choice; swap to vLLM, TGI, or cloud inference (Bedrock, OpenAI) in production |
| **HuggingFace** | Lightweight text embeddings | Design choice; any embedding provider works |
| **Streamlit** | Quick to build, hot reload | Design choice; could swap to Tableau, QuickSight, or custom BI |
        """)

    ""  # spacing

    # System Architecture
    arch_image = Path(__file__).parent / "assets" / "sys_architecture.png"
    if arch_image.exists():
        st.markdown("## System Architecture")
        st.image(str(arch_image), use_container_width=True)

    ""  # spacing

    # Schema Section
    """
    ## Data Model
    """

    st.markdown("""
**Raw Layer** — Source data ingested from DiffusionDB + simulated telemetry

| Table | Description |
|-------|-------------|
| `raw_prompts` | Prompt text, seed, cfg, sampler, dimensions |
| `raw_users` | User profiles with tier, region, device |
| `raw_generations` | Telemetry: status, latency, cost, feedback, downloaded |
| `raw_prompt_enrichments` | ML features: LLM analysis + text embeddings |

**Staging Layer** — Cleaned and typed (`stg_*`)

| View | Source |
|------|--------|
| `stg_prompts` | raw_prompts with char_count, token_count |
| `stg_users` | raw_users (pass-through) |
| `stg_generations` | raw_generations with session_id assigned |

**Marts Layer** — Star schema dimensions and facts

| Table | Key Fields |
|-------|------------|
| `dim_users` | user_id, user_tier, region, lifetime_cost, success_rate_pct |
| `dim_prompts` | prompt_id, prompt_text, token_count, prompt_length_category |
| `fct_generations` | generation_id, user_id, prompt_id, status, latency_ms, cost_credits |
| `fct_sessions` | session_id, friction_score, friction_category, total_cost_credits |

**Feature Layer** — ML enrichment views (`ftr_*`)

| View | Fields |
|------|--------|
| `ftr_llm_analysis` | llm_domain, llm_art_style, llm_complexity_score, image_path |
| `ftr_text_embeddings` | text_embedding (384-dim vector for semantic search) |
    """)


# =============================================================================
# PAGE: Analytics
# =============================================================================
elif st.session_state.page == ":material/monitoring: Analytics":
    """
    # :material/monitoring: Analytics

    Explore metrics, trends, and friction patterns across user segments.
    """

    ""  # spacing

    try:
        # Key Metrics Row
        metrics = query("""
            SELECT
                COUNT(DISTINCT user_id) as total_users,
                COUNT(DISTINCT session_id) as total_sessions,
                SUM(total_generations) as total_generations,
                ROUND(AVG(friction_score), 2) as avg_friction,
                ROUND(SUM(total_cost_credits), 2) as total_cost,
                ROUND(AVG(success_rate_pct), 1) as avg_success_rate,
                ROUND(1.0 * COUNT(DISTINCT session_id) / COUNT(DISTINCT user_id), 1) as sessions_per_user,
                (SELECT ROUND(100.0 * SUM(CASE WHEN downloaded THEN 1 ELSE 0 END) / COUNT(*), 1) FROM fct_generations) as download_rate
            FROM fct_sessions
        """).iloc[0]

        cols = st.columns([1, 2])

        with cols[0].container(border=True):
            st.markdown("**Key Metrics**")
            ""
            metric_cols = st.columns(2)
            metric_cols[0].metric("Users", f"{int(metrics['total_users']):,}")
            metric_cols[1].metric("Sessions", f"{int(metrics['total_sessions']):,}")
            ""
            metric_cols = st.columns(2)
            metric_cols[0].metric("Generations", f"{int(metrics['total_generations']):,}")
            metric_cols[1].metric("Success Rate", f"{metrics['avg_success_rate']}%")
            ""
            metric_cols = st.columns(2)
            metric_cols[0].metric("Avg Friction", f"{metrics['avg_friction']}")
            metric_cols[1].metric("Revenue", f"${metrics['total_cost']:,.0f}")
            ""
            metric_cols = st.columns(2)
            metric_cols[0].metric("Sessions/User", f"{metrics['sessions_per_user']}")
            metric_cols[1].metric("Download Rate", f"{metrics['download_rate']}%")

        with cols[1].container(border=True):
            st.markdown("**Daily Generation Activity**")
            daily_activity = query("""
                SELECT
                    session_date as date,
                    COUNT(*) as generations,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes
                FROM raw_generations
                GROUP BY session_date
                ORDER BY session_date
            """)
            if not daily_activity.empty:
                fig = px.area(
                    daily_activity, x='date', y='generations',
                    color_discrete_sequence=[CHART_COLORS['primary']]
                )
                fig.update_traces(
                    line=dict(width=2, color=CHART_COLORS['primary']),
                    fillcolor='rgba(97, 95, 255, 0.15)'
                )
                fig.update_layout(
                    height=340, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="", yaxis_title="",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # =================================================================
        # SECTION: User Segments
        # =================================================================
        """
        ## User Segments
        Who's using the platform? Breakdown by device, region, and tier.
        """

        col1, col2, col3 = st.columns(3)

        with col1.container(border=True):
            st.markdown("**By Device**")
            device_data = query("""
                SELECT device_type, COUNT(*) as users
                FROM dim_users
                GROUP BY device_type
                ORDER BY users DESC
            """)
            if not device_data.empty:
                fig = px.pie(device_data, values='users', names='device_type',
                            color_discrete_sequence=[CHART_COLORS['primary'], CHART_COLORS['secondary'], CHART_COLORS['info']],
                            hole=0.5)
                fig.update_traces(textposition='outside', textinfo='percent+label', textfont_size=11)
                fig.update_layout(height=250, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

        with col2.container(border=True):
            st.markdown("**By Region**")
            region_data = query("""
                SELECT region, COUNT(*) as users
                FROM dim_users
                GROUP BY region
                ORDER BY users DESC
            """)
            if not region_data.empty:
                fig = px.bar(region_data, x='users', y='region', orientation='h',
                            color_discrete_sequence=[CHART_COLORS['success']])
                fig.update_traces(marker_line_width=0)
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="", yaxis_title="", showlegend=False,
                                yaxis=dict(categoryorder='total ascending'))
                st.plotly_chart(fig, use_container_width=True)

        with col3.container(border=True):
            st.markdown("**By Tier**")
            tier_data = query("""
                SELECT user_tier, COUNT(*) as users
                FROM dim_users
                GROUP BY user_tier
                ORDER BY users DESC
            """)
            if not tier_data.empty:
                fig = px.pie(tier_data, values='users', names='user_tier',
                            color='user_tier',
                            color_discrete_map={'free': CHART_COLORS['info'], 'pro': CHART_COLORS['warning'], 'enterprise': CHART_COLORS['success']},
                            hole=0.5)
                fig.update_traces(textposition='outside', textinfo='percent+label', textfont_size=11)
                fig.update_layout(height=250, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # =================================================================
        # SECTION: Content Intelligence (LLM Features - HERO SECTION)
        # =================================================================
        """
        ## Content Intelligence
        ML-extracted features from prompt analysis. These are **real features** derived from actual DiffusionDB prompts using Qwen2.5 LLM.
        """

        col1, col2 = st.columns(2)

        with col1.container(border=True):
            st.markdown("**Domain × Error Rate**")
            st.caption("Which content types fail more?")
            domain_errors = query("""
                SELECT
                    f.llm_domain as domain,
                    COUNT(*) as total,
                    ROUND(100.0 * SUM(CASE WHEN g.status != 'success' THEN 1 ELSE 0 END) / COUNT(*), 1) as error_rate
                FROM fct_generations g
                JOIN ftr_llm_analysis f ON g.prompt_id = f.prompt_id
                WHERE f.llm_domain IS NOT NULL
                GROUP BY f.llm_domain
                ORDER BY error_rate DESC
            """)
            if not domain_errors.empty:
                fig = px.bar(domain_errors, x='error_rate', y='domain', orientation='h',
                            color='error_rate',
                            color_continuous_scale=[[0, CHART_COLORS['success']], [0.5, CHART_COLORS['warning']], [1, CHART_COLORS['danger']]])
                fig.update_traces(marker_line_width=0, text=domain_errors['error_rate'].apply(lambda x: f'{x}%'), textposition='outside')
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="Error Rate %", yaxis_title="",
                                showlegend=False, coloraxis_showscale=False,
                                yaxis=dict(categoryorder='total ascending'))
                st.plotly_chart(fig, use_container_width=True)

        with col2.container(border=True):
            st.markdown("**Art Style × Avg Latency**")
            st.caption("Which styles take longer to generate?")
            style_latency = query("""
                SELECT
                    f.llm_art_style as style,
                    COUNT(*) as total,
                    ROUND(AVG(g.latency_ms), 0) as avg_latency
                FROM fct_generations g
                JOIN ftr_llm_analysis f ON g.prompt_id = f.prompt_id
                WHERE f.llm_art_style IS NOT NULL
                GROUP BY f.llm_art_style
                ORDER BY avg_latency DESC
            """)
            if not style_latency.empty:
                fig = px.bar(style_latency, x='avg_latency', y='style', orientation='h',
                            color='avg_latency',
                            color_continuous_scale=[[0, CHART_COLORS['success']], [0.5, CHART_COLORS['warning']], [1, CHART_COLORS['danger']]])
                fig.update_traces(marker_line_width=0, text=style_latency['avg_latency'].apply(lambda x: f'{int(x)}ms'), textposition='outside')
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="Avg Latency (ms)", yaxis_title="",
                                showlegend=False, coloraxis_showscale=False,
                                yaxis=dict(categoryorder='total ascending'))
                st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1.container(border=True):
            st.markdown("**Complexity × Cost**")
            st.caption("Do complex prompts cost more?")
            complexity_cost = query("""
                SELECT
                    f.complexity_category,
                    ROUND(AVG(g.cost_credits), 3) as avg_cost,
                    COUNT(*) as count
                FROM fct_generations g
                JOIN ftr_llm_analysis f ON g.prompt_id = f.prompt_id
                WHERE f.complexity_category IS NOT NULL
                GROUP BY f.complexity_category
                ORDER BY CASE f.complexity_category WHEN 'low' THEN 1 WHEN 'medium' THEN 2 WHEN 'high' THEN 3 END
            """)
            if not complexity_cost.empty:
                fig = px.bar(complexity_cost, x='complexity_category', y='avg_cost',
                            color='complexity_category',
                            color_discrete_map={'low': CHART_COLORS['success'], 'medium': CHART_COLORS['warning'], 'high': CHART_COLORS['danger']},
                            text='avg_cost')
                fig.update_traces(texttemplate='$%{text:.3f}', textposition='outside', marker_line_width=0)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="", yaxis_title="Avg Cost ($)",
                                showlegend=False, bargap=0.4)
                st.plotly_chart(fig, use_container_width=True)

        with col2.container(border=True):
            st.markdown("**Complexity × Friction**")
            st.caption("Do complex prompts cause more friction?")
            complexity_friction = query("""
                SELECT
                    f.complexity_category,
                    ROUND(AVG(s.friction_score), 2) as avg_friction,
                    COUNT(*) as count
                FROM fct_sessions s
                JOIN fct_generations g ON s.session_id = g.session_id
                JOIN ftr_llm_analysis f ON g.prompt_id = f.prompt_id
                WHERE f.complexity_category IS NOT NULL
                GROUP BY f.complexity_category
                ORDER BY CASE f.complexity_category WHEN 'low' THEN 1 WHEN 'medium' THEN 2 WHEN 'high' THEN 3 END
            """)
            if not complexity_friction.empty:
                fig = px.bar(complexity_friction, x='complexity_category', y='avg_friction',
                            color='complexity_category',
                            color_discrete_map={'low': CHART_COLORS['success'], 'medium': CHART_COLORS['warning'], 'high': CHART_COLORS['danger']},
                            text='avg_friction')
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside', marker_line_width=0)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="", yaxis_title="Avg Friction",
                                showlegend=False, bargap=0.4)
                st.plotly_chart(fig, use_container_width=True)

        # Download Rate row - key success indicator
        col1, col2 = st.columns(2)

        with col1.container(border=True):
            st.markdown("**Download Rate × Domain**")
            st.caption("Which content types do users keep?")
            domain_downloads = query("""
                SELECT
                    f.llm_domain as domain,
                    COUNT(*) as total,
                    ROUND(100.0 * SUM(CASE WHEN g.downloaded THEN 1 ELSE 0 END) / COUNT(*), 1) as download_rate
                FROM fct_generations g
                JOIN ftr_llm_analysis f ON g.prompt_id = f.prompt_id
                WHERE f.llm_domain IS NOT NULL AND g.status = 'success'
                GROUP BY f.llm_domain
                ORDER BY download_rate DESC
            """)
            if not domain_downloads.empty:
                fig = px.bar(domain_downloads, x='download_rate', y='domain', orientation='h',
                            color='download_rate',
                            color_continuous_scale=[[0, CHART_COLORS['danger']], [0.5, CHART_COLORS['warning']], [1, CHART_COLORS['success']]])
                fig.update_traces(marker_line_width=0, text=domain_downloads['download_rate'].apply(lambda x: f'{x}%'), textposition='outside')
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="Download Rate %", yaxis_title="",
                                showlegend=False, coloraxis_showscale=False,
                                yaxis=dict(categoryorder='total ascending'))
                st.plotly_chart(fig, use_container_width=True)

        with col2.container(border=True):
            st.markdown("**Download Rate × Art Style**")
            st.caption("Which styles do users actually keep?")
            style_downloads = query("""
                SELECT
                    f.llm_art_style as style,
                    COUNT(*) as total,
                    ROUND(100.0 * SUM(CASE WHEN g.downloaded THEN 1 ELSE 0 END) / COUNT(*), 1) as download_rate
                FROM fct_generations g
                JOIN ftr_llm_analysis f ON g.prompt_id = f.prompt_id
                WHERE f.llm_art_style IS NOT NULL AND g.status = 'success'
                GROUP BY f.llm_art_style
                ORDER BY download_rate DESC
            """)
            if not style_downloads.empty:
                fig = px.bar(style_downloads, x='download_rate', y='style', orientation='h',
                            color='download_rate',
                            color_continuous_scale=[[0, CHART_COLORS['danger']], [0.5, CHART_COLORS['warning']], [1, CHART_COLORS['success']]])
                fig.update_traces(marker_line_width=0, text=style_downloads['download_rate'].apply(lambda x: f'{x}%'), textposition='outside')
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="Download Rate %", yaxis_title="",
                                showlegend=False, coloraxis_showscale=False,
                                yaxis=dict(categoryorder='total ascending'))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # =================================================================
        # SECTION: Quality by Segment
        # =================================================================
        """
        ## Quality by Segment
        Where is friction occurring? Breaking down quality metrics by user segments.
        """

        col1, col2 = st.columns(2)

        with col1.container(border=True):
            st.markdown("**Friction by Device**")
            st.caption("Are mobile users struggling more?")
            device_friction = query("""
                SELECT
                    u.device_type,
                    ROUND(AVG(s.friction_score), 2) as avg_friction,
                    COUNT(*) as sessions
                FROM fct_sessions s
                JOIN dim_users u USING (user_id)
                GROUP BY u.device_type
                ORDER BY avg_friction DESC
            """)
            if not device_friction.empty:
                fig = px.bar(device_friction, x='device_type', y='avg_friction',
                            color='avg_friction',
                            color_continuous_scale=[[0, CHART_COLORS['success']], [0.5, CHART_COLORS['warning']], [1, CHART_COLORS['danger']]],
                            text='avg_friction')
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside', marker_line_width=0)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="", yaxis_title="Avg Friction",
                                showlegend=False, coloraxis_showscale=False, bargap=0.4)
                st.plotly_chart(fig, use_container_width=True)

        with col2.container(border=True):
            st.markdown("**Friction by Tier**")
            st.caption("Free users experience more friction")
            tier_friction = query("""
                SELECT
                    u.user_tier,
                    ROUND(AVG(s.friction_score), 2) as avg_friction,
                    COUNT(*) as sessions
                FROM fct_sessions s
                JOIN dim_users u USING (user_id)
                GROUP BY u.user_tier
                ORDER BY avg_friction DESC
            """)
            if not tier_friction.empty:
                fig = px.bar(tier_friction, x='user_tier', y='avg_friction',
                            color='user_tier',
                            color_discrete_map={'free': CHART_COLORS['danger'], 'pro': CHART_COLORS['warning'], 'enterprise': CHART_COLORS['success']},
                            text='avg_friction')
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside', marker_line_width=0)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="", yaxis_title="Avg Friction",
                                showlegend=False, bargap=0.4)
                st.plotly_chart(fig, use_container_width=True)

        # Model Version Performance
        col3, col4 = st.columns(2)

        with col3.container(border=True):
            st.markdown("**Error Rate by Model Version**")
            st.caption("Newer models should have lower error rates")
            model_errors = query("""
                SELECT
                    model_version,
                    ROUND(100.0 * SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) / COUNT(*), 1) as error_rate,
                    COUNT(*) as generations
                FROM fct_generations
                GROUP BY model_version
                ORDER BY model_version
            """)
            if not model_errors.empty:
                fig = px.bar(model_errors, x='model_version', y='error_rate',
                            color='error_rate',
                            color_continuous_scale=[[0, CHART_COLORS['success']], [0.5, CHART_COLORS['warning']], [1, CHART_COLORS['danger']]],
                            text='error_rate')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="", yaxis_title="Error Rate %",
                                showlegend=False, coloraxis_showscale=False, bargap=0.4)
                st.plotly_chart(fig, use_container_width=True)

        with col4.container(border=True):
            st.markdown("**Latency by Model Version**")
            st.caption("Newer models may trade speed for quality")
            model_latency = query("""
                SELECT
                    model_version,
                    ROUND(AVG(latency_ms)) as avg_latency,
                    COUNT(*) as generations
                FROM fct_generations
                GROUP BY model_version
                ORDER BY model_version
            """)
            if not model_latency.empty:
                fig = px.bar(model_latency, x='model_version', y='avg_latency',
                            color='avg_latency',
                            color_continuous_scale=[[0, CHART_COLORS['success']], [0.5, CHART_COLORS['warning']], [1, CHART_COLORS['danger']]],
                            text='avg_latency')
                fig.update_traces(texttemplate='%{text:,.0f}ms', textposition='outside', marker_line_width=0)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="", yaxis_title="Avg Latency (ms)",
                                showlegend=False, coloraxis_showscale=False, bargap=0.4)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # =================================================================
        # SECTION: Weekly Trends
        # =================================================================
        """
        ## Weekly Trends
        Growth and engagement patterns across the month.
        """

        weekly = query("""
            SELECT
                CASE
                    WHEN DAY(session_date) <= 7 THEN 'Week 1'
                    WHEN DAY(session_date) <= 14 THEN 'Week 2'
                    WHEN DAY(session_date) <= 21 THEN 'Week 3'
                    ELSE 'Week 4'
                END as week,
                COUNT(*) as generations,
                COUNT(DISTINCT user_id) as active_users,
                ROUND(SUM(cost_credits), 2) as revenue,
                ROUND(AVG(CASE WHEN status != 'success' THEN 1 ELSE 0 END) * 100, 1) as error_rate
            FROM raw_generations
            GROUP BY week
            ORDER BY week
        """)

        if not weekly.empty:
            col1, col2 = st.columns(2)

            with col1.container(border=True):
                st.markdown("**Generations & Users**")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=weekly['week'], y=weekly['generations'], name='Generations',
                                    marker_color=CHART_COLORS['primary'], marker_line_width=0))
                fig.add_trace(go.Scatter(x=weekly['week'], y=weekly['active_users'], name='Active Users',
                                        line=dict(color=CHART_COLORS['warning'], width=3), yaxis='y2'))
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                yaxis=dict(title='Generations'), yaxis2=dict(title='Users', overlaying='y', side='right'),
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, bgcolor='rgba(0,0,0,0)'),
                                bargap=0.4)
                st.plotly_chart(fig, use_container_width=True)

            with col2.container(border=True):
                st.markdown("**Revenue & Error Rate**")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=weekly['week'], y=weekly['revenue'], name='Revenue ($)',
                                    marker_color=CHART_COLORS['success'], marker_line_width=0))
                fig.add_trace(go.Scatter(x=weekly['week'], y=weekly['error_rate'], name='Error Rate %',
                                        line=dict(color=CHART_COLORS['danger'], width=3), yaxis='y2'))
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                yaxis=dict(title='Revenue ($)'), yaxis2=dict(title='Error Rate %', overlaying='y', side='right'),
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, bgcolor='rgba(0,0,0,0)'),
                                bargap=0.4)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # =================================================================
        # SECTION: Deep Dive
        # =================================================================
        """
        ## Deep Dive
        Explore sessions and compare tiers.
        """

        # Tier Comparison Table
        st.markdown("**Tier Comparison**")

        tier_table = query("""
            SELECT
                user_tier as "Tier",
                total_users as "Users",
                total_sessions as "Sessions",
                avg_friction_score as "Avg Friction",
                avg_success_rate_pct as "Success %",
                avg_latency_ms as "Avg Latency (ms)",
                total_cost_credits as "Total Cost",
                feedback_rate_pct as "Feedback %",
                download_rate_pct as "Download %"
            FROM user_friction_summary
            ORDER BY total_users DESC
        """)

        st.dataframe(tier_table, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Session Filters
        st.markdown("**Session Filters**")

        col1, col2, col3 = st.columns(3)
        with col1:
            tier_filter = st.selectbox("User Tier", ["All", "free", "pro", "enterprise"])
        with col2:
            friction_filter = st.selectbox("Friction Level", ["All", "high", "medium", "low"])
        with col3:
            sort_by = st.selectbox("Sort By", ["friction_score DESC", "total_cost_credits DESC", "total_generations DESC"])

        where_clauses = []
        if tier_filter != "All":
            where_clauses.append(f"user_tier = '{tier_filter}'")
        if friction_filter != "All":
            where_clauses.append(f"friction_category = '{friction_filter}'")

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        sessions = query(f"""
            SELECT
                session_id,
                user_tier,
                total_generations,
                success_rate_pct,
                avg_latency_ms,
                total_cost_credits,
                friction_score,
                friction_category
            FROM fct_sessions
            {where_sql}
            ORDER BY {sort_by}
            LIMIT 50
        """)

        st.dataframe(sessions, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        st.info("Make sure the pipeline has been run and data exists.")


# =============================================================================
# PAGE: SQL Copilot
# =============================================================================
elif st.session_state.page == ":material/smart_toy: SQL Copilot":
    st.title(":material/smart_toy: SQL Copilot")
    st.markdown("Ask questions in natural language, get SQL queries powered by local LLM")
    st.caption("🧠 **Model:** `Qwen2.5-7B-Instruct-4bit` via MLX")

    st.markdown("---")

    # How it works
    with st.expander("How it works", expanded=False):
        st.markdown("""
        1. **You ask** a question in plain English
        2. **LLM generates** SQL using your schema context
        3. **You review** and optionally edit the SQL
        4. **Execute** against DuckDB and see results

        **Model:** Qwen2.5-7B-Instruct-4bit (runs locally via MLX, ~15s per query)

        **Example questions:**
        - "What is the average latency by user tier?"
        - "Show me the top 10 users by session cost"
        - "Which art styles have the highest error rate?"
        """)

    st.markdown("---")

    try:
        from src.copilot.schema import get_full_schema
        from src.copilot.prompt import construct_sql_prompt
        from src.copilot.llm import load_model, generate_sql, validate_sql

        @st.cache_resource
        def get_copilot_model():
            return load_model()

        @st.cache_data
        def get_schema_text():
            con = get_connection()
            return get_full_schema(con)

        # Example questions
        example_questions = [
            "What is the average latency by user tier?",
            "Show me error rate by art style",
            "Top 10 users by total cost",
            "Which domains have the highest download rate?",
            "Average friction score by device type",
            "Daily generation count for December"
        ]

        # Initialize session state
        if 'copilot_question' not in st.session_state:
            st.session_state.copilot_question = ""
        if 'copilot_generated_sql' not in st.session_state:
            st.session_state.copilot_generated_sql = ""
        if 'copilot_last_question' not in st.session_state:
            st.session_state.copilot_last_question = ""
        if 'copilot_gen_time' not in st.session_state:
            st.session_state.copilot_gen_time = 0

        user_question = st.text_input(
            "Ask a question about your data:",
            value=st.session_state.copilot_question,
            placeholder="e.g., What is the average latency by user tier?"
        )

        # Example question buttons
        st.caption("**Try an example:**")
        cols = st.columns(3)
        for i, example in enumerate(example_questions):
            with cols[i % 3]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.copilot_question = example
                    st.session_state.copilot_generated_sql = ""  # Clear cached SQL
                    st.session_state.copilot_last_question = ""
                    st.rerun()

        if user_question:
            # Only generate if question changed
            need_generation = (user_question != st.session_state.copilot_last_question)

            if need_generation:
                import time
                import threading

                status_placeholder = st.empty()
                result = {"sql": "", "error": None}
                generation_done = threading.Event()

                def run_generation():
                    try:
                        model, tokenizer, sampler = get_copilot_model()
                        schema_text = get_schema_text()
                        prompt = construct_sql_prompt(user_question, schema_text)
                        result["sql"] = generate_sql(prompt, model, tokenizer, sampler)
                    except Exception as e:
                        result["error"] = e
                    finally:
                        generation_done.set()

                thread = threading.Thread(target=run_generation)
                start_time = time.time()
                thread.start()

                while not generation_done.is_set():
                    elapsed = time.time() - start_time
                    status_placeholder.info(f"🔄 Generating SQL... **{elapsed:.1f}s**")
                    time.sleep(0.1)

                thread.join()
                elapsed = time.time() - start_time

                if result["error"]:
                    status_placeholder.error(f"Model error: {result['error']}")
                else:
                    st.session_state.copilot_generated_sql = result["sql"]
                    st.session_state.copilot_last_question = user_question
                    st.session_state.copilot_gen_time = elapsed
                    status_placeholder.success(f"✅ Generated in **{elapsed:.1f}s**")
            else:
                # Show cached generation time
                st.success(f"✅ Generated in **{st.session_state.copilot_gen_time:.1f}s**")

            if st.session_state.copilot_generated_sql:
                st.markdown("#### Generated SQL")
                edited_sql = st.text_area(
                    "Review and edit if needed:",
                    value=st.session_state.copilot_generated_sql,
                    height=150,
                    label_visibility="collapsed"
                )

                if st.button("▶️ Run Query", type="primary"):
                    is_valid, error_msg = validate_sql(edited_sql)

                    if not is_valid:
                        st.error(f"⚠️ {error_msg}")
                    else:
                        try:
                            result_df = query(edited_sql)
                            st.markdown("#### Results")
                            st.dataframe(result_df, use_container_width=True, hide_index=True)
                            st.caption(f"Returned {len(result_df)} rows")
                        except Exception as e:
                            st.error(f"Query error: {str(e)}")

    except ImportError:
        st.warning("SQL Copilot modules not found. Run from project root.")


# =============================================================================
# PAGE: Session Explorer
# =============================================================================
elif st.session_state.page == ":material/search: Session Explorer":
    st.title(":material/search: Session Explorer")
    st.markdown("Semantic search + LLM-generated filters + image preview")
    st.caption("🧠 **Model:** `all-MiniLM-L6-v2` (384-dim embeddings) + DuckDB VSS")

    st.markdown("---")

    # Check if we have embeddings and images (from raw_prompt_enrichments table)
    has_embeddings = False
    has_images = False
    try:
        embedding_check = query("""
            SELECT
                COUNT(*) FILTER (WHERE text_embedding IS NOT NULL) as embeddings,
                COUNT(*) FILTER (WHERE image_path IS NOT NULL) as images
            FROM raw_prompt_enrichments
        """)
        has_embeddings = embedding_check['embeddings'].iloc[0] > 0
        has_images = embedding_check['images'].iloc[0] > 0
    except Exception:
        pass

    if not has_embeddings:
        st.warning("No embeddings found. Run: `uv run python -m src.enrichment.precompute_embeddings`")
        st.info("Session Explorer requires pre-computed text embeddings for semantic search.")
    else:
        try:
            from sentence_transformers import SentenceTransformer

            @st.cache_resource
            def get_embedding_model():
                return SentenceTransformer('all-MiniLM-L6-v2')

            # Example searches
            example_searches = [
                "cyberpunk city at night",
                "fantasy castle landscape",
                "portrait of a woman",
                "anime character illustration",
                "futuristic spaceship",
                "watercolor painting nature"
            ]

            # Initialize session state for search
            if 'explorer_search' not in st.session_state:
                st.session_state.explorer_search = ""

            # Search input
            search_query = st.text_input(
                "Search prompts:",
                value=st.session_state.explorer_search,
                placeholder="e.g., cyberpunk city, fantasy castle, portrait of a woman"
            )

            # Clickable example searches
            search_cols = st.columns(3)
            for i, example in enumerate(example_searches):
                with search_cols[i % 3]:
                    if st.button(example, key=f"search_example_{i}", use_container_width=True):
                        st.session_state.explorer_search = example
                        st.rerun()

            st.markdown("---")

            if search_query:
                with st.spinner("Searching..."):
                    model = get_embedding_model()
                    query_embedding = model.encode([search_query])[0].tolist()

                    results = query(f"""
                        SELECT
                            p.prompt_id,
                            p.prompt_text,
                            e.image_path,
                            array_cosine_similarity(e.text_embedding, {query_embedding}::FLOAT[384]) AS similarity
                        FROM dim_prompts p
                        JOIN raw_prompt_enrichments e ON p.prompt_id = e.prompt_id
                        WHERE e.text_embedding IS NOT NULL
                        ORDER BY similarity DESC
                        LIMIT 12
                    """)

                if len(results) == 0:
                    st.warning("No results found. Try a different search query.")
                else:
                    st.subheader(f"Results for '{search_query}'")
                    st.caption(f"Found {len(results)} similar prompts")

                    # Image grid (4 columns)
                    if has_images:
                        cols = st.columns(4)
                        for idx, row in results.iterrows():
                            with cols[idx % 4]:
                                image_path = PROJECT_ROOT / row['image_path'] if row['image_path'] else None
                                full_prompt = row['prompt_text'] or ''

                                if image_path and image_path.exists():
                                    with st.popover(f"🔍 {row['prompt_text'][:30]}..."):
                                        st.markdown("**Full Prompt:**")
                                        st.write(full_prompt)
                                        st.markdown(f"**Similarity:** {row['similarity']:.3f}")
                                    st.image(str(image_path), use_container_width=True)
                                else:
                                    st.markdown(
                                        f"""<div style="background:#2d2d2d;height:150px;display:flex;
                                        align-items:center;justify-content:center;border-radius:8px;
                                        color:#888;font-size:12px;">No image</div>""",
                                        unsafe_allow_html=True
                                    )

                                st.caption(f"**{row['prompt_text'][:40]}...**")
                                st.caption(f"Similarity: {row['similarity']:.2f}")

                    else:
                        st.info("No images downloaded yet. Run: `uv run python -m src.ingestion.download_images`")
                        display_df = results[['prompt_text', 'similarity']].copy()
                        display_df.columns = ['Prompt', 'Similarity']
                        display_df['Similarity'] = display_df['Similarity'].round(3)
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

            else:
                st.info("Enter a search query or click an example above to find similar prompts")

        except ImportError:
            st.warning("sentence-transformers not installed. Run: `uv add sentence-transformers`")


# =============================================================================
# Footer
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Links**")
st.sidebar.markdown("[GitHub](https://github.com) | [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb)")
