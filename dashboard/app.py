"""
GenAI Session Analyzer Dashboard

A comprehensive dashboard for analyzing user friction and session costs
in image generation workflows.

Run with:
    cd /Users/jaychava/Documents/Luma/genai-session-analyzer
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
    page_title="Session Analyzer",
    page_icon="📊",
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
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(97, 95, 255, 0.1);
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
st.sidebar.title("Session Analyzer")
st.sidebar.caption("built on **DiffusionDB**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "📈 Analytics", "🤖 SQL Copilot", "🔍 Session Explorer"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: DiffusionDB + Simulated Telemetry")


# =============================================================================
# PAGE: Overview
# =============================================================================
if page == "📊 Overview":
    """
    # :material/query_stats: Session Analyzer

    A data platform for understanding user friction and costs in generative AI workflows.
    """

    ""  # spacing

    # About Section
    with st.container(border=True):
        st.markdown("**What is Session Analyzer?**")
        st.markdown("""
        Session Analyzer is a **portfolio project** demonstrating a production-grade data pipeline
        for analyzing user behavior in GenAI applications. It answers critical business questions:

        - **Where do users experience friction?** Identify error-prone prompts, slow generations, and rate-limiting patterns
        - **What drives session costs?** Understand the relationship between prompt complexity, latency, and compute costs
        - **Which users are at risk of churning?** Correlate friction scores with user retention patterns
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

    with cols[0].container(border=True):
        st.markdown("**DiffusionDB Dataset**")
        st.markdown("""
        [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb) is a large-scale
        text-to-image prompt dataset containing **14 million images** generated by Stable Diffusion.
        For this project, we use a **10K sample** to demonstrate the analytics pipeline.

        | Field | Type | Description |
        |-------|------|-------------|
        | `prompt` | text | The text prompt used for generation |
        | `seed` | int | Random seed for reproducibility |
        | `cfg` | float | Classifier-free guidance scale |
        | `sampler` | string | Sampling method (DDIM, etc.) |
        | `width/height` | int | Image dimensions in pixels |
        """)

    with cols[1].container(border=True):
        st.markdown("**Sample Image**")
        if IMAGES_PATH.exists():
            images = list(IMAGES_PATH.glob("*.png"))[:20]
            if images:
                selected_img = random.choice(images)
                st.image(str(selected_img), use_container_width=True)
                st.caption("Random sample from DiffusionDB")

    ""  # spacing

    # Simulation Section
    cols = st.columns(2)

    with cols[0].container(border=True):
        st.markdown("**Simulated Users**")
        st.markdown("""
        **500 synthetic users** with realistic tier distribution:

        | Tier | Distribution | Behavior Pattern |
        |------|-------------|------------------|
        | Free | 70% | Casual, rate-limited |
        | Pro | 25% | Regular, 3x activity |
        | Enterprise | 5% | Power users, 8x activity |

        Users sign up throughout December with exponential growth.
        """)

    with cols[1].container(border=True):
        st.markdown("**Simulated Telemetry**")
        st.markdown("""
        Each generation is enriched with realistic telemetry:

        | Field | Simulation Logic |
        |-------|-----------------|
        | `status` | Based on prompt content & user tier |
        | `latency_ms` | ~50ms per token + log-normal noise |
        | `cost_credits` | Token count + compute time |
        | `feedback` | 15% of users leave thumbs up/down |
        """)

    ""  # spacing

    # ML Enrichment Section
    cols = st.columns(2)

    with cols[0].container(border=True):
        st.markdown("**LLM Prompt Analysis**")
        st.markdown("""
        **Model:** Qwen2.5-1.5B-Instruct via MLX

        Pre-computed extraction for each prompt:
        - `llm_domain` — portrait, character, animal, environment, etc.
        - `llm_art_style` — photography, digital art, oil painting, etc.
        - `llm_complexity_score` — 1-5 based on detail level
        """)

    with cols[1].container(border=True):
        st.markdown("**Text Embeddings**")
        st.markdown("""
        **Model:** all-MiniLM-L6-v2 (sentence-transformers)

        - 384-dimensional vectors stored in DuckDB
        - HNSW index for fast similarity search
        - Enables semantic search in Session Explorer
        - ~2ms query time for 10K vectors
        """)

    ""  # spacing

    # Tech Stack Section
    """
    ## Tech Stack
    """

    with st.container(border=True):
        tech_cols = st.columns(6)
        tech_stack = [
            ("https://cdn.simpleicons.org/duckdb/FFF000", "DuckDB", "Storage"),
            (str(Path(__file__).parent / "assets" / "dagster.png"), "Dagster", "Orchestration"),
            (str(Path(__file__).parent / "assets" / "dbt.png"), "dbt", "Transforms"),
            ("https://cdn.simpleicons.org/apple/FFFFFF", "MLX", "ML Inference"),
            ("https://cdn.simpleicons.org/huggingface/FFD21E", "HuggingFace", "Embeddings"),
            ("https://cdn.simpleicons.org/streamlit/FF4B4B", "Streamlit", "Dashboard"),
        ]
        for col, (logo_url, name, layer) in zip(tech_cols, tech_stack):
            with col:
                st.image(logo_url, width=40)
                st.markdown(f"**{name}**")
                st.caption(layer)

    ""  # spacing

    # System Architecture
    arch_image = Path(__file__).parent / "assets" / "system_architecture.png"
    if arch_image.exists():
        """
        ## System Architecture
        """
        with st.container(border=True):
            st.image(str(arch_image), use_container_width=True)

    ""  # spacing

    # Schema Section
    """
    ## Data Model
    """

    with st.container(border=True):
        st.code("""
    dim_users              dim_prompts                    fct_generations
    ─────────────          ─────────────────              ─────────────────
    user_id (PK)           prompt_id (PK)                 generation_id (PK)
    user_tier              prompt_text                    user_id (FK)
    signup_date            token_count                    prompt_id (FK)
    region                 llm_domain                     session_id
    device_type            llm_art_style                  timestamp
                           llm_complexity_score           latency_ms
                           text_embedding [384]           status
                           image_path                     cost_credits
                                                          feedback
                                    │
                                    ▼
                              fct_sessions
                              ────────────────
                              session_id (PK)
                              user_id (FK)
                              friction_score
                              success_rate_pct
                              total_cost_credits
                              churned_after
        """, language=None)


# =============================================================================
# PAGE: Analytics
# =============================================================================
elif page == "📈 Analytics":
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
                ROUND(AVG(success_rate_pct), 1) as avg_success_rate
            FROM fct_sessions
        """).iloc[0]

        cols = st.columns([1, 2])

        with cols[0].container(border=True):
            st.markdown("**Key Metrics**")
            ""
            metric_cols = st.columns(2)
            metric_cols[0].metric("Users", f"{metrics['total_users']:,}")
            metric_cols[1].metric("Sessions", f"{metrics['total_sessions']:,}")
            ""
            metric_cols = st.columns(2)
            metric_cols[0].metric("Generations", f"{int(metrics['total_generations']):,}")
            metric_cols[1].metric("Success Rate", f"{metrics['avg_success_rate']}%")
            ""
            metric_cols = st.columns(2)
            metric_cols[0].metric("Avg Friction", f"{metrics['avg_friction']}")
            metric_cols[1].metric("Revenue", f"${metrics['total_cost']:,.0f}")

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
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="", yaxis_title="",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        ""  # spacing

        # Status + Friction Row
        col1, col2 = st.columns(2)

        with col1.container(border=True):
            st.markdown("**Status Distribution**")
            status_dist = query("""
                SELECT status, COUNT(*) as count
                FROM raw_generations
                GROUP BY status
                ORDER BY count DESC
            """)
            if not status_dist.empty:
                colors = {
                    'success': CHART_COLORS['success'],
                    'timeout': CHART_COLORS['danger'],
                    'safety_violation': '#a78bfa',
                    'rate_limited': CHART_COLORS['warning'],
                    'model_error': '#fb7185'
                }
                fig = px.pie(
                    status_dist, values='count', names='status',
                    color='status', color_discrete_map=colors,
                    hole=0.5
                )
                fig.update_traces(
                    textposition='outside',
                    textinfo='percent+label',
                    textfont_size=11
                )
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2.container(border=True):
            st.markdown("**Friction by User Tier**")
            friction_by_tier = query("""
                SELECT
                    u.user_tier,
                    ROUND(AVG(s.friction_score), 2) as avg_friction,
                    ROUND(AVG(s.success_rate_pct), 1) as success_rate
                FROM fct_sessions s
                JOIN dim_users u USING (user_id)
                GROUP BY u.user_tier
                ORDER BY avg_friction DESC
            """)
            if not friction_by_tier.empty:
                fig = px.bar(
                    friction_by_tier, x='user_tier', y='avg_friction',
                    color='avg_friction',
                    color_continuous_scale=[[0, CHART_COLORS['success']], [0.5, CHART_COLORS['warning']], [1, CHART_COLORS['danger']]]
                )
                fig.update_traces(marker_line_width=0)
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="", yaxis_title="",
                    showlegend=False, coloraxis_showscale=False,
                    bargap=0.4
                )
                st.plotly_chart(fig, use_container_width=True)

        ""  # spacing

        # Weekly Performance
        """
        ## Weekly Performance
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
                ROUND(SUM(cost_credits), 2) as revenue
            FROM raw_generations
            GROUP BY week
            ORDER BY week
        """)

        if not weekly.empty:
            cols = st.columns(3)
            with cols[0].container(border=True):
                st.markdown("**Generations**")
                fig = px.bar(weekly, x='week', y='generations',
                            color_discrete_sequence=[CHART_COLORS['primary']])
                fig.update_traces(marker_line_width=0)
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0),
                                showlegend=False, bargap=0.4,
                                xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            with cols[1].container(border=True):
                st.markdown("**Active Users**")
                fig = px.bar(weekly, x='week', y='active_users',
                            color_discrete_sequence=[CHART_COLORS['secondary']])
                fig.update_traces(marker_line_width=0)
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0),
                                showlegend=False, bargap=0.4,
                                xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            with cols[2].container(border=True):
                st.markdown("**Revenue ($)**")
                fig = px.bar(weekly, x='week', y='revenue',
                            color_discrete_sequence=[CHART_COLORS['success']])
                fig.update_traces(marker_line_width=0)
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0),
                                showlegend=False, bargap=0.4,
                                xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

        ""  # spacing

        # Friction Analysis Section
        """
        ## Friction Analysis
        """

        col1, col2 = st.columns(2)

        with col1.container(border=True):
            st.markdown("**Friction by Tier (Summary)**")
            tier_data = query("""
                SELECT * FROM user_friction_summary
                ORDER BY avg_friction_score DESC
            """)

            fig = px.bar(
                tier_data,
                x="user_tier",
                y="avg_friction_score",
                color="user_tier",
                color_discrete_map={
                    "free": CHART_COLORS['danger'],
                    "pro": CHART_COLORS['warning'],
                    "enterprise": CHART_COLORS['success']
                },
                text="avg_friction_score",
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="",
                            height=300, bargap=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with col2.container(border=True):
            st.markdown("**Friction Distribution**")
            friction_dist = query("""
                SELECT friction_category, COUNT(*) as sessions
                FROM fct_sessions
                GROUP BY friction_category
            """)

            fig = px.pie(
                friction_dist,
                values="sessions",
                names="friction_category",
                color="friction_category",
                color_discrete_map={
                    "low": CHART_COLORS['success'],
                    "medium": CHART_COLORS['warning'],
                    "high": CHART_COLORS['danger']
                },
                hole=0.5
            )
            fig.update_traces(textposition='outside', textinfo='percent+label', textfont_size=11)
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        ""  # spacing

        # Daily Trends
        """
        ## Daily Trends
        """

        daily = query("SELECT * FROM daily_metrics ORDER BY session_date")

        tab1, tab2, tab3 = st.tabs(["📉 Friction & Success", "📊 Volume & Cost", "👍 Engagement"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["session_date"], y=daily["avg_friction_score"],
                name="Friction Score",
                line=dict(color=CHART_COLORS['danger'], width=2.5),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            fig.add_trace(go.Scatter(
                x=daily["session_date"], y=daily["avg_success_rate_pct"],
                name="Success Rate %",
                line=dict(color=CHART_COLORS['success'], width=2.5),
                yaxis="y2"
            ))
            fig.update_layout(
                yaxis=dict(title="Friction Score", side="left", gridcolor='rgba(255,255,255,0.05)'),
                yaxis2=dict(title="Success Rate %", side="right", overlaying="y", gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor='rgba(0,0,0,0)'),
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily["session_date"], y=daily["total_generations"],
                name="Generations",
                marker_color=CHART_COLORS['info'],
                marker_line_width=0,
                opacity=0.8
            ))
            fig.add_trace(go.Scatter(
                x=daily["session_date"], y=daily["total_cost_credits"],
                name="Cost ($)",
                line=dict(color=CHART_COLORS['warning'], width=2.5),
                yaxis="y2"
            ))
            fig.update_layout(
                yaxis=dict(title="Generations", side="left", gridcolor='rgba(255,255,255,0.05)'),
                yaxis2=dict(title="Cost ($)", side="right", overlaying="y", gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor='rgba(0,0,0,0)'),
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["session_date"], y=daily["thumbs_up"],
                name="Thumbs Up",
                line=dict(color=CHART_COLORS['success'], width=2.5),
                mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=daily["session_date"], y=daily["thumbs_down"],
                name="Thumbs Down",
                line=dict(color=CHART_COLORS['danger'], width=2.5),
                mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=daily["session_date"], y=daily["downloads"],
                name="Downloads",
                line=dict(color=CHART_COLORS['info'], width=2.5),
                mode='lines'
            ))
            fig.update_layout(
                yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor='rgba(0,0,0,0)'),
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Tier Comparison
        st.markdown("### User Tier Comparison")

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

        # Session Explorer
        st.markdown("### Session Explorer")

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
elif page == "🤖 SQL Copilot":
    st.title("🤖 SQL Copilot")
    st.markdown("Ask questions in natural language, get SQL queries powered by local LLM")

    st.markdown("---")

    # How it works
    with st.expander("How it works", expanded=False):
        st.markdown("""
        1. **You ask** a question in plain English
        2. **LLM generates** SQL using your schema context
        3. **You review** and optionally edit the SQL
        4. **Execute** against DuckDB and see results

        **Model:** Qwen2.5-1.5B-Instruct (runs locally via MLX)

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

        user_question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., What is the average latency by user tier?"
        )

        if user_question:
            with st.spinner("Generating SQL..."):
                try:
                    model, tokenizer, sampler = get_copilot_model()
                    schema_text = get_schema_text()
                    prompt = construct_sql_prompt(user_question, schema_text)
                    generated_sql = generate_sql(prompt, model, tokenizer, sampler)
                except Exception as e:
                    st.error(f"Model error: {e}")
                    generated_sql = ""

            if generated_sql:
                st.markdown("#### Generated SQL")
                edited_sql = st.text_area(
                    "Review and edit if needed:",
                    value=generated_sql,
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
elif page == "🔍 Session Explorer":
    st.title("🔍 Session Explorer")
    st.markdown("Semantic search + LLM-generated filters + image preview")

    st.markdown("---")

    # Check if we have embeddings and images (from prompt_enrichments table)
    has_embeddings = False
    has_images = False
    try:
        embedding_check = query("""
            SELECT
                COUNT(*) FILTER (WHERE text_embedding IS NOT NULL) as embeddings,
                COUNT(*) FILTER (WHERE image_path IS NOT NULL) as images
            FROM prompt_enrichments
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

            # Get filter options from LLM-generated labels (from prompt_enrichments)
            @st.cache_data
            def get_filter_options():
                try:
                    domains = query("SELECT DISTINCT llm_domain FROM prompt_enrichments WHERE llm_domain IS NOT NULL ORDER BY llm_domain")
                    styles = query("SELECT DISTINCT llm_art_style FROM prompt_enrichments WHERE llm_art_style IS NOT NULL ORDER BY llm_art_style")
                    statuses = query("SELECT DISTINCT status FROM fct_generations ORDER BY status")
                    tiers = query("SELECT DISTINCT user_tier FROM dim_users ORDER BY user_tier")
                    return {
                        'domains': ['All'] + domains['llm_domain'].tolist(),
                        'styles': ['All'] + styles['llm_art_style'].tolist(),
                        'statuses': ['All'] + statuses['status'].tolist(),
                        'tiers': ['All'] + tiers['user_tier'].tolist()
                    }
                except:
                    return {'domains': ['All'], 'styles': ['All'], 'statuses': ['All'], 'tiers': ['All']}

            filters = get_filter_options()

            # Search input
            search_query = st.text_input(
                "Search prompts:",
                placeholder="e.g., cyberpunk city, fantasy castle, portrait of a woman"
            )

            # Filter row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                domain_filter = st.selectbox("Domain", filters['domains'])
            with col2:
                style_filter = st.selectbox("Art Style", filters['styles'])
            with col3:
                status_filter = st.selectbox("Status", filters['statuses'])
            with col4:
                tier_filter = st.selectbox("User Tier", filters['tiers'])

            st.markdown("---")

            if search_query:
                with st.spinner("Searching..."):
                    model = get_embedding_model()
                    query_embedding = model.encode([search_query])[0].tolist()

                    # Build WHERE clause from filters
                    where_clauses = []
                    if domain_filter != 'All':
                        where_clauses.append(f"e.llm_domain = '{domain_filter}'")
                    if style_filter != 'All':
                        where_clauses.append(f"e.llm_art_style = '{style_filter}'")
                    if status_filter != 'All':
                        where_clauses.append(f"g.status = '{status_filter}'")
                    if tier_filter != 'All':
                        where_clauses.append(f"u.user_tier = '{tier_filter}'")

                    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

                    results = query(f"""
                        SELECT
                            p.prompt_id,
                            p.prompt_text,
                            e.image_path,
                            e.llm_domain,
                            e.llm_art_style,
                            e.llm_complexity_score,
                            u.user_tier,
                            g.latency_ms,
                            g.status,
                            array_cosine_similarity(e.text_embedding, {query_embedding}::FLOAT[384]) AS similarity
                        FROM dim_prompts p
                        JOIN prompt_enrichments e ON p.prompt_id = e.prompt_id
                        JOIN fct_generations g ON p.prompt_id = g.prompt_id
                        JOIN dim_users u ON g.user_id = u.user_id
                        WHERE e.text_embedding IS NOT NULL AND {where_sql}
                        ORDER BY similarity DESC
                        LIMIT 12
                    """)

                if len(results) == 0:
                    st.warning("No results found. Try adjusting filters or search query.")
                else:
                    # Cluster insights
                    st.subheader(f"Results for '{search_query}'")

                    col1, col2, col3, col4 = st.columns(4)
                    error_rate = (results['status'] != 'success').mean()
                    avg_latency = results['latency_ms'].mean()
                    top_style = results['llm_art_style'].mode().iloc[0] if len(results) > 0 and not results['llm_art_style'].mode().empty else "N/A"
                    top_domain = results['llm_domain'].mode().iloc[0] if len(results) > 0 and not results['llm_domain'].mode().empty else "N/A"

                    col1.metric("Error Rate", f"{error_rate:.1%}")
                    col2.metric("Avg Latency", f"{avg_latency:,.0f}ms")
                    col3.metric("Top Style", top_style)
                    col4.metric("Top Domain", top_domain)

                    st.markdown("---")

                    # Image grid (4 columns)
                    if has_images:
                        cols = st.columns(4)
                        for idx, row in results.iterrows():
                            with cols[idx % 4]:
                                # Load image from blob storage
                                image_path = PROJECT_ROOT / row['image_path'] if row['image_path'] else None
                                full_prompt = row['prompt_text'] or ''

                                # Status emoji
                                status_emoji = {
                                    'success': '✅', 'timeout': '⏱️', 'safety_violation': '🛑',
                                    'rate_limited': '⚠️', 'model_error': '❌'
                                }.get(row['status'], '❓')

                                if image_path and image_path.exists():
                                    # Use popover for full prompt on click
                                    with st.popover(f"🔍 {row['prompt_text'][:30]}..."):
                                        st.markdown("**Full Prompt:**")
                                        st.write(full_prompt)
                                        st.markdown(f"**Similarity:** {row['similarity']:.3f}")
                                        st.markdown(f"**Status:** {status_emoji} {row['status']}")
                                        st.markdown(f"**Latency:** {row['latency_ms']:,}ms")
                                        st.markdown(f"**Style:** {row['llm_art_style'] or 'unknown'}")
                                        st.markdown(f"**Domain:** {row['llm_domain'] or 'unknown'}")
                                    st.image(str(image_path), use_container_width=True)
                                else:
                                    # Placeholder for missing images
                                    st.markdown(
                                        f"""<div style="background:#2d2d2d;height:150px;display:flex;
                                        align-items:center;justify-content:center;border-radius:8px;
                                        color:#888;font-size:12px;">No image</div>""",
                                        unsafe_allow_html=True
                                    )

                                st.caption(f"**{row['prompt_text'][:40]}...**")
                                st.caption(f"Sim: {row['similarity']:.2f} | {status_emoji} {row['latency_ms']:,}ms")

                    else:
                        # No images - show table instead
                        st.info("No images downloaded yet. Run: `uv run python -m src.ingestion.download_images`")

                        display_df = results[['prompt_text', 'llm_domain', 'llm_art_style', 'status', 'latency_ms', 'similarity']].copy()
                        display_df.columns = ['Prompt', 'Domain', 'Style', 'Status', 'Latency (ms)', 'Similarity']
                        display_df['Similarity'] = display_df['Similarity'].round(3)
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

            else:
                st.info("Enter a search query to find similar prompts")

                # Show some example queries
                st.markdown("**Example searches:**")
                examples = ["cyberpunk city at night", "fantasy landscape", "portrait of a woman", "anime character", "oil painting style"]
                cols = st.columns(len(examples))
                for col, example in zip(cols, examples):
                    with col:
                        st.code(example, language=None)

        except ImportError:
            st.warning("sentence-transformers not installed. Run: `uv add sentence-transformers`")


# =============================================================================
# Footer
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Links**")
st.sidebar.markdown("[GitHub](https://github.com) | [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb)")
