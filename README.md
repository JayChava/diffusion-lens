# GenAI Session Analyzer

A local data platform that analyzes user friction and session costs in GenAI image generation workflows. Built as a portfolio project demonstrating data engineering and analytics skills.

## What It Does

Ingests real prompt data from DiffusionDB (2M prompts), enriches it with simulated telemetry (latency, errors, feedback, downloads), and exposes a dashboard showing:

- **Friction Score** - Weighted metric of errors, latency, and retries
- **Session Costs** - Credit consumption by user tier
- **Engagement Signals** - Thumbs up/down, download rates

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **Ingestion** | HuggingFace `datasets` | Stream DiffusionDB prompts |
| **Storage** | DuckDB | Local OLAP, zero config |
| **Orchestration** | Dagster | Asset-based pipeline |
| **Transformation** | dbt-duckdb | Star schema modeling |
| **Simulation** | Faker + NumPy | Realistic synthetic telemetry |
| **Dashboard** | Streamlit + Plotly | Interactive analytics |
| **Environment** | uv | Fast Python packaging |

## Quick Start

```bash
# 1. Install dependencies
cd genai-session-analyzer
uv sync

# 2. Run the data pipeline (loads 10K prompts, generates users, enriches telemetry)
uv run python -m src.simulation.run_all

# 3. Build dbt models (star schema)
uv run dbt run --profiles-dir dbt --project-dir dbt

# 4. Launch dashboard
uv run streamlit run dashboard/app.py
```

Or use Dagster UI for visual pipeline control:
```bash
uv run dagster dev -m src.pipeline.definitions
# Open http://localhost:3000 → Materialize all
```

## Project Structure

```
genai-session-analyzer/
├── src/
│   ├── ingestion/
│   │   └── diffusiondb_loader.py    # HuggingFace → DuckDB
│   ├── simulation/
│   │   ├── user_generator.py        # Synthetic users (free/pro/enterprise)
│   │   ├── telemetry_enricher.py    # Latency, status, feedback, downloads
│   │   ├── session_builder.py       # 30-min session windowing
│   │   └── run_all.py               # One-command pipeline
│   └── pipeline/
│       ├── assets.py                # Dagster asset definitions
│       └── definitions.py           # Dagster entry point
├── dbt/
│   ├── models/
│   │   ├── staging/                 # stg_users, stg_prompts, stg_generations
│   │   ├── marts/                   # dim_users, dim_prompts, fct_generations, fct_sessions
│   │   └── metrics/                 # user_friction_summary, daily_metrics
│   ├── dbt_project.yml
│   └── profiles.yml
├── dashboard/
│   └── app.py                       # Streamlit dashboard
├── data/
│   └── warehouse.duckdb             # Local analytics database
└── src/
    └── explore.ipynb                # Interactive data exploration
```

## Data Model

### Star Schema

```
              ┌─────────────┐
              │  dim_users  │
              │─────────────│
              │ user_id     │
              │ user_tier   │
              │ region      │
              │ lifetime_*  │
              └──────┬──────┘
                     │
┌─────────────┐      │      ┌─────────────────────┐
│ dim_prompts │      │      │   fct_generations   │
│─────────────│      │      │─────────────────────│
│ prompt_id   │──────┼──────│ generation_id       │
│ prompt_text │      │      │ user_id (FK)        │
│ complexity  │      └──────│ prompt_id (FK)      │
│ token_count │             │ session_id          │
└─────────────┘             │ latency_ms, status  │
                            │ cost_credits        │
                            │ feedback, downloaded│
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │    fct_sessions     │
                            │─────────────────────│
                            │ session_id          │
                            │ friction_score      │
                            │ friction_category   │
                            │ success_rate_pct    │
                            │ total_cost_credits  │
                            └─────────────────────┘
```

### Friction Score Formula

```
friction_score = (
    (error_rate × 3.0 × 33.33) +
    (latency_normalized × 2.0 × 33.33) +
    (retry_rate × 1.0 × 33.33)
)
```

- **error_rate**: 1 - success_rate (0-1)
- **latency_normalized**: avg_latency_ms / 5000 (capped at 1)
- **retry_rate**: avg_retries / 2 (capped at 1)

### Simulated Telemetry Logic

| Signal | Logic |
|--------|-------|
| **Status** | NSFW keywords → 85% safety_violation; >75 tokens → 15% timeout; free tier → 8% rate_limited |
| **Latency** | ~50ms/token + log-normal noise; timeout = 30s; safety rejection = 100-500ms |
| **Feedback** | 10% free, 20% pro, 25% enterprise leave feedback; success → 80% thumbs up |
| **Download** | Only on success; 40% free, 65% pro, 80% enterprise base rate |

## Key Metrics (December 2025 Dataset)

| Metric | Value |
|--------|-------|
| Total Users | 500 |
| Total Sessions | 9,748 |
| Total Generations | 10,000 |
| Overall Success Rate | 94.1% |
| Avg Friction Score | 24.8 |

### By User Tier

| Tier | Users | Avg Friction | Success Rate | Download Rate | Feedback Rate |
|------|-------|--------------|--------------|---------------|---------------|
| Free | 342 | 28.3 | 89.7% | 42.8% | 10.7% |
| Pro | 130 | 23.1 | 96.7% | 67.5% | 19.4% |
| Enterprise | 28 | 23.1 | 96.5% | 83.6% | 24.1% |

## Demo Script

### 1. Show the Pipeline (30 sec)
```bash
uv run dagster dev -m src.pipeline.definitions
# Open localhost:3000, show asset graph
```

### 2. Raw → Transformed (2 min)
```bash
uv run python -c "
import duckdb
conn = duckdb.connect('data/warehouse.duckdb')
print(conn.execute('SELECT * FROM fct_sessions LIMIT 5').fetchdf())
"
```

### 3. Dashboard Walkthrough (3 min)
```bash
uv run streamlit run dashboard/app.py
```
- Friction by tier (free users struggle more)
- Daily trends (stable ops)
- Session explorer (filter high-friction sessions)

### 4. Live Query (2 min)
```sql
-- Do power users churn less?
SELECT
    user_tier,
    AVG(friction_score) as avg_friction,
    COUNT(*) as sessions
FROM fct_sessions
GROUP BY user_tier
ORDER BY avg_friction;
```

### 5. Design Decisions (3 min)
- **Why DuckDB?** Zero-config OLAP, perfect for local analytics
- **Why star schema?** Separates dimensions from facts, enables flexible querying
- **Why Dagster over Airflow?** Asset-based (not task-based), better for analytics pipelines
- **Simulation realism** - Status derived from prompt content, not random

## Development

```bash
# Run pipeline with custom size
uv run python -m src.simulation.run_all --prompts 100000 --users 2000

# Rebuild dbt models
uv run dbt run --profiles-dir dbt --project-dir dbt

# Explore data interactively
uv run jupyter notebook src/explore.ipynb

# Format code
uv run ruff format .
```

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                   │
└──────────────────────────────────────────────────────────────────────┘

  HuggingFace                                              Streamlit
  DiffusionDB                                              Dashboard
      │                                                        ▲
      ▼                                                        │
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┤
│ Ingestion│───▶│Simulation│───▶│  Dagster │───▶│     dbt      │
│  Loader  │    │ Enricher │    │  Assets  │    │  Transform   │
└──────────┘    └──────────┘    └──────────┘    └──────────────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │   DuckDB     │
                               │  warehouse   │
                               └──────────────┘
```

---

*Built for Luma AI Data Science Interview | January 2025*
