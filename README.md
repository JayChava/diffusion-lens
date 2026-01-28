# Diffusion Lens

A local data platform analyzing user friction and session costs in GenAI image generation workflows. Features a SQL Copilot (local LLM) and semantic search over 10K prompts.

## Features

- **Friction Analytics** - Weighted metric combining errors, latency, and retries
- **SQL Copilot** - Natural language → SQL using local Qwen2.5-7B via MLX
- **Session Explorer** - Semantic search over prompts with vector similarity
- **Full Pipeline** - Dagster orchestration, dbt transformations, 22 data quality tests

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| **Storage** | DuckDB | OLAP-optimized, zero config, native vector search |
| **Orchestration** | Dagster | Asset-based pipeline, beautiful UI |
| **Transformation** | dbt-duckdb | Star schema, testable SQL models |
| **ML Enrichment** | MLX + sentence-transformers | Local LLM + embeddings on Apple Silicon |
| **Dashboard** | Streamlit | Interactive analytics + SQL Copilot |
| **Environment** | uv | Fast Python packaging |

## Quick Start

```bash
# Clone and install
cd genai-session-analyzer
uv sync

# Option 1: Demo scripts (recommended)
./demo/dagster.sh      # Start Dagster UI at :3000
# Click "Materialize all" in UI, then:
./demo/build_dbt.sh    # Build dbt models
./demo/streamlit.sh    # Start dashboard at :8501

# Option 2: Manual
uv run dagster dev -m src.pipeline.definitions
# Materialize assets, then:
cd dbt && uv run dbt build && cd ..
uv run streamlit run dashboard/app.py
```

## Project Structure

```
genai-session-analyzer/
├── src/
│   ├── ingestion/
│   │   ├── diffusiondb_loader.py  # HuggingFace → DuckDB
│   │   ├── download_images.py     # Download images to blob storage
│   │   └── link_images.py         # Link images to enrichment table
│   ├── simulation/
│   │   ├── user_generator.py      # Synthetic users (free/pro/enterprise)
│   │   ├── telemetry_enricher.py  # Latency, status, feedback, downloads
│   │   └── session_builder.py     # Session assignment
│   ├── enrichment/
│   │   ├── precompute_embeddings.py  # Text embeddings (all-MiniLM-L6-v2)
│   │   └── precompute_llm.py         # LLM analysis (Qwen2.5 via MLX)
│   ├── copilot/
│   │   ├── llm.py                 # SQL generation with MLX
│   │   ├── schema.py              # Schema extraction for LLM
│   │   ├── prompt.py              # Prompt templates
│   │   └── sample.py              # CLI playground
│   ├── explorer/
│   │   └── sample.py              # Semantic search CLI
│   └── pipeline/
│       ├── assets.py              # Dagster asset definitions
│       └── definitions.py         # Dagster entry point
├── dbt/
│   └── models/
│       ├── staging/               # stg_users, stg_prompts, stg_generations
│       ├── marts/                 # dim_users, dim_prompts, fct_generations, fct_sessions
│       ├── features/              # ftr_llm_analysis, ftr_text_embeddings
│       └── metrics/               # user_friction_summary, daily_metrics
├── dashboard/
│   └── app.py                     # Streamlit (1500+ lines, 5 pages)
├── tests/
│   └── test_pipeline.py           # 22 data quality tests
├── demo/
│   ├── dagster.sh                 # Start Dagster with lock check
│   ├── streamlit.sh               # Start dashboard
│   ├── build_dbt.sh               # Rebuild dbt models
│   └── test.sh                    # Run data quality tests
├── docs/
│   ├── DEMO.md                    # Demo walkthrough
│   ├── INTERVIEW_PREP.md          # Q&A for interviews
│   └── LIVE_CODING_PREP.md        # Common modification patterns
└── data/
    ├── warehouse.duckdb           # Analytics database
    └── blob/images/generations/   # 10K images (~5GB)
```

## Data Model

### Star Schema

```
┌─────────────┐     ┌─────────────┐     ┌───────────────────┐
│  dim_users  │     │ dim_prompts │     │  fct_generations  │
├─────────────┤     ├─────────────┤     ├───────────────────┤
│ user_id     │◄────│ prompt_id   │◄────│ generation_id     │
│ user_tier   │     │ prompt_text │     │ user_id (FK)      │
│ region      │     │ token_count │     │ prompt_id (FK)    │
│ device_type │     │             │     │ session_id        │
└─────────────┘     └─────────────┘     │ status, latency_ms│
                                        │ cost_credits      │
                                        └─────────┬─────────┘
                                                  │
                                        ┌─────────▼─────────┐
                                        │   fct_sessions    │
                                        ├───────────────────┤
                                        │ session_id        │
                                        │ friction_score    │
                                        │ success_rate_pct  │
                                        │ total_cost_credits│
                                        └───────────────────┘
```

### Friction Score

```
friction_score = (error_rate × 50) + (latency_norm × 33) + (retry_norm × 17)
```

- **error_rate**: `1 - success_rate` (0-1)
- **latency_norm**: `avg_latency_ms / 5000`, capped at 1
- **retry_norm**: `avg_retries / 2`, capped at 1

Score is 0-100. Higher = worse experience.

## Dashboard Pages

| Page | What it shows |
|------|---------------|
| **Overview** | Key metrics, navigation cards |
| **Architecture** | Tech stack, data flow diagram |
| **Analytics** | Friction by tier, time series, quality by segment |
| **Session Explorer** | Semantic search with vector similarity |
| **SQL Copilot** | Natural language → SQL with local LLM |

## SQL Copilot

Ask questions in natural language, get SQL:

```bash
# CLI playground
uv run python -m src.copilot.sample
```

Examples:
- "Show error rate by user tier"
- "Top 10 users by session cost"
- "Average latency by model version"

**Model:** Qwen2.5-7B-Instruct-4bit via MLX (~15s generation)

## Session Explorer

Semantic search over prompts:

```bash
# CLI playground
uv run python -m src.explorer.sample "cyberpunk city"
```

**Model:** all-MiniLM-L6-v2 (384-dim embeddings) + DuckDB HNSW index

## Testing

```bash
# Run all 22 data quality tests
./demo/test.sh

# Or directly
uv run pytest tests/test_pipeline.py -v
```

Tests include:
- Table existence
- Row counts and data integrity
- Valid values (tiers, statuses)
- Business logic (success rate 50-99%, friction 0-100)
- No orphan records

## Development

```bash
# Run pipeline with custom size
uv run python -m src.simulation.run_all --prompts 1000 --users 100

# Rebuild dbt models
./demo/build_dbt.sh

# Run ML enrichment (takes ~1hr for 10K)
uv run python -m src.enrichment.precompute_embeddings
uv run python -m src.enrichment.precompute_llm --limit 100

# Format code
uv run ruff format .
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                │
└─────────────────────────────────────────────────────────────────┘

  HuggingFace        Simulation         Dagster           dbt
  DiffusionDB        (Faker/NumPy)      Assets            Transform
      │                   │                │                  │
      ▼                   ▼                ▼                  ▼
┌──────────┐       ┌──────────┐      ┌──────────┐      ┌──────────┐
│raw_prompts│──────│raw_users │──────│raw_gens  │──────│ dim/fct  │
│  (10K)   │       │  (500)   │      │(telemetry)│     │  tables  │
└──────────┘       └──────────┘      └──────────┘      └──────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │   DuckDB     │
                                    │  warehouse   │
                                    └──────────────┘
                                           │
                   ┌───────────────────────┼───────────────────────┐
                   ▼                       ▼                       ▼
            ┌──────────┐            ┌──────────┐            ┌──────────┐
            │ Dashboard │            │SQL Copilot│            │ Explorer │
            │(Streamlit)│            │  (MLX)   │            │(Embeddings)│
            └──────────┘            └──────────┘            └──────────┘
```

## Production Migration Path

| Local | Production |
|-------|------------|
| DuckDB | MotherDuck or Snowflake |
| Local blob storage | S3 + CloudFront |
| MLX (Apple Silicon) | vLLM/TGI on GPU or Claude API |
| Dagster dev | Dagster Cloud |
| Streamlit | Evidence or custom React |
