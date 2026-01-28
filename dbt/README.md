# dbt Models - GenAI Diffusion Lens

## Naming Conventions

| Prefix | Type | Description | Managed By |
|--------|------|-------------|------------|
| `raw_*` | Table | Source data tables | Python scripts |
| `stg_*` | View | Staging transformations | dbt |
| `dim_*` | Table | Dimension tables | dbt |
| `fct_*` | Table | Fact tables | dbt |
| `ftr_*` | View | Feature views | dbt |

## Table Lineage

```
RAW (Python Scripts)
====================

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐
│   raw_prompts    │  │    raw_users     │  │  raw_generations │  │ raw_prompt_enrichments  │
│                  │  │                  │  │                  │  │                         │
│ diffusiondb_     │  │ user_generator   │  │ telemetry_       │  │ precompute_llm.py       │
│ loader.py        │  │ .py              │  │ enricher.py      │  │ precompute_embeddings.py│
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │ link_images.py          │
         │                     │                     │            └────────────┬────────────┘
         │                     │                     │                         │
         ▼                     ▼                     ▼                         │
                                                                               │
STG (dbt views)                                                                │
===============                                                                │
                                                                               │
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐               │
│   stg_prompts    │  │    stg_users     │  │  stg_generations │               │
│                  │  │                  │  │                  │               │
│ • clean text     │  │ • validate tiers │  │ • clean status   │               │
│ • add token_cnt  │  │ • format dates   │  │ • cast types     │               │
└────────┬─────────┘  └────────┬─────────┘  └─────────┬────────┘               │
         │                     │                      │                        │
         │                     │                      │                        │
         ▼                     ▼                      ▼                        │
                                                                               │
DIM & FCT (dbt tables)                                                         │
======================                                                         │
                                                                               │
┌──────────────────┐  ┌──────────────────┐                                     │
│   dim_prompts    │  │    dim_users     │                                     │
│                  │  │                  │                                     │
│ • prompt_id (PK) │  │ • user_id (PK)   │                                     │
│ • prompt_text    │  │ • user_tier      │                                     │
│ • token_count    │  │ • lifetime_cost  │                                     │
│ • cfg, sampler   │  │ • total_sessions │                                     │
└────────┬─────────┘  └────────┬─────────┘                                     │
         │                     │                                               │
         │   ┌─────────────────┘                                               │
         │   │                                                                 │
         ▼   ▼                                                                 │
┌──────────────────────────────┐                                               │
│      fct_generations         │                                               │
│                              │                                               │
│ • generation_id (PK)         │                                               │
│ • user_id (FK → dim_users)   │                                               │
│ • prompt_id (FK → dim_prompts)                                               │
│ • latency_ms, status, cost   │                                               │
└──────────────┬───────────────┘                                               │
               │                                                               │
               ▼                                                               │
┌──────────────────────────────┐                                               │
│       fct_sessions           │                                               │
│                              │                                               │
│ • session_id (PK)            │                                               │
│ • friction_score             │                                               │
│ • success_rate_pct           │                                               │
│ • total_cost_credits         │                                               │
└──────────────┬───────────────┘                                               │
               │                                                               │
               │                                                               │
         ┌─────┴─────┐                                                         │
         ▼           ▼                                                         │
                                                                               │
METRICS (dbt tables)                                                           │
====================                                                           │
                                                                               │
┌──────────────────┐  ┌──────────────────┐                                     │
│  daily_metrics   │  │user_friction_    │                                     │
│                  │  │    summary       │                                     │
│ • daily KPIs     │  │ • by user_tier   │                                     │
└──────────────────┘  └──────────────────┘                                     │
                                                                               │
                                                                               │
FTR (dbt views) ◄──────────────────────────────────────────────────────────────┘
================

┌─────────────────────────────────────────────────────────────┐
│                     ftr_llm_analysis                        │
│                                                             │
│  SELECT p.*, e.llm_domain, e.llm_art_style, e.image_path   │
│  FROM dim_prompts p                                         │
│  LEFT JOIN raw_prompt_enrichments e ON p.prompt_id = e...   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   ftr_text_embeddings                       │
│                                                             │
│  SELECT p.*, e.text_embedding                               │
│  FROM dim_prompts p                                         │
│  LEFT JOIN raw_prompt_enrichments e ON p.prompt_id = e...   │
└─────────────────────────────────────────────────────────────┘
```

## Dependency Summary

| Model | Depends On |
|-------|------------|
| `stg_prompts` | `raw_prompts` |
| `stg_users` | `raw_users` |
| `stg_generations` | `raw_generations` |
| `dim_prompts` | `stg_prompts` |
| `dim_users` | `stg_users`, `stg_generations` |
| `fct_generations` | `stg_generations`, `dim_users`, `dim_prompts` |
| `fct_sessions` | `fct_generations` |
| `ftr_llm_analysis` | `dim_prompts`, `raw_prompt_enrichments` |
| `ftr_text_embeddings` | `dim_prompts`, `raw_prompt_enrichments` |
| `daily_metrics` | `fct_sessions` |
| `user_friction_summary` | `fct_sessions` |

## Why raw_prompt_enrichments is separate

The `raw_prompt_enrichments` table stores ML-computed features:
- LLM analysis (domain, art_style, complexity_score)
- Text embeddings (384-dim vectors)
- Image paths

**Key design decision:** This table is NOT managed by dbt because:
1. ML enrichment takes ~50 minutes to compute
2. We don't want `dbt run` to wipe out expensive computations
3. Feature views (`ftr_*`) just JOIN with it - read only, never write

## Running dbt

```bash
# From dbt directory
cd dbt && uv run dbt run

# Specific models
cd dbt && uv run dbt run --select ftr_llm_analysis ftr_text_embeddings

# Full refresh
cd dbt && uv run dbt run --full-refresh
```

## Model Materialization

| Folder | Materialization | Reason |
|--------|-----------------|--------|
| staging/ | view | Lightweight, always fresh |
| marts/ | table | Performance for dashboard queries |
| features/ | view | Joins with external table, always fresh |
| metrics/ | table | Pre-aggregated for fast dashboard loads |
