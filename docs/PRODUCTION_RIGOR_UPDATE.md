# Production Rigor Update - Code Breakdown

**Date:** 2026-01-26
**Purpose:** Add data engineering rigor with dbt tests, daily simulation, and backfill framework

---

## Table of Contents

1. [Overview](#overview)
2. [File Changes Summary](#file-changes-summary)
3. [dbt Tests Deep Dive](#dbt-tests-deep-dive)
4. [Daily Simulation Script](#daily-simulation-script)
5. [Backfill Framework](#backfill-framework)
6. [How to Use](#how-to-use)
7. [Architecture Decisions](#architecture-decisions)

---

## Overview

This update adds three production-grade data engineering patterns:

| Feature | Purpose | Interview Talking Point |
|---------|---------|------------------------|
| **dbt Tests** | Data quality validation | "I test my data models like I test code" |
| **Daily Simulation** | Incremental processing demo | "This simulates a production daily batch job" |
| **Backfill Framework** | Adding new metrics | "Here's how we'd add a new KPI and backfill history" |

---

## File Changes Summary

### New Files Created

```
dbt/
├── packages.yml                          # dbt package dependencies
├── models/
│   ├── staging/
│   │   ├── schema.yml                    # Staging layer tests
│   │   └── sources.yml                   # Source definitions + tests
│   ├── marts/
│   │   └── schema.yml                    # Mart layer tests
│   └── metrics/
│       └── schema.yml                    # Metrics layer tests
└── tests/
    ├── assert_no_orphan_generations.sql  # FK integrity
    ├── assert_no_orphan_prompts.sql      # FK integrity
    ├── assert_session_dates_valid.sql    # Time logic
    ├── assert_user_signup_before_activity.sql  # Business rule
    ├── assert_cost_positive_for_success.sql    # Business logic
    ├── assert_friction_score_consistent.sql    # Metric validation
    ├── assert_daily_metrics_complete.sql       # Completeness
    └── assert_tier_distribution_reasonable.sql # Sanity check

scripts/
├── run_daily_simulation.py               # 30-day incremental simulation
└── backfill_metric.py                    # Metric backfill framework
```

### Modified Files

```
dbt/dbt_project.yml                       # Removed features folder config
```

### Removed Files

```
dbt/models/features/                      # Removed (conflicts with Python enrichment)
    ├── ftr_text_embeddings.sql
    ├── ftr_llm_analysis.sql
    └── schema.yml
```

---

## dbt Tests Deep Dive

### Test Types

dbt supports two types of tests:

1. **Schema Tests** - Declared in `schema.yml` files
2. **Data Tests** - Custom SQL in `tests/` folder

### Schema Tests (in schema.yml files)

#### Location: `dbt/models/staging/sources.yml`

```yaml
# Tests on raw source tables
sources:
  - name: raw
    tables:
      - name: raw_users
        columns:
          - name: user_id
            data_tests:
              - unique        # No duplicate user_ids
              - not_null      # user_id is required
          - name: user_tier
            data_tests:
              - accepted_values:
                  values: ['free', 'pro', 'enterprise']
```

**What each test does:**

| Test | SQL Generated | Purpose |
|------|---------------|---------|
| `unique` | `SELECT ... GROUP BY col HAVING COUNT(*) > 1` | Primary key integrity |
| `not_null` | `SELECT ... WHERE col IS NULL` | Required field validation |
| `accepted_values` | `SELECT ... WHERE col NOT IN (...)` | Enum validation |
| `relationships` | `SELECT ... WHERE FK NOT IN (SELECT PK FROM parent)` | Foreign key integrity |
| `dbt_utils.accepted_range` | `SELECT ... WHERE col < min OR col > max` | Numeric bounds |

#### Location: `dbt/models/staging/schema.yml`

```yaml
# Tests on staging models
models:
  - name: stg_generations
    columns:
      - name: generation_id
        data_tests:
          - unique
          - not_null
      - name: prompt_id
        data_tests:
          - relationships:
              to: ref('stg_prompts')
              field: prompt_id
```

#### Location: `dbt/models/marts/schema.yml`

```yaml
# Tests on mart models (star schema)
models:
  - name: fct_sessions
    columns:
      - name: friction_score
        data_tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 100
```

### Custom Data Tests (in tests/ folder)

These are SQL queries that return rows when there's a problem.
**Test passes if query returns 0 rows.**

#### `tests/assert_no_orphan_generations.sql`

```sql
-- Test: No generations without valid user
-- Returns rows where generation has no matching user

select
    generation_id,
    user_id
from {{ ref('fct_generations') }} g
where not exists (
    select 1
    from {{ ref('dim_users') }} u
    where u.user_id = g.user_id
)
```

#### `tests/assert_friction_score_consistent.sql`

```sql
-- Test: High error rate sessions should have high friction
-- Sessions with 0% success should have friction >= 50

select
    session_id,
    success_rate_pct,
    friction_score
from {{ ref('fct_sessions') }}
where success_rate_pct = 0
  and friction_score < 50
```

#### `tests/assert_daily_metrics_complete.sql`

```sql
-- Test: No gaps in daily metrics time series

with date_range as (
    select
        min(session_date) as min_date,
        max(session_date) as max_date,
        count(distinct session_date) as actual_days,
        datediff('day', min(session_date), max(session_date)) + 1 as expected_days
    from {{ ref('daily_metrics') }}
)
select *
from date_range
where actual_days != expected_days  -- Fails if there are gaps
```

### Running Tests

```bash
# Run all tests
cd dbt && uv run dbt test

# Run only custom tests
uv run dbt test --select test_type:singular

# Run only schema tests
uv run dbt test --select test_type:generic

# Run tests for specific model
uv run dbt test --select fct_sessions

# Run tests and show failures
uv run dbt test --store-failures
```

---

## Daily Simulation Script

### Location: `scripts/run_daily_simulation.py`

### Purpose

Simulates 30 days of incremental data processing, like a production daily batch job.

### How It Works

```
Day 1                    Day 2                    Day 30
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│ Initial users  │      │ + New users    │      │ + New users    │
│ (40% of base)  │ ───> │ + New gens     │ ───> │ + New gens     │
│ + Generations  │      │ Rebuild dbt    │      │ Run tests      │
└────────────────┘      └────────────────┘      └────────────────┘
```

### Key Code Sections

#### 1. State Management

```python
def get_simulation_state(conn) -> dict:
    """Track simulation progress in database."""
    state = conn.execute("""
        SELECT state_key, state_value
        FROM simulation_state
    """).fetchall()
    return {k: v for k, v in state}
```

#### 2. Daily User Growth

```python
# Exponential growth pattern
DAILY_USER_GROWTH_RATE = 0.03  # 3% daily

if day_num == 1:
    new_users_today = int(BASE_USERS * 0.4)  # 40% on day 1
else:
    growth_multiplier = (1 + DAILY_USER_GROWTH_RATE) ** (day_num - 1)
    daily_base = BASE_USERS * 0.02
    new_users_today = int(daily_base * growth_multiplier)
```

#### 3. Generation Logic

```python
# Each user generates based on tier
tier_activity = {'free': 1, 'pro': 3, 'enterprise': 8}

for user in users:
    activity_level = tier_activity[user['user_tier']]
    num_gens = np.random.poisson(activity_level * 0.5)  # Poisson distribution
    num_gens = min(num_gens, 5)  # Cap at 5 per day
```

#### 4. Pipeline Integration

```python
def run_dbt():
    """Run dbt to rebuild models after each day."""
    subprocess.run(['uv', 'run', 'dbt', 'run'], cwd=DBT_PATH)

def run_dbt_tests():
    """Run dbt tests to validate data quality."""
    subprocess.run(['uv', 'run', 'dbt', 'test'], cwd=DBT_PATH)
```

### Usage

```bash
# Full 30-day simulation
uv run python scripts/run_daily_simulation.py --reset --days 30

# Quick 7-day test
uv run python scripts/run_daily_simulation.py --reset --days 7

# Continue from where you left off
uv run python scripts/run_daily_simulation.py --days 30

# Skip dbt for faster iteration
uv run python scripts/run_daily_simulation.py --reset --days 30 --skip-dbt
```

### Output Example

```
Day  1 (2025-12-01)... +200 users, + 847 gens, $ 42.31 revenue, 94.2% success
Day  2 (2025-12-02)... + 11 users, + 892 gens, $ 44.12 revenue, 93.8% success
...
Day 30 (2025-12-30)... + 18 users, +1247 gens, $ 62.35 revenue, 94.1% success

Weekly Growth:
  Week 1: + 234 users, + 6123 generations, $  305.42
  Week 2: +  89 users, + 7456 generations, $  372.80
  Week 3: + 102 users, + 8234 generations, $  411.70
  Week 4: + 115 users, + 9012 generations, $  450.60
```

---

## Backfill Framework

### Location: `scripts/backfill_metric.py`

### Purpose

Standardized way to add new metrics and backfill historical data.

### Architecture

```python
# Registry pattern for backfill jobs
BACKFILL_JOBS: Dict[str, Dict[str, Any]] = {}

@register_backfill(
    name='prompt_toxicity',
    description='Add toxicity score based on keyword detection'
)
def backfill_prompt_toxicity(conn, dry_run: bool = False) -> int:
    """Each backfill function follows the same pattern."""

    # 1. Check if column exists
    columns = conn.execute("DESCRIBE dim_prompts").fetchdf()
    has_column = 'toxicity_score' in columns['column_name'].values

    # 2. Dry run preview
    if dry_run:
        print(f"Would add column: toxicity_score")
        return count

    # 3. Add column if needed
    if not has_column:
        conn.execute("ALTER TABLE dim_prompts ADD COLUMN toxicity_score FLOAT")

    # 4. Compute and update values
    conn.execute("""
        UPDATE dim_prompts
        SET toxicity_score = <calculation>
    """)

    return rows_updated
```

### Available Backfills

| Name | Table | Column Added | Logic |
|------|-------|--------------|-------|
| `prompt_toxicity` | dim_prompts | toxicity_score (FLOAT) | Keyword-based score 0-1 |
| `user_lifetime_value` | dim_users | lifetime_value (FLOAT) | LTV with tier multipliers |
| `session_engagement_score` | fct_sessions | engagement_score (FLOAT) | Downloads + thumbs up + activity |
| `generation_quality_flag` | fct_generations | quality_tier (VARCHAR) | excellent/good/acceptable/poor/failed |

### Usage

```bash
# List available backfills
uv run python scripts/backfill_metric.py --list

# Preview what would happen
uv run python scripts/backfill_metric.py --metric prompt_toxicity --dry-run

# Execute backfill
uv run python scripts/backfill_metric.py --metric prompt_toxicity

# Run all backfills
uv run python scripts/backfill_metric.py --all
```

### Adding a New Backfill

```python
@register_backfill(
    name='my_new_metric',
    description='Description of what it does'
)
def backfill_my_new_metric(conn, dry_run: bool = False) -> int:
    # 1. Check/add column
    # 2. Compute values
    # 3. Return rows updated
    pass
```

---

## How to Use

### Quick Reference

```bash
# Navigate to project
cd /Users/jaychava/Documents/Luma/genai-session-analyzer

# Run dbt tests
cd dbt && uv run dbt test

# Run 30-day simulation
uv run python scripts/run_daily_simulation.py --reset --days 30

# Add a new metric via backfill
uv run python scripts/backfill_metric.py --metric user_lifetime_value

# Start dashboard to see results
uv run streamlit run dashboard/app.py
```

### Demo Flow

1. **Show dbt tests passing:**
   ```bash
   cd dbt && uv run dbt test
   # "104 tests passed - data quality is verified"
   ```

2. **Run a few days of simulation:**
   ```bash
   uv run python scripts/run_daily_simulation.py --reset --days 5 --skip-tests
   # "See how daily batches work - exponential user growth"
   ```

3. **Show backfill for new metric:**
   ```bash
   uv run python scripts/backfill_metric.py --metric user_lifetime_value --dry-run
   # "This is how we'd add LTV and backfill historical data"
   ```

---

## Architecture Decisions

### Why dbt Tests Over Python Tests?

| dbt Tests | Python Tests |
|-----------|--------------|
| SQL-native (where data lives) | Requires data extraction |
| Runs during pipeline | Runs separately |
| Documents expectations | Code-based |
| Industry standard | Custom |

### Why Custom Tests in SQL?

```sql
-- This is readable by anyone who knows SQL
select *
from fct_sessions
where success_rate_pct = 0
  and friction_score < 50
```

vs Python:

```python
# Requires understanding pandas/pytest
def test_friction_consistency():
    df = load_table('fct_sessions')
    violations = df[(df.success_rate_pct == 0) & (df.friction_score < 50)]
    assert len(violations) == 0
```

### Why Registry Pattern for Backfills?

1. **Discoverable** - `--list` shows all available backfills
2. **Consistent** - All backfills follow same interface
3. **Safe** - `--dry-run` previews changes
4. **Extensible** - Just add a decorated function

### Production Equivalents

| Local Demo | Production |
|------------|------------|
| `run_daily_simulation.py` | Airflow/Dagster scheduled DAG |
| `backfill_metric.py` | Parameterized backfill DAG |
| `dbt test` | dbt Cloud CI/CD |
| DuckDB | Snowflake/BigQuery/Redshift |

---

## Files Quick Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `dbt/packages.yml` | dbt dependencies | Adds dbt_utils |
| `dbt/models/staging/sources.yml` | Raw table definitions | Source tests |
| `dbt/models/staging/schema.yml` | Staging model tests | unique, not_null, relationships |
| `dbt/models/marts/schema.yml` | Mart model tests | accepted_range, accepted_values |
| `dbt/models/metrics/schema.yml` | Metrics model tests | Bounds validation |
| `dbt/tests/*.sql` | Custom data tests | Business logic validation |
| `scripts/run_daily_simulation.py` | Daily batch simulation | simulate_day(), run_dbt() |
| `scripts/backfill_metric.py` | Backfill framework | register_backfill(), run_backfill() |

---

## Interview Talking Points

1. **"Why dbt tests?"**
   > "I treat data quality like code quality. These 104 tests run every time the pipeline executes, catching issues before they reach the dashboard."

2. **"How would you add a new metric?"**
   > "I built a backfill framework. You define the calculation, run with `--dry-run` to preview, then execute. It handles schema changes and historical data."

3. **"How does this scale to production?"**
   > "The daily simulation shows incremental processing. In production, this would be an Airflow DAG with the same logic - just swap DuckDB for Snowflake."

4. **"What happens when a test fails?"**
   > "The pipeline stops and alerts. We have both schema tests (uniqueness, nulls) and business logic tests (friction score consistency, date completeness)."
