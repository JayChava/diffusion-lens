#!/usr/bin/env python
"""
Metric Backfill Framework - Add new metrics and backfill historical data.

This script demonstrates the production pattern for:
1. Adding a new column/metric to the data model
2. Backfilling historical data with the new metric
3. Validating the backfill
4. Updating dbt models

Usage:
    # Example: Add a new "prompt_complexity" metric
    uv run python scripts/backfill_metric.py --metric prompt_toxicity --dry-run
    uv run python scripts/backfill_metric.py --metric prompt_toxicity

    # List available backfill jobs
    uv run python scripts/backfill_metric.py --list

Production equivalent:
    - This would be a parameterized Airflow DAG or Dagster job
    - Run with backfill date range
    - Include data quality checks before/after
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any
import subprocess

import duckdb

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"
DBT_PATH = PROJECT_ROOT / "dbt"


# =============================================================================
# BACKFILL JOB REGISTRY
# =============================================================================
# Add new backfill jobs here. Each job is a function that:
# 1. Adds a column if needed
# 2. Computes the metric for all rows
# 3. Returns count of rows updated

BACKFILL_JOBS: Dict[str, Dict[str, Any]] = {}


def register_backfill(name: str, description: str):
    """Decorator to register a backfill job."""
    def decorator(func: Callable):
        BACKFILL_JOBS[name] = {
            'function': func,
            'description': description,
        }
        return func
    return decorator


# =============================================================================
# EXAMPLE BACKFILL JOBS
# =============================================================================

@register_backfill(
    name='prompt_toxicity',
    description='Add toxicity score (0-1) to prompts based on keyword detection'
)
def backfill_prompt_toxicity(conn, dry_run: bool = False) -> int:
    """
    Backfill prompt toxicity scores.

    This is a simple keyword-based approach for demo purposes.
    Production would use a proper toxicity model (Perspective API, etc.)
    """
    # Check if column exists
    columns = conn.execute("DESCRIBE dim_prompts").fetchdf()
    has_column = 'toxicity_score' in columns['column_name'].values

    if dry_run:
        print(f"  Would add column: toxicity_score FLOAT (exists: {has_column})")
        count = conn.execute("""
            SELECT COUNT(*) FROM dim_prompts WHERE prompt_text IS NOT NULL
        """).fetchone()[0]
        print(f"  Would update {count:,} rows")
        return count

    # Add column if needed
    if not has_column:
        print("  Adding toxicity_score column...")
        conn.execute("ALTER TABLE dim_prompts ADD COLUMN toxicity_score FLOAT")

    # Compute toxicity (simple keyword approach)
    print("  Computing toxicity scores...")

    # Toxicity keywords with weights
    toxicity_keywords = {
        'hate': 0.8, 'kill': 0.7, 'death': 0.5, 'blood': 0.4, 'gore': 0.6,
        'violence': 0.6, 'weapon': 0.4, 'drug': 0.4, 'nsfw': 0.3, 'nude': 0.3,
        'explicit': 0.4, 'racist': 0.9, 'terror': 0.7,
    }

    # Update in batches
    prompts = conn.execute("""
        SELECT prompt_id, prompt_text FROM dim_prompts
    """).fetchall()

    updates = []
    for prompt_id, prompt_text in prompts:
        if not prompt_text:
            score = 0.0
        else:
            prompt_lower = prompt_text.lower()
            matched_weights = [w for kw, w in toxicity_keywords.items() if kw in prompt_lower]
            score = min(1.0, sum(matched_weights)) if matched_weights else 0.0

        updates.append((score, prompt_id))

    # Batch update
    conn.executemany("""
        UPDATE dim_prompts SET toxicity_score = ? WHERE prompt_id = ?
    """, updates)

    return len(updates)


@register_backfill(
    name='user_lifetime_value',
    description='Calculate and store user LTV based on historical spend'
)
def backfill_user_ltv(conn, dry_run: bool = False) -> int:
    """
    Backfill user lifetime value (LTV) metric.

    LTV = total_spend * (1 + tier_multiplier)
    where tier_multiplier accounts for expected retention
    """
    columns = conn.execute("DESCRIBE dim_users").fetchdf()
    has_column = 'lifetime_value' in columns['column_name'].values

    if dry_run:
        print(f"  Would add column: lifetime_value FLOAT (exists: {has_column})")
        count = conn.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]
        print(f"  Would update {count:,} rows")
        return count

    if not has_column:
        print("  Adding lifetime_value column...")
        conn.execute("ALTER TABLE dim_users ADD COLUMN lifetime_value FLOAT")

    # Calculate LTV with tier multipliers
    print("  Computing lifetime values...")

    tier_multipliers = {'free': 0.1, 'pro': 0.5, 'enterprise': 1.0}

    # Update using SQL for efficiency
    conn.execute("""
        UPDATE dim_users
        SET lifetime_value = lifetime_cost * (1.0 + CASE
            WHEN user_tier = 'free' THEN 0.1
            WHEN user_tier = 'pro' THEN 0.5
            WHEN user_tier = 'enterprise' THEN 1.0
            ELSE 0.0
        END)
    """)

    count = conn.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]
    return count


@register_backfill(
    name='session_engagement_score',
    description='Add engagement score to sessions based on actions'
)
def backfill_session_engagement(conn, dry_run: bool = False) -> int:
    """
    Backfill session engagement scores.

    Engagement = weighted sum of:
    - Downloads (high signal)
    - Thumbs up (positive signal)
    - Multiple generations (engagement signal)
    - Low retry rate (satisfaction signal)
    """
    columns = conn.execute("DESCRIBE fct_sessions").fetchdf()
    has_column = 'engagement_score' in columns['column_name'].values

    if dry_run:
        print(f"  Would add column: engagement_score FLOAT (exists: {has_column})")
        count = conn.execute("SELECT COUNT(*) FROM fct_sessions").fetchone()[0]
        print(f"  Would update {count:,} rows")
        return count

    if not has_column:
        print("  Adding engagement_score column...")
        conn.execute("ALTER TABLE fct_sessions ADD COLUMN engagement_score FLOAT")

    print("  Computing engagement scores...")

    # Engagement formula (0-100 scale)
    conn.execute("""
        UPDATE fct_sessions
        SET engagement_score = LEAST(100, (
            -- Downloads are high-intent (40 points max)
            (download_count * 10.0) +
            -- Thumbs up is positive signal (20 points max)
            (thumbs_up_count * 5.0) +
            -- Multiple generations show engagement (20 points max)
            LEAST(20, total_generations * 2.0) +
            -- Low retry rate means satisfaction (20 points max)
            (1.0 - LEAST(1.0, avg_retry_count)) * 20.0
        ))
    """)

    count = conn.execute("SELECT COUNT(*) FROM fct_sessions").fetchone()[0]
    return count


@register_backfill(
    name='generation_quality_flag',
    description='Add quality tier flag to generations based on latency and success'
)
def backfill_generation_quality(conn, dry_run: bool = False) -> int:
    """
    Backfill generation quality tier.

    Quality tiers:
    - 'excellent': success + latency < 2s
    - 'good': success + latency < 5s
    - 'acceptable': success + latency < 10s
    - 'poor': success + latency >= 10s
    - 'failed': not success
    """
    columns = conn.execute("DESCRIBE fct_generations").fetchdf()
    has_column = 'quality_tier' in columns['column_name'].values

    if dry_run:
        print(f"  Would add column: quality_tier VARCHAR (exists: {has_column})")
        count = conn.execute("SELECT COUNT(*) FROM fct_generations").fetchone()[0]
        print(f"  Would update {count:,} rows")
        return count

    if not has_column:
        print("  Adding quality_tier column...")
        conn.execute("ALTER TABLE fct_generations ADD COLUMN quality_tier VARCHAR")

    print("  Computing quality tiers...")

    conn.execute("""
        UPDATE fct_generations
        SET quality_tier = CASE
            WHEN status != 'success' THEN 'failed'
            WHEN latency_ms < 2000 THEN 'excellent'
            WHEN latency_ms < 5000 THEN 'good'
            WHEN latency_ms < 10000 THEN 'acceptable'
            ELSE 'poor'
        END
    """)

    count = conn.execute("SELECT COUNT(*) FROM fct_generations").fetchone()[0]
    return count


# =============================================================================
# MAIN BACKFILL RUNNER
# =============================================================================

def run_backfill(metric_name: str, dry_run: bool = False):
    """Run a backfill job."""
    if metric_name not in BACKFILL_JOBS:
        print(f"Error: Unknown metric '{metric_name}'")
        print(f"Available metrics: {', '.join(BACKFILL_JOBS.keys())}")
        return

    job = BACKFILL_JOBS[metric_name]
    print(f"\n{'DRY RUN: ' if dry_run else ''}Backfill: {metric_name}")
    print(f"Description: {job['description']}")
    print("-" * 50)

    conn = duckdb.connect(str(DB_PATH))

    start_time = datetime.now()
    rows_updated = job['function'](conn, dry_run=dry_run)
    elapsed = (datetime.now() - start_time).total_seconds()

    if not dry_run:
        conn.commit()

        # Validate
        print(f"\nValidation:")
        print(f"  Rows updated: {rows_updated:,}")
        print(f"  Time elapsed: {elapsed:.2f}s")

    conn.close()

    # Remind to run dbt
    if not dry_run:
        print(f"\n{'='*50}")
        print("NEXT STEPS:")
        print("1. Review the changes in the database")
        print("2. Update dbt models if needed (add column to schema.yml)")
        print("3. Run dbt to rebuild downstream models:")
        print("   cd dbt && uv run dbt run")
        print("4. Run dbt tests to validate:")
        print("   cd dbt && uv run dbt test")


def list_backfills():
    """List available backfill jobs."""
    print("\nAvailable Backfill Jobs:")
    print("-" * 50)
    for name, job in BACKFILL_JOBS.items():
        print(f"  {name}")
        print(f"    {job['description']}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Backfill metrics to historical data')
    parser.add_argument('--metric', type=str, help='Metric to backfill')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--list', action='store_true', help='List available backfills')
    parser.add_argument('--all', action='store_true', help='Run all backfills')
    args = parser.parse_args()

    if args.list:
        list_backfills()
        return

    if args.all:
        for metric_name in BACKFILL_JOBS.keys():
            run_backfill(metric_name, dry_run=args.dry_run)
        return

    if not args.metric:
        parser.print_help()
        print("\n\nUse --list to see available backfill jobs")
        return

    run_backfill(args.metric, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
