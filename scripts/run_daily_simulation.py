#!/usr/bin/env python
"""
Daily Simulation Runner - Simulates 30 days of pipeline execution.

This script demonstrates production-style incremental data processing:
1. Day 1: Initial load with base users and prompts
2. Days 2-30: Incremental new users, new generations, new sessions

Usage:
    uv run python scripts/run_daily_simulation.py
    uv run python scripts/run_daily_simulation.py --days 7  # Just one week
    uv run python scripts/run_daily_simulation.py --reset   # Clear and restart

What this simulates:
- Daily user signups (exponential growth pattern)
- Daily generations from all active users
- Session building and aggregation
- dbt model refresh

Production equivalent:
- This would be an Airflow/Dagster daily scheduled job
- Each "day" = one pipeline run with date parameter
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import random

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"
DBT_PATH = PROJECT_ROOT / "dbt"

# Simulation parameters
BASE_YEAR = 2025
BASE_MONTH = 12
TOTAL_PROMPTS = 10_000
BASE_USERS = 500
DAILY_USER_GROWTH_RATE = 0.03  # 3% daily growth


def reset_database():
    """Clear incremental tables for fresh simulation."""
    print("Resetting database for fresh simulation...")
    conn = duckdb.connect(str(DB_PATH))

    # Keep raw_prompts (source data), reset everything else
    conn.execute("DROP TABLE IF EXISTS raw_generations_incremental")
    conn.execute("DROP TABLE IF EXISTS raw_users_incremental")
    conn.execute("DROP TABLE IF EXISTS simulation_state")

    # Create state tracking table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS simulation_state (
            state_key VARCHAR PRIMARY KEY,
            state_value VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.close()
    print("Database reset complete.")


def get_simulation_state(conn) -> dict:
    """Get current simulation state."""
    try:
        state = conn.execute("""
            SELECT state_key, state_value
            FROM simulation_state
        """).fetchall()
        return {k: v for k, v in state}
    except:
        return {}


def set_simulation_state(conn, key: str, value: str):
    """Set simulation state."""
    conn.execute("""
        INSERT INTO simulation_state (state_key, state_value, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT (state_key) DO UPDATE SET
            state_value = EXCLUDED.state_value,
            updated_at = CURRENT_TIMESTAMP
    """, [key, value])


def simulate_day(day_num: int, simulation_date: datetime, conn) -> dict:
    """
    Simulate one day of platform activity.

    Returns metrics for the day.
    """
    random.seed(42 + day_num)
    np.random.seed(42 + day_num)

    state = get_simulation_state(conn)
    current_user_count = int(state.get('total_users', '0'))
    current_gen_count = int(state.get('total_generations', '0'))

    # Calculate new users for this day (exponential growth)
    if day_num == 1:
        # Day 1: Initial user base
        new_users_today = int(BASE_USERS * 0.4)  # 40% of users on day 1
    else:
        # Subsequent days: Exponential growth
        growth_multiplier = (1 + DAILY_USER_GROWTH_RATE) ** (day_num - 1)
        daily_base = BASE_USERS * 0.02  # 2% of total per day
        new_users_today = int(daily_base * growth_multiplier)

    new_users_today = max(1, new_users_today)  # At least 1 user

    # User tier distribution
    tier_weights = {'free': 0.70, 'pro': 0.25, 'enterprise': 0.05}
    regions = ['NA', 'EU', 'APAC', 'LATAM']
    region_weights = [0.40, 0.30, 0.20, 0.10]
    devices = ['desktop', 'mobile', 'tablet']
    device_weights = [0.50, 0.40, 0.10]

    # Generate new users
    new_users = []
    for i in range(new_users_today):
        user_id = current_user_count + i
        tier = random.choices(list(tier_weights.keys()),
                              weights=list(tier_weights.values()))[0]
        region = random.choices(regions, weights=region_weights)[0]
        device = random.choices(devices, weights=device_weights)[0]

        # Signup time is during this day
        signup_hour = random.randint(0, 23)
        signup_minute = random.randint(0, 59)
        signup_time = simulation_date.replace(hour=signup_hour, minute=signup_minute)

        new_users.append({
            'user_id': user_id,
            'user_tier': tier,
            'signup_date': signup_time,
            'cohort_week': f"W{(day_num - 1) // 7 + 1}",
            'region': region,
            'device_type': device,
        })

    # Insert new users
    if new_users:
        conn.executemany("""
            INSERT INTO raw_users (user_id, user_tier, signup_date, cohort_week, region, device_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [(u['user_id'], u['user_tier'], u['signup_date'],
               u['cohort_week'], u['region'], u['device_type']) for u in new_users])

    # Calculate generations for this day
    # More users = more generations, with tier-based activity multipliers
    users_df = conn.execute("""
        SELECT user_id, user_tier, signup_date
        FROM raw_users
        WHERE signup_date <= ?
    """, [simulation_date.replace(hour=23, minute=59)]).fetchdf()

    tier_activity = {'free': 1, 'pro': 3, 'enterprise': 8}

    # Each user generates 0-3 prompts per day based on tier
    new_generations = []
    prompt_pool = conn.execute("""
        SELECT prompt_id, prompt, token_count
        FROM raw_prompts
    """).fetchall()

    model_versions = ["v1.4", "v1.5", "v2.0", "v2.1"]
    model_weights = [0.05, 0.15, 0.30, 0.50]

    for _, user in users_df.iterrows():
        # Skip if user signed up after today
        if user['signup_date'].date() > simulation_date.date():
            continue

        # Number of generations based on tier
        activity_level = tier_activity[user['user_tier']]
        num_gens = np.random.poisson(activity_level * 0.5)  # Poisson distribution
        num_gens = min(num_gens, 5)  # Cap at 5 per day

        for _ in range(num_gens):
            # Pick a random prompt
            prompt_row = random.choice(prompt_pool)
            prompt_id, prompt_text, token_count = prompt_row

            # Determine status
            status = _determine_status(prompt_text, user['user_tier'], token_count)
            latency = _generate_latency(token_count, status)
            cost = _calculate_cost(token_count, latency, status, user['user_tier'])
            retry_count = random.choices([0, 1, 2], weights=[0.85, 0.12, 0.03])[0]

            # Feedback
            feedback_prob = {'free': 0.10, 'pro': 0.20, 'enterprise': 0.25}
            feedback = None
            if random.random() < feedback_prob[user['user_tier']]:
                feedback = 'thumbs_up' if (status == 'success' and random.random() < 0.8) else 'thumbs_down'

            # Downloaded
            downloaded = (status == 'success' and random.random() < 0.5)

            # Timestamp within the day
            gen_hour = random.randint(8, 23)  # Activity between 8am-11pm
            gen_minute = random.randint(0, 59)
            gen_time = simulation_date.replace(hour=gen_hour, minute=gen_minute)

            model_version = random.choices(model_versions, weights=model_weights)[0]

            new_generations.append({
                'generation_id': current_gen_count + len(new_generations),
                'prompt_id': prompt_id,
                'user_id': user['user_id'],
                'timestamp': gen_time,
                'session_date': simulation_date.date(),
                'latency_ms': latency,
                'status': status,
                'cost_credits': cost,
                'retry_count': retry_count,
                'feedback': feedback,
                'downloaded': downloaded,
                'model_version': model_version,
                'token_count': token_count,
            })

    # Insert new generations
    if new_generations:
        conn.executemany("""
            INSERT INTO raw_generations
            (generation_id, prompt_id, user_id, timestamp, session_date,
             latency_ms, status, cost_credits, retry_count, feedback,
             downloaded, model_version, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(g['generation_id'], g['prompt_id'], g['user_id'], g['timestamp'],
               g['session_date'], g['latency_ms'], g['status'], g['cost_credits'],
               g['retry_count'], g['feedback'], g['downloaded'], g['model_version'],
               g['token_count']) for g in new_generations])

    # Update state
    new_total_users = current_user_count + len(new_users)
    new_total_gens = current_gen_count + len(new_generations)
    set_simulation_state(conn, 'total_users', str(new_total_users))
    set_simulation_state(conn, 'total_generations', str(new_total_gens))
    set_simulation_state(conn, 'last_simulation_date', simulation_date.isoformat())
    set_simulation_state(conn, 'last_simulation_day', str(day_num))

    # Calculate day's metrics
    success_count = sum(1 for g in new_generations if g['status'] == 'success')
    total_cost = sum(g['cost_credits'] for g in new_generations)

    return {
        'day': day_num,
        'date': simulation_date.strftime('%Y-%m-%d'),
        'new_users': len(new_users),
        'total_users': new_total_users,
        'new_generations': len(new_generations),
        'total_generations': new_total_gens,
        'success_rate': success_count / len(new_generations) if new_generations else 0,
        'daily_revenue': total_cost,
    }


def _determine_status(prompt: str, user_tier: str, token_count: int) -> str:
    """Determine generation status."""
    safety_keywords = ['nude', 'naked', 'nsfw', 'explicit', 'gore', 'violence']

    if any(kw in prompt.lower() for kw in safety_keywords):
        return 'safety_violation' if random.random() < 0.85 else 'success'
    if token_count > 75:
        return 'timeout' if random.random() < 0.15 else 'success'
    if user_tier == 'free' and random.random() < 0.08:
        return 'rate_limited'
    if random.random() < 0.02:
        return 'model_error'
    return 'success'


def _generate_latency(token_count: int, status: str) -> int:
    """Generate realistic latency."""
    if status == 'timeout':
        return 30000
    if status == 'safety_violation':
        return random.randint(100, 500)
    if status in ('rate_limited', 'model_error'):
        return random.randint(50, 200)
    base = token_count * 50
    noise = np.random.lognormal(0, 0.4)
    return int(max(200, min(25000, base * noise)))


def _calculate_cost(token_count: int, latency_ms: int, status: str, user_tier: str) -> float:
    """Calculate cost in credits."""
    if status != 'success':
        return round(token_count * 0.001, 4)
    token_cost = token_count * 0.01
    compute_cost = (latency_ms / 1000) * 0.05
    total = token_cost + compute_cost
    if user_tier == 'enterprise':
        total *= 0.80
    elif user_tier == 'pro':
        total *= 0.90
    return round(total, 4)


def run_dbt():
    """Run dbt to rebuild models."""
    print("\n  Running dbt models...")
    result = subprocess.run(
        ['uv', 'run', 'dbt', 'run'],
        cwd=str(DBT_PATH),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  dbt error: {result.stderr}")
        return False
    return True


def run_dbt_tests():
    """Run dbt tests."""
    print("  Running dbt tests...")
    result = subprocess.run(
        ['uv', 'run', 'dbt', 'test'],
        cwd=str(DBT_PATH),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  Test failures:\n{result.stdout}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Run daily simulation for N days')
    parser.add_argument('--days', type=int, default=30, help='Number of days to simulate')
    parser.add_argument('--reset', action='store_true', help='Reset and start fresh')
    parser.add_argument('--skip-dbt', action='store_true', help='Skip dbt runs (faster)')
    parser.add_argument('--skip-tests', action='store_true', help='Skip dbt tests')
    args = parser.parse_args()

    print("=" * 60)
    print("GenAI Session Analyzer - Daily Simulation")
    print("=" * 60)

    conn = duckdb.connect(str(DB_PATH))

    # Check if we need to reset
    if args.reset:
        reset_database()

        # Ensure raw tables exist with correct schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_users (
                user_id INTEGER PRIMARY KEY,
                user_tier VARCHAR,
                signup_date TIMESTAMP,
                cohort_week VARCHAR,
                region VARCHAR,
                device_type VARCHAR
            )
        """)
        conn.execute("DELETE FROM raw_users")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_generations (
                generation_id INTEGER PRIMARY KEY,
                prompt_id INTEGER,
                user_id INTEGER,
                timestamp TIMESTAMP,
                session_date DATE,
                latency_ms INTEGER,
                status VARCHAR,
                cost_credits DOUBLE,
                retry_count INTEGER,
                feedback VARCHAR,
                downloaded BOOLEAN,
                model_version VARCHAR,
                token_count INTEGER
            )
        """)
        conn.execute("DELETE FROM raw_generations")

        # Create state table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS simulation_state (
                state_key VARCHAR PRIMARY KEY,
                state_value VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    # Get current state
    state = get_simulation_state(conn)
    start_day = int(state.get('last_simulation_day', '0')) + 1

    if start_day > args.days:
        print(f"\nSimulation already complete ({start_day-1} days).")
        print("Use --reset to start fresh.")
        return

    print(f"\nSimulating days {start_day} to {args.days}...")
    print(f"Base date: {BASE_YEAR}-{BASE_MONTH:02d}-01")
    print()

    # Run simulation for each day
    daily_results = []
    for day_num in range(start_day, args.days + 1):
        simulation_date = datetime(BASE_YEAR, BASE_MONTH, 1) + timedelta(days=day_num - 1)

        print(f"Day {day_num:2d} ({simulation_date.strftime('%Y-%m-%d')})...", end=' ')

        metrics = simulate_day(day_num, simulation_date, conn)
        daily_results.append(metrics)

        print(f"+{metrics['new_users']:3d} users, +{metrics['new_generations']:4d} gens, "
              f"${metrics['daily_revenue']:6.2f} revenue, "
              f"{metrics['success_rate']*100:4.1f}% success")

        conn.commit()

    # Rebuild sessions
    print("\nRebuilding sessions...")
    from src.simulation.session_builder import build_sessions
    session_count = build_sessions(db_path=str(DB_PATH))
    print(f"  {session_count} sessions built")

    # Run dbt
    if not args.skip_dbt:
        print("\nRunning dbt pipeline...")

        # First install packages
        print("  Installing dbt packages...")
        subprocess.run(['uv', 'run', 'dbt', 'deps'], cwd=str(DBT_PATH),
                       capture_output=True)

        if run_dbt():
            print("  dbt models rebuilt successfully")

        if not args.skip_tests:
            if run_dbt_tests():
                print("  All dbt tests passed!")

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)

    final_state = get_simulation_state(conn)
    print(f"Total days simulated: {args.days}")
    print(f"Total users: {final_state.get('total_users', 'N/A')}")
    print(f"Total generations: {final_state.get('total_generations', 'N/A')}")

    # Weekly breakdown
    print("\nWeekly Growth:")
    for week in range(1, 5):
        week_days = [d for d in daily_results if (d['day'] - 1) // 7 + 1 == week]
        if week_days:
            week_users = sum(d['new_users'] for d in week_days)
            week_gens = sum(d['new_generations'] for d in week_days)
            week_revenue = sum(d['daily_revenue'] for d in week_days)
            print(f"  Week {week}: +{week_users:4d} users, +{week_gens:5d} generations, ${week_revenue:8.2f}")

    conn.close()
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
