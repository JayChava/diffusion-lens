"""
User Generator - Creates synthetic users with realistic tier distribution.

Distribution (power law, typical freemium):
- 70% free tier (casual users)
- 25% pro tier (regular users)
- 5% enterprise tier (power users)

Signups are spread uniformly across the month.
"""

import duckdb
from faker import Faker
import random
import numpy as np
from datetime import datetime, timedelta


fake = Faker()
# No seeds - each run produces different users

# User tier distribution
TIER_DISTRIBUTION = {
    "free": 0.70,
    "pro": 0.25,
    "enterprise": 0.05,
}

# Region distribution (US-heavy, realistic for GenAI tools)
REGION_DISTRIBUTION = {
    "us-west": 0.30,
    "us-east": 0.25,
    "europe": 0.20,
    "asia": 0.15,
    "other": 0.10,
}

DEVICE_TYPES = ["desktop", "mobile", "tablet"]
DEVICE_WEIGHTS = [0.70, 0.25, 0.05]  # Desktop-heavy for creative tools


def _weighted_choice(distribution: dict) -> str:
    """Pick from a weighted distribution."""
    return random.choices(
        list(distribution.keys()),
        weights=list(distribution.values()),
        k=1
    )[0]


def _generate_growth_timestamps(num_samples: int, start_date: datetime, end_date: datetime, growth_rate: float = 2.0):
    """
    Generate timestamps following exponential growth pattern.

    Args:
        num_samples: Number of timestamps to generate
        start_date: Start of period
        end_date: End of period
        growth_rate: Higher = more skewed toward end (2.0 = doubles over period)

    Returns:
        List of datetime objects weighted toward end of period
    """
    total_seconds = (end_date - start_date).total_seconds()

    # Generate timestamps uniformly spread across month
    # Use uniform distribution for even coverage of all weeks
    random_values = np.random.uniform(0, 1, num_samples)

    timestamps = []
    for val in random_values:
        seconds_offset = int(val * total_seconds)
        timestamps.append(start_date + timedelta(seconds=seconds_offset))

    return sorted(timestamps)


def generate_users(
    num_users: int,
    db_path: str = "data/warehouse.duckdb",
    year: int = 2025,
    month: int = 12,
) -> int:
    """
    Generate synthetic users with exponential growth pattern.

    Users sign up throughout the month with growth curve:
    - Week 1: ~15% of users
    - Week 2: ~20% of users
    - Week 3: ~28% of users
    - Week 4: ~37% of users

    Args:
        num_users: Number of users to generate
        db_path: Path to DuckDB file
        year: Year for signups
        month: Month for signups

    Returns:
        Number of users created
    """
    # Set date range
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year, month, 31, 23, 59, 59)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)

    # Generate signup dates spread across month (mild growth pattern)
    signup_dates = _generate_growth_timestamps(num_users, start_date, end_date, growth_rate=1.2)

    users = []
    for i, signup_date in enumerate(signup_dates):
        tier = _weighted_choice(TIER_DISTRIBUTION)

        users.append({
            "user_id": i,
            "user_tier": tier,
            "signup_date": signup_date,
            "cohort_week": signup_date.strftime("%Y-W%W"),
            "region": _weighted_choice(REGION_DISTRIBUTION),
            "device_type": random.choices(DEVICE_TYPES, weights=DEVICE_WEIGHTS, k=1)[0],
        })

    # Write to DuckDB
    conn = duckdb.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS raw_users")
    conn.execute("""
        CREATE TABLE raw_users (
            user_id INTEGER PRIMARY KEY,
            user_tier VARCHAR,
            signup_date TIMESTAMP,
            cohort_week VARCHAR,
            region VARCHAR,
            device_type VARCHAR
        )
    """)

    conn.executemany(
        "INSERT INTO raw_users VALUES (?, ?, ?, ?, ?, ?)",
        [(u["user_id"], u["user_tier"], u["signup_date"],
          u["cohort_week"], u["region"], u["device_type"]) for u in users]
    )

    row_count = conn.execute("SELECT COUNT(*) FROM raw_users").fetchone()[0]

    # Print distribution summary
    print(f"Generated {row_count} users with growth pattern:")

    # Tier distribution
    tier_counts = conn.execute("""
        SELECT user_tier, COUNT(*) as cnt,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
        FROM raw_users GROUP BY user_tier ORDER BY cnt DESC
    """).fetchall()
    print("\nTier distribution:")
    for tier, cnt, pct in tier_counts:
        print(f"  {tier}: {cnt} ({pct}%)")

    # Weekly signup distribution (shows growth)
    weekly_counts = conn.execute("""
        SELECT
            CASE
                WHEN DAY(signup_date) <= 7 THEN 'Week 1 (Dec 1-7)'
                WHEN DAY(signup_date) <= 14 THEN 'Week 2 (Dec 8-14)'
                WHEN DAY(signup_date) <= 21 THEN 'Week 3 (Dec 15-21)'
                ELSE 'Week 4 (Dec 22-31)'
            END as week,
            COUNT(*) as cnt,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
        FROM raw_users
        GROUP BY week
        ORDER BY week
    """).fetchall()
    print("\nSignup growth (weekly):")
    for week, cnt, pct in weekly_counts:
        bar = "█" * int(pct / 3)
        print(f"  {week}: {cnt:4d} ({pct:4.1f}%) {bar}")

    conn.close()
    return row_count


if __name__ == "__main__":
    import sys
    num_users = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    generate_users(num_users)
