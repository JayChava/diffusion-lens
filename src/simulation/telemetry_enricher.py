"""
Telemetry Enricher - Adds realistic synthetic telemetry to prompts.

Generates:
- status: success/timeout/safety_violation/rate_limited/model_error
- latency_ms: based on prompt complexity and outcome
- cost_credits: based on tokens + compute time
- retry_count: how many retries before final status
- feedback: thumbs_up/thumbs_down/null (explicit user feedback)
- downloaded: true/false (did user download the result)
- timestamp: when generation occurred (follows growth pattern)
- model_version: which model was used

Growth pattern:
- Timestamps weighted toward end of month (more activity as platform grows)
- Users can only generate after their signup date
- Compound growth: more users + each user more engaged over time
"""

import duckdb
import random
import numpy as np
from datetime import datetime, timedelta
import re

# Seeds removed - each run produces different telemetry data
# random.seed(42)
# np.random.seed(42)

# NSFW/safety keywords (subset for demo)
SAFETY_KEYWORDS = [
    "nude", "naked", "nsfw", "explicit", "gore", "violence", "blood",
    "weapon", "drug", "kill", "death", "hate", "racist"
]

# Model versions (weighted toward newer)
MODEL_VERSIONS = ["v1.4", "v1.5", "v2.0", "v2.1"]
MODEL_WEIGHTS = [0.05, 0.15, 0.30, 0.50]


def _count_tokens(text: str) -> int:
    """Simple token count approximation (words + punctuation clusters)."""
    return len(text.split())


def _has_safety_concern(prompt: str) -> bool:
    """Check if prompt contains safety-flagged content."""
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in SAFETY_KEYWORDS)


# Per-run randomized rates (set once at module load for consistency within a run)
_RUN_RATES = {
    "safety_violation": random.uniform(0.70, 0.95),  # 70-95% for NSFW
    "timeout": random.uniform(0.05, 0.25),           # 5-25% for long prompts
    "rate_limited": random.uniform(0.03, 0.15),      # 3-15% for free tier
    "model_error": random.uniform(0.01, 0.08),       # 1-8% baseline errors
}

def _determine_status(prompt: str, user_tier: str, token_count: int) -> str:
    """
    Determine generation status based on prompt characteristics.
    Rates vary between pipeline runs for visible changes in metrics.
    """
    # Safety check first
    if _has_safety_concern(prompt):
        return "safety_violation" if random.random() < _RUN_RATES["safety_violation"] else "success"

    # Timeout for complex prompts
    if token_count > 75:
        return "timeout" if random.random() < _RUN_RATES["timeout"] else "success"

    # Rate limiting for free tier
    if user_tier == "free" and random.random() < _RUN_RATES["rate_limited"]:
        return "rate_limited"

    # Random model errors
    if random.random() < _RUN_RATES["model_error"]:
        return "model_error"

    return "success"


# Per-run latency multiplier (varies average latency between runs)
_LATENCY_MULTIPLIER = random.uniform(0.6, 1.5)

def _generate_latency(token_count: int, status: str) -> int:
    """
    Generate realistic latency based on prompt and outcome.
    Latency scale varies between runs for visible changes.
    """
    if status == "timeout":
        return random.randint(25000, 35000)  # 25-35s timeout

    if status == "safety_violation":
        return random.randint(100, 500)

    if status in ("rate_limited", "model_error"):
        return random.randint(50, 200)

    # Success: base latency with realistic variance + per-run multiplier
    base = token_count * 50 * _LATENCY_MULTIPLIER
    noise = np.random.lognormal(0, 0.4)
    return int(max(200, min(25000, base * noise)))


def _calculate_cost(token_count: int, latency_ms: int, status: str, user_tier: str) -> float:
    """
    Calculate cost in credits.

    - Failed generations cost less (no compute used)
    - Enterprise gets 20% discount
    - Pro gets 10% discount
    """
    if status != "success":
        return round(token_count * 0.001, 4)  # Minimal token processing cost

    # Base cost: tokens + compute time
    token_cost = token_count * 0.01
    compute_cost = (latency_ms / 1000) * 0.05
    total = token_cost + compute_cost

    # Tier discounts
    if user_tier == "enterprise":
        total *= 0.80
    elif user_tier == "pro":
        total *= 0.90

    return round(total, 4)


def _determine_retry_count(status: str) -> int:
    """How many retries before this final status."""
    if status == "success":
        # Most succeed first try, some retry once
        return random.choices([0, 1, 2], weights=[0.85, 0.12, 0.03], k=1)[0]
    elif status == "rate_limited":
        # Rate limited users often retry
        return random.choices([0, 1, 2, 3], weights=[0.30, 0.35, 0.25, 0.10], k=1)[0]
    else:
        # Errors/violations usually don't retry
        return random.choices([0, 1], weights=[0.90, 0.10], k=1)[0]


def _determine_feedback(status: str, user_tier: str):
    """
    Determine explicit feedback (thumbs up/down).

    Logic:
    - Only ~15% of users leave feedback
    - Pro/enterprise users more likely to give feedback
    - Success → mostly positive
    - Failures → mostly negative
    """
    # Feedback probability by tier
    feedback_prob = {"free": 0.10, "pro": 0.20, "enterprise": 0.25}

    if random.random() > feedback_prob.get(user_tier, 0.10):
        return None  # No feedback given

    # Feedback sentiment based on outcome
    if status == "success":
        return "thumbs_up" if random.random() < 0.80 else "thumbs_down"
    else:
        return "thumbs_down" if random.random() < 0.70 else "thumbs_up"


def _determine_download(status: str, user_tier: str, token_count: int) -> bool:
    """
    Did the user download the generated image?

    Logic:
    - Can only download successful generations
    - Pro/enterprise download more (they're paying)
    - Longer/more specific prompts → higher download rate
    """
    if status != "success":
        return False

    # Base download rate by tier
    base_rate = {"free": 0.40, "pro": 0.65, "enterprise": 0.80}
    rate = base_rate.get(user_tier, 0.40)

    # Longer prompts = more intentional = more downloads
    if token_count > 30:
        rate += 0.15
    elif token_count < 10:
        rate -= 0.10

    return random.random() < min(0.95, rate)


def _generate_growth_timestamp(user_signup: datetime, end_date: datetime, growth_bias: float = 0.7):
    """
    Generate a timestamp for a generation with growth bias toward end of month.

    Args:
        user_signup: User's signup date (generation must be after this)
        end_date: End of activity period
        growth_bias: 0-1, higher = more skewed toward end (0.7 = strong growth)

    Returns:
        datetime: Timestamp between signup and end_date, biased toward end
    """
    if user_signup >= end_date:
        return end_date - timedelta(hours=random.randint(1, 24))

    total_seconds = (end_date - user_signup).total_seconds()

    # Uniform distribution for even spread across available time window
    random_val = np.random.uniform(0, 1)

    seconds_offset = int(random_val * total_seconds)
    return user_signup + timedelta(seconds=seconds_offset)


def enrich_generations(
    db_path: str = "data/warehouse.duckdb",
    year: int = 2025,
    month: int = 12,
) -> int:
    """
    Create enriched generations by joining prompts with users and adding telemetry.

    Growth pattern:
    - Users can only generate after their signup date
    - Timestamps biased toward end of month (platform growth)
    - Pro/enterprise users more active (weighted selection)

    Each prompt is assigned to a random user and enriched with:
    - timestamp, session_date, latency, status, cost, retry_count
    - feedback (thumbs_up/thumbs_down/null)
    - downloaded (true/false)
    - model_version

    Args:
        db_path: Path to DuckDB file
        year: Year for activity (default 2025)
        month: Month for activity (default 12 = December)

    Returns:
        Number of generations created
    """
    conn = duckdb.connect(db_path)

    # Get prompts and users (including signup date for realistic timestamps)
    prompts = conn.execute("SELECT * FROM raw_prompts").fetchdf()
    users = conn.execute("SELECT user_id, user_tier, signup_date FROM raw_users").fetchdf()

    if len(prompts) == 0 or len(users) == 0:
        raise ValueError("Must have raw_prompts and raw_users tables populated first")

    print(f"Enriching {len(prompts)} prompts with telemetry (growth pattern)...")
    print(f"Activity period: {year}-{month:02d}")

    # Assign users to prompts (power users generate more)
    user_weights = users["user_tier"].map({
        "free": 1,
        "pro": 3,      # Pro users 3x more active
        "enterprise": 8  # Enterprise 8x more active
    }).values

    # Set date range to specified month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year, month, 31, 23, 59, 59)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)

    generations = []
    for _, prompt_row in prompts.iterrows():
        # Pick a user (weighted by tier activity)
        user_idx = random.choices(range(len(users)), weights=user_weights, k=1)[0]
        user = users.iloc[user_idx]

        prompt_text = prompt_row["prompt"]
        token_count = _count_tokens(prompt_text)

        # Generate telemetry
        status = _determine_status(prompt_text, user["user_tier"], token_count)
        latency = _generate_latency(token_count, status)
        cost = _calculate_cost(token_count, latency, status, user["user_tier"])
        retry_count = _determine_retry_count(status)
        feedback = _determine_feedback(status, user["user_tier"])
        downloaded = _determine_download(status, user["user_tier"], token_count)
        model_version = random.choices(MODEL_VERSIONS, weights=MODEL_WEIGHTS, k=1)[0]

        # Generate timestamp with growth pattern (after user signup)
        user_signup = user["signup_date"]
        if isinstance(user_signup, str):
            user_signup = datetime.fromisoformat(user_signup)
        elif hasattr(user_signup, 'to_pydatetime'):
            user_signup = user_signup.to_pydatetime()

        timestamp = _generate_growth_timestamp(user_signup, end_date, growth_bias=0.65)

        generations.append({
            "generation_id": len(generations),
            "prompt_id": prompt_row["prompt_id"],
            "user_id": user["user_id"],
            "timestamp": timestamp,
            "session_date": timestamp.date(),
            "latency_ms": latency,
            "status": status,
            "cost_credits": cost,
            "retry_count": retry_count,
            "feedback": feedback,
            "downloaded": downloaded,
            "model_version": model_version,
            "token_count": token_count,
        })

    # Write to DuckDB
    conn.execute("DROP TABLE IF EXISTS raw_generations")
    conn.execute("""
        CREATE TABLE raw_generations (
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

    # Convert to native Python types (DuckDB doesn't like numpy types)
    conn.executemany(
        """INSERT INTO raw_generations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(int(g["generation_id"]), int(g["prompt_id"]), int(g["user_id"]), g["timestamp"],
          g["session_date"], int(g["latency_ms"]), g["status"], float(g["cost_credits"]),
          int(g["retry_count"]), g["feedback"], bool(g["downloaded"]), g["model_version"],
          int(g["token_count"]))
         for g in generations]
    )

    # Print summary stats
    print(f"\nGenerated {len(generations)} generation records:")

    # Weekly growth distribution
    weekly_stats = conn.execute("""
        SELECT
            CASE
                WHEN DAY(session_date) <= 7 THEN 'Week 1 (Dec 1-7)'
                WHEN DAY(session_date) <= 14 THEN 'Week 2 (Dec 8-14)'
                WHEN DAY(session_date) <= 21 THEN 'Week 3 (Dec 15-21)'
                ELSE 'Week 4 (Dec 22-31)'
            END as week,
            COUNT(*) as cnt,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
        FROM raw_generations
        GROUP BY week
        ORDER BY week
    """).fetchall()
    print("\nGeneration growth (weekly):")
    for week, cnt, pct in weekly_stats:
        bar = "█" * int(pct / 3)
        print(f"  {week}: {cnt:5d} ({pct:4.1f}%) {bar}")

    status_stats = conn.execute("""
        SELECT status, COUNT(*) as cnt,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
        FROM raw_generations GROUP BY status ORDER BY cnt DESC
    """).fetchall()
    print("\nStatus distribution:")
    for status, cnt, pct in status_stats:
        print(f"  {status}: {cnt} ({pct}%)")

    feedback_stats = conn.execute("""
        SELECT
            COUNT(*) FILTER (WHERE feedback = 'thumbs_up') as thumbs_up,
            COUNT(*) FILTER (WHERE feedback = 'thumbs_down') as thumbs_down,
            COUNT(*) FILTER (WHERE feedback IS NULL) as no_feedback
        FROM raw_generations
    """).fetchone()
    print(f"\nFeedback: 👍 {feedback_stats[0]} | 👎 {feedback_stats[1]} | No feedback: {feedback_stats[2]}")

    download_stats = conn.execute("""
        SELECT
            COUNT(*) FILTER (WHERE downloaded) as downloads,
            COUNT(*) FILTER (WHERE status = 'success') as successful
        FROM raw_generations
    """).fetchone()
    print(f"Downloads: {download_stats[0]} / {download_stats[1]} successful ({100*download_stats[0]//max(1,download_stats[1])}%)")

    conn.close()
    return len(generations)


if __name__ == "__main__":
    enrich_generations()
