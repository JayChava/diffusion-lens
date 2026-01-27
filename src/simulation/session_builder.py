"""
Session Builder - Groups generations into user sessions.

A session is defined as a sequence of generations from the same user
with no gap longer than SESSION_TIMEOUT_MINUTES between them.
"""

import duckdb

SESSION_TIMEOUT_MINUTES = 30


def build_sessions(db_path: str = "data/warehouse.duckdb") -> int:
    """
    Build sessions from raw_generations using SQL window functions.

    Session logic:
    - Group by user
    - If gap between generations > 30 min, start new session
    - Assign session_id to each generation

    Returns:
        Number of sessions created
    """
    conn = duckdb.connect(db_path)

    print("Building sessions from generations...")

    # Add session_id to generations using window functions
    # This is the "SQL over Python" approach from CLAUDE.md
    conn.execute("DROP TABLE IF EXISTS raw_sessions_staging")
    conn.execute(f"""
        CREATE TABLE raw_sessions_staging AS
        WITH ordered_gens AS (
            SELECT
                *,
                LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) as prev_timestamp
            FROM raw_generations
        ),
        session_starts AS (
            SELECT
                *,
                CASE
                    WHEN prev_timestamp IS NULL THEN 1
                    WHEN EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) / 60 > {SESSION_TIMEOUT_MINUTES} THEN 1
                    ELSE 0
                END as is_session_start
            FROM ordered_gens
        ),
        with_session_id AS (
            SELECT
                *,
                SUM(is_session_start) OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp
                    ROWS UNBOUNDED PRECEDING
                ) as user_session_num
            FROM session_starts
        )
        SELECT
            generation_id,
            prompt_id,
            user_id,
            -- Create globally unique session_id
            user_id * 10000 + user_session_num as session_id,
            timestamp,
            session_date,
            latency_ms,
            status,
            cost_credits,
            retry_count,
            feedback,
            downloaded,
            model_version,
            token_count
        FROM with_session_id
    """)

    # Replace raw_generations with session-enriched version
    conn.execute("DROP TABLE IF EXISTS raw_generations_backup")
    conn.execute("ALTER TABLE raw_generations RENAME TO raw_generations_backup")
    conn.execute("ALTER TABLE raw_sessions_staging RENAME TO raw_generations")
    conn.execute("DROP TABLE raw_generations_backup")

    # Count unique sessions
    session_count = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM raw_generations"
    ).fetchone()[0]

    # Session stats
    print(f"\nCreated {session_count} sessions")

    stats = conn.execute("""
        SELECT
            AVG(gens_per_session) as avg_gens,
            MAX(gens_per_session) as max_gens,
            MEDIAN(gens_per_session) as median_gens
        FROM (
            SELECT session_id, COUNT(*) as gens_per_session
            FROM raw_generations
            GROUP BY session_id
        )
    """).fetchone()
    print(f"Generations per session: avg={stats[0]:.1f}, median={stats[2]:.0f}, max={stats[1]}")

    # Session duration stats
    duration_stats = conn.execute("""
        SELECT
            AVG(duration_min) as avg_duration,
            MEDIAN(duration_min) as median_duration
        FROM (
            SELECT
                session_id,
                EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 60 as duration_min
            FROM raw_generations
            GROUP BY session_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()
    if duration_stats[0]:
        print(f"Session duration (multi-gen only): avg={duration_stats[0]:.1f}min, median={duration_stats[1]:.1f}min")

    conn.close()
    return session_count


if __name__ == "__main__":
    build_sessions()
