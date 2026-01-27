-- Fact table for sessions
-- Aggregated session-level metrics with friction score

with session_metrics as (
    select
        session_id,
        user_id,
        min(generated_at) as session_start,
        max(generated_at) as session_end,
        min(session_date) as session_date,
        count(*) as total_generations,
        -- Success metrics
        sum(is_success) as successful_generations,
        avg(is_success) as success_rate,
        -- Error breakdown
        sum(is_timeout) as timeout_count,
        sum(is_safety_violation) as safety_violation_count,
        sum(is_rate_limited) as rate_limited_count,
        sum(is_model_error) as model_error_count,
        -- Latency
        avg(latency_ms) as avg_latency_ms,
        max(latency_ms) as max_latency_ms,
        -- Cost
        sum(cost_credits) as total_cost_credits,
        -- Retries
        sum(retry_count) as total_retries,
        avg(retry_count) as avg_retry_count,
        -- Feedback
        sum(is_thumbs_up) as thumbs_up_count,
        sum(is_thumbs_down) as thumbs_down_count,
        sum(is_downloaded) as download_count,
        -- User context
        max(user_tier) as user_tier,
        max(region) as region
    from {{ ref('fct_generations') }}
    group by session_id, user_id
),

session_with_friction as (
    select
        *,
        -- Error rate (0-1)
        1.0 - success_rate as error_rate,
        -- Retry rate (normalized 0-1, capped at 1)
        least(1.0, avg_retry_count / 2.0) as retry_rate_normalized,
        -- Latency score (normalized 0-1, based on 5s being "bad")
        least(1.0, avg_latency_ms / 5000.0) as latency_score_normalized,
        -- Session duration in minutes
        extract(epoch from (session_end - session_start)) / 60.0 as session_duration_min
    from session_metrics
)

select
    session_id,
    user_id,
    user_tier,
    region,
    session_start,
    session_end,
    session_date,
    round(session_duration_min, 2) as session_duration_min,
    total_generations,
    successful_generations,
    round(success_rate * 100, 1) as success_rate_pct,
    -- Error counts
    timeout_count,
    safety_violation_count,
    rate_limited_count,
    model_error_count,
    -- Latency
    round(avg_latency_ms, 0) as avg_latency_ms,
    max_latency_ms,
    -- Cost
    round(total_cost_credits, 4) as total_cost_credits,
    -- Retries
    total_retries,
    round(avg_retry_count, 2) as avg_retry_count,
    -- Feedback signals
    thumbs_up_count,
    thumbs_down_count,
    download_count,
    -- Friction Score (0-100 scale)
    -- Weighted composite: errors (3x), latency (2x), retries (1x)
    round(least(100, (
        (error_rate * 3.0 * 33.33) +
        (latency_score_normalized * 2.0 * 33.33) +
        (retry_rate_normalized * 1.0 * 33.33)
    )), 1) as friction_score,
    -- Friction category
    case
        when (error_rate * 3 + latency_score_normalized * 2 + retry_rate_normalized) / 6.0 < 0.1 then 'low'
        when (error_rate * 3 + latency_score_normalized * 2 + retry_rate_normalized) / 6.0 < 0.3 then 'medium'
        else 'high'
    end as friction_category
from session_with_friction
