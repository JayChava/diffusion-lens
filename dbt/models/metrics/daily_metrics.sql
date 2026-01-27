-- Daily metrics summary
-- Time series data for dashboard trends

select
    session_date,
    count(distinct user_id) as active_users,
    count(distinct session_id) as total_sessions,
    sum(total_generations) as total_generations,
    -- Friction
    round(avg(friction_score), 1) as avg_friction_score,
    -- Success
    round(avg(success_rate_pct), 1) as avg_success_rate_pct,
    sum(successful_generations) as successful_generations,
    -- Errors
    sum(timeout_count) as timeouts,
    sum(safety_violation_count) as safety_violations,
    sum(rate_limited_count) as rate_limits,
    sum(model_error_count) as model_errors,
    -- Latency
    round(avg(avg_latency_ms), 0) as avg_latency_ms,
    -- Cost
    round(sum(total_cost_credits), 2) as total_cost_credits,
    -- Engagement
    sum(thumbs_up_count) as thumbs_up,
    sum(thumbs_down_count) as thumbs_down,
    sum(download_count) as downloads
from {{ ref('fct_sessions') }}
group by session_date
order by session_date
