-- User friction summary
-- Aggregated metrics by user tier for dashboard

select
    user_tier,
    count(distinct user_id) as total_users,
    count(*) as total_sessions,
    sum(total_generations) as total_generations,
    -- Friction
    round(avg(friction_score), 1) as avg_friction_score,
    sum(case when friction_category = 'high' then 1 else 0 end) as high_friction_sessions,
    sum(case when friction_category = 'medium' then 1 else 0 end) as medium_friction_sessions,
    sum(case when friction_category = 'low' then 1 else 0 end) as low_friction_sessions,
    -- Success
    round(avg(success_rate_pct), 1) as avg_success_rate_pct,
    -- Latency
    round(avg(avg_latency_ms), 0) as avg_latency_ms,
    -- Cost
    round(sum(total_cost_credits), 2) as total_cost_credits,
    round(avg(total_cost_credits), 2) as avg_session_cost,
    -- Engagement
    round(avg(total_generations), 1) as avg_generations_per_session,
    sum(thumbs_up_count) as total_thumbs_up,
    sum(thumbs_down_count) as total_thumbs_down,
    sum(download_count) as total_downloads,
    -- Feedback rate
    round(100.0 * sum(thumbs_up_count + thumbs_down_count) / sum(total_generations), 1) as feedback_rate_pct,
    -- Download rate (of successful only)
    round(100.0 * sum(download_count) / nullif(sum(successful_generations), 0), 1) as download_rate_pct
from {{ ref('fct_sessions') }}
group by user_tier
order by total_users desc
