-- Dimension table for users
-- Enriched with aggregated activity metrics

with user_stats as (
    select
        user_id,
        count(*) as total_generations,
        sum(case when status = 'success' then 1 else 0 end) as successful_generations,
        sum(cost_credits) as lifetime_cost,
        min(generated_at) as first_generation_at,
        max(generated_at) as last_generation_at,
        count(distinct session_id) as total_sessions
    from {{ ref('stg_generations') }}
    group by user_id
)

select
    u.user_id,
    u.user_tier,
    u.signup_date,
    u.cohort_week,
    u.region,
    u.device_type,
    -- Activity metrics
    coalesce(s.total_generations, 0) as total_generations,
    coalesce(s.successful_generations, 0) as successful_generations,
    round(coalesce(s.lifetime_cost, 0), 2) as lifetime_cost,
    coalesce(s.total_sessions, 0) as total_sessions,
    s.first_generation_at,
    s.last_generation_at,
    -- Derived
    case
        when s.total_generations > 0
        then round(100.0 * s.successful_generations / s.total_generations, 1)
        else 0
    end as success_rate_pct
from {{ ref('stg_users') }} u
left join user_stats s on u.user_id = s.user_id
