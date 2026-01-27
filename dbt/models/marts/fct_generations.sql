-- Fact table for generations
-- Core transaction table with all metrics

select
    g.generation_id,
    g.user_id,
    g.prompt_id,
    g.session_id,
    g.generated_at,
    g.session_date,
    -- Metrics
    g.latency_ms,
    g.status,
    g.cost_credits,
    g.retry_count,
    g.token_count,
    g.model_version,
    -- Feedback signals
    g.feedback,
    g.downloaded,
    -- Derived flags
    case when g.status = 'success' then 1 else 0 end as is_success,
    case when g.status = 'timeout' then 1 else 0 end as is_timeout,
    case when g.status = 'safety_violation' then 1 else 0 end as is_safety_violation,
    case when g.status = 'rate_limited' then 1 else 0 end as is_rate_limited,
    case when g.status = 'model_error' then 1 else 0 end as is_model_error,
    case when g.feedback = 'thumbs_up' then 1 else 0 end as is_thumbs_up,
    case when g.feedback = 'thumbs_down' then 1 else 0 end as is_thumbs_down,
    case when g.downloaded then 1 else 0 end as is_downloaded,
    -- Denormalized dimensions for easy querying
    u.user_tier,
    u.region,
    p.prompt_length_category
from {{ ref('stg_generations') }} g
left join {{ ref('dim_users') }} u on g.user_id = u.user_id
left join {{ ref('dim_prompts') }} p on g.prompt_id = p.prompt_id
