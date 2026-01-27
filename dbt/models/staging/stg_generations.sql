-- Staging model for generations
-- Cleans raw_generations and standardizes types

select
    generation_id,
    prompt_id,
    user_id,
    cast(session_id as integer) as session_id,
    timestamp as generated_at,
    session_date,
    latency_ms,
    status,
    cost_credits,
    retry_count,
    feedback,
    downloaded,
    model_version,
    token_count
from {{ source('raw', 'raw_generations') }}
