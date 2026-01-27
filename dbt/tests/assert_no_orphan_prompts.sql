-- Test: No generations without valid prompt
-- Business rule: Every generation must have a prompt

select
    generation_id,
    prompt_id
from {{ ref('fct_generations') }} g
where not exists (
    select 1
    from {{ ref('dim_prompts') }} p
    where p.prompt_id = g.prompt_id
)
