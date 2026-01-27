-- Test: No generations without valid user
-- Business rule: Every generation must belong to a real user

select
    generation_id,
    user_id
from {{ ref('fct_generations') }} g
where not exists (
    select 1
    from {{ ref('dim_users') }} u
    where u.user_id = g.user_id
)
