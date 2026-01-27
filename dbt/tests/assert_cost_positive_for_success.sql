-- Test: Successful generations should have positive cost
-- Business rule: No free lunches for successful generations

select
    generation_id,
    status,
    cost_credits
from {{ ref('fct_generations') }}
where status = 'success'
  and cost_credits <= 0
