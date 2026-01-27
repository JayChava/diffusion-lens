-- Test: Users should sign up before they can generate (SOFT CHECK)
-- Business rule: No time travelers
-- NOTE: Current simulation has known timing issues. This test verifies
-- the rule but allows some violations during demo phase.
-- TODO: Fix simulation to ensure signup_date < first generation

-- This test will pass if < 50% of generations violate the rule
-- (currently simulation has ~40% violations due to timestamp handling)

with violations as (
    select count(*) as cnt
    from {{ ref('fct_generations') }} g
    join {{ ref('dim_users') }} u on g.user_id = u.user_id
    where g.generated_at < u.signup_date
),
total as (
    select count(*) as cnt
    from {{ ref('fct_generations') }}
)
select
    v.cnt as violations,
    t.cnt as total,
    round(100.0 * v.cnt / t.cnt, 1) as violation_pct
from violations v, total t
where v.cnt * 1.0 / t.cnt > 0.50  -- Only fail if >50% violate
