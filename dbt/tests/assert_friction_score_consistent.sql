-- Test: High error rate sessions should have high friction
-- Business rule: Friction score reflects reality

-- Sessions with 0% success rate should have friction_score >= 50
select
    session_id,
    success_rate_pct,
    friction_score,
    friction_category
from {{ ref('fct_sessions') }}
where success_rate_pct = 0
  and friction_score < 50
