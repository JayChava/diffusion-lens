-- Test: No gaps in daily metrics
-- Business rule: Every day in the activity period should have metrics

with date_range as (
    select
        min(session_date) as min_date,
        max(session_date) as max_date,
        count(distinct session_date) as actual_days,
        datediff('day', min(session_date), max(session_date)) + 1 as expected_days
    from {{ ref('daily_metrics') }}
)
select *
from date_range
where actual_days != expected_days
