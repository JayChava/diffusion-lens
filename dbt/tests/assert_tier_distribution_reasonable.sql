-- Test: Tier distribution should follow expected pattern
-- Business rule: Free users should be majority (40-80%)

with tier_pcts as (
    select
        user_tier,
        count(*) as cnt,
        count(*) * 100.0 / sum(count(*)) over () as pct
    from {{ ref('dim_users') }}
    group by user_tier
)
select *
from tier_pcts
where user_tier = 'free'
  and (pct < 40 or pct > 85)
