-- Staging model for users
-- Cleans and standardizes raw_users

select
    user_id,
    user_tier,
    signup_date,
    cohort_week,
    region,
    device_type
from {{ source('raw', 'raw_users') }}
