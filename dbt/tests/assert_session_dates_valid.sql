-- Test: Session start must be before or equal to session end
-- Business rule: Time flows forward

select
    session_id,
    session_start,
    session_end
from {{ ref('fct_sessions') }}
where session_start > session_end
