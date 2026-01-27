-- Staging model for prompts
-- Cleans raw_prompts and adds derived fields

select
    prompt_id,
    prompt as prompt_text,
    generation_seed,
    step,
    cfg,
    sampler,
    width,
    height,
    -- Derived fields
    length(prompt) as char_count,
    array_length(string_split(prompt, ' ')) as token_count
from {{ source('raw', 'raw_prompts') }}
