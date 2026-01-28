-- Dimension table for prompts
-- Core prompt attributes only
-- ML enrichment stored separately in raw_prompt_enrichments table (not managed by dbt)
-- Access enriched data via ftr_llm_analysis and ftr_text_embeddings views

select
    prompt_id,
    prompt_text,
    char_count,
    token_count,
    generation_seed,
    step,
    cfg,
    sampler,
    width,
    height,
    -- Prompt length category (useful for basic segmentation)
    case
        when token_count <= 10 then 'short'
        when token_count <= 30 then 'medium'
        when token_count <= 60 then 'long'
        else 'very_long'
    end as prompt_length_category

from {{ ref('stg_prompts') }}
