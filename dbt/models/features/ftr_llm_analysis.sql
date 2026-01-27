-- Feature table: LLM-extracted prompt analysis + image paths
-- Source: Pre-computed by src/enrichment/precompute_llm.py
-- Model: Qwen2.5-1.5B-Instruct-4bit via MLX
-- Storage: prompt_enrichments table (separate from dbt-managed tables)

{{ config(materialized='view') }}

select
    p.prompt_id,
    p.prompt_text,

    -- LLM-extracted features (from prompt_enrichments)
    e.llm_domain,
    e.llm_art_style,
    e.llm_complexity_score,

    -- Image path (from prompt_enrichments - survives dbt runs)
    e.image_path,

    -- Feature metadata
    case when e.llm_domain is not null then true else false end as has_llm_features,
    case when e.image_path is not null then true else false end as has_image,
    'Qwen2.5-1.5B-Instruct-4bit' as llm_model,
    'MLX (Apple Silicon)' as inference_engine,
    e.llm_enriched_at

from {{ ref('dim_prompts') }} p
left join prompt_enrichments e on p.prompt_id = e.prompt_id
