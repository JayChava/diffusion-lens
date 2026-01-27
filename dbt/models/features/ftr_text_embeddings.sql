-- Feature table: Text embeddings for semantic search
-- Source: Pre-computed by src/enrichment/precompute_embeddings.py
-- Model: all-MiniLM-L6-v2 (384 dimensions)
-- Storage: prompt_enrichments table (separate from dbt-managed tables)

{{ config(materialized='view') }}

select
    p.prompt_id,
    p.prompt_text,

    -- Embeddings (from prompt_enrichments)
    e.text_embedding,

    -- Embedding metadata
    case when e.text_embedding is not null then true else false end as has_embedding,
    384 as embedding_dimensions,
    'all-MiniLM-L6-v2' as embedding_model,
    e.embedding_enriched_at

from {{ ref('dim_prompts') }} p
left join prompt_enrichments e on p.prompt_id = e.prompt_id
