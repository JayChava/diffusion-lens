"""
Schema extractor for SQL Copilot.

Extracts table schemas from DuckDB and formats them for LLM consumption.
"""

import duckdb


def get_table_schema(con: duckdb.DuckDBPyConnection, table_name: str) -> str:
    """
    Extract table schema and format as CREATE TABLE statement.
    LLMs understand this format well for SQL generation.
    """
    schema_df = con.execute(f"DESCRIBE {table_name}").fetchdf()

    columns = []
    for _, row in schema_df.iterrows():
        col_name = row['column_name']
        col_type = row['column_type']
        # Mark vector embedding columns (FLOAT[N] arrays)
        if 'FLOAT[' in col_type and 'embedding' in col_name.lower():
            columns.append(f"    {col_name} FLOAT[]  -- vector embedding, use array_cosine_similarity()")
        else:
            columns.append(f"    {col_name} {col_type}")

    schema_text = f"CREATE TABLE {table_name} (\n"
    schema_text += ",\n".join(columns)
    schema_text += "\n);"

    return schema_text


def get_full_schema(con: duckdb.DuckDBPyConnection) -> str:
    """
    Get schema for all analytics tables.
    Includes helpful comments for the LLM.
    """
    schema_parts = []

    # Add relationship hints at the top
    schema_parts.append("""-- IMPORTANT RELATIONSHIPS:
-- fct_sessions.user_id -> dim_users.user_id (JOIN to get device_type, signup_date)
-- fct_generations.user_id -> dim_users.user_id
-- fct_generations.prompt_id -> dim_prompts.prompt_id
-- For device_type analysis, JOIN fct_sessions with dim_users""")

    # Core dimension and fact tables
    tables_with_comments = {
        'dim_users': '-- User dimension: user_tier ("free"/"pro"/"enterprise"), device_type ("desktop"/"mobile"/"tablet"), region',
        'dim_prompts': '-- Prompt dimension: prompt_text, token_count, complexity metrics',
        'fct_generations': '-- Fact table: each row is one image generation attempt with status, latency_ms, cost_credits',
        'fct_sessions': '-- Session aggregates: friction_score (0-1, higher=worse), success_rate_pct, total_cost_credits. NOTE: device_type is in dim_users, JOIN on user_id',
    }

    for table_name, comment in tables_with_comments.items():
        try:
            schema = get_table_schema(con, table_name)
            schema_parts.append(f"{comment}\n{schema}")
        except Exception:
            pass  # Skip if table doesn't exist

    # Feature views with LLM-extracted data (joins dim_prompts with raw_prompt_enrichments)
    feature_views = {
        'ftr_llm_analysis': '-- LLM-extracted features: llm_domain, llm_art_style, llm_complexity_score, image_path',
        'ftr_text_embeddings': '-- Text embeddings for semantic search (use array_cosine_similarity)',
    }

    for view_name, comment in feature_views.items():
        try:
            schema = get_table_schema(con, view_name)
            schema_parts.append(f"{comment}\n{schema}")
        except Exception:
            pass  # Skip if view doesn't exist

    # Metrics tables (pre-aggregated for fast queries)
    metrics_tables = {
        'user_friction_summary': '-- Metrics by user tier: avg_friction_score, total_sessions, avg_success_rate_pct, total_cost_credits',
        'daily_metrics': '-- Daily metrics: session_date, total_generations, active_users, avg_friction_score, total_cost_credits',
    }

    for table_name, comment in metrics_tables.items():
        try:
            schema = get_table_schema(con, table_name)
            schema_parts.append(f"{comment}\n{schema}")
        except Exception:
            pass  # Skip if table doesn't exist

    return "\n\n".join(schema_parts)


def get_sample_data(con: duckdb.DuckDBPyConnection, table_name: str, limit: int = 3) -> str:
    """Get sample data from a table for context."""
    try:
        df = con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchdf()
        return df.to_string()
    except Exception:
        return ""
