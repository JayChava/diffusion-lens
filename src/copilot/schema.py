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

    # Core dimension and fact tables
    tables_with_comments = {
        'dim_users': '-- User dimension: user_tier is "free", "pro", or "enterprise"',
        'dim_prompts': '-- Prompt dimension: basic prompt metadata',
        'fct_generations': '-- Fact table: each row is one image generation attempt',
        'fct_sessions': '-- Session aggregates: friction_score, success_rate, costs per session',
    }

    for table_name, comment in tables_with_comments.items():
        try:
            schema = get_table_schema(con, table_name)
            schema_parts.append(f"{comment}\n{schema}")
        except Exception:
            pass  # Skip if table doesn't exist

    # Feature views with LLM-extracted data (joins dim_prompts with prompt_enrichments)
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

    # Add useful aggregated views if they exist
    try:
        con.execute("SELECT 1 FROM user_friction_summary LIMIT 1")
        schema_parts.append("-- Aggregated metrics by user tier\n-- Table: user_friction_summary (pre-computed)")
    except Exception:
        pass

    return "\n\n".join(schema_parts)


def get_sample_data(con: duckdb.DuckDBPyConnection, table_name: str, limit: int = 3) -> str:
    """Get sample data from a table for context."""
    try:
        df = con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchdf()
        return df.to_string()
    except Exception:
        return ""
