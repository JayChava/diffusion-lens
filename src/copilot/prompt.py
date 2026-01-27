"""
Prompt templates for SQL Copilot.

Optimized for SQL generation with Qwen2.5 and similar models.
"""


def construct_sql_prompt(user_question: str, schema_text: str) -> str:
    """
    Construct prompt optimized for SQL generation.

    This format works well with:
    - SQLCoder models
    - Qwen2.5-Instruct
    - Llama-3-Instruct
    """
    prompt = f"""### Task
Generate a SQL query to answer the following question:
{user_question}

### Database Schema
The query will run on a DuckDB database with the following schema:
{schema_text}

### Key relationships
- fct_generations.user_id → dim_users.user_id
- fct_generations.prompt_id → dim_prompts.prompt_id
- fct_sessions contains aggregated metrics per session
- ftr_llm_analysis.prompt_id → dim_prompts.prompt_id (for LLM features like llm_domain, llm_art_style)
- ftr_text_embeddings.prompt_id → dim_prompts.prompt_id (for semantic search)

### Instructions
- Use DuckDB SQL syntax
- Use LIMIT instead of TOP
- Use ILIKE for case-insensitive string matching
- Use ROUND() for decimal formatting
- For status checks: 'success', 'timeout', 'safety_violation', 'rate_limited', 'model_error'
- For user tiers: 'free', 'pro', 'enterprise'
- Return ONLY the SQL query, no explanations or markdown

### SQL Query
SELECT"""

    return prompt


def construct_explanation_prompt(sql: str, schema_text: str) -> str:
    """
    Construct prompt to explain a SQL query.
    """
    prompt = f"""### Task
Explain what this SQL query does in plain English (2-3 sentences).

### Database Schema
{schema_text}

### SQL Query
{sql}

### Explanation
This query"""

    return prompt
