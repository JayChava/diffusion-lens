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

### Current Date Context
- The data is from December 2025
- For "December" or "this month", use: generated_at >= '2025-12-01' AND generated_at < '2026-01-01'

### Key relationships
- fct_sessions.user_id → dim_users.user_id (JOIN to get device_type, signup_date)
- fct_generations.user_id → dim_users.user_id
- fct_generations.prompt_id → dim_prompts.prompt_id
- fct_sessions contains aggregated metrics per session (friction_score, success_rate_pct, total_cost_credits)
- IMPORTANT: fct_generations already contains user_tier and region (denormalized). No JOIN needed for user_tier analysis on fct_generations.
- IMPORTANT: device_type is ONLY in dim_users, not in fct_sessions. To analyze by device_type, JOIN fct_sessions with dim_users on user_id
- ftr_llm_analysis.prompt_id → dim_prompts.prompt_id (for LLM features like llm_domain, llm_art_style)

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
