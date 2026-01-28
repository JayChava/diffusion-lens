"""
LLM integration for SQL Copilot using MLX.

Provides SQL generation from natural language using local inference.
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# Global model cache (loaded once per session)
_model_cache = {}


def load_model(model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"):
    """
    Load and cache the LLM model.

    Returns:
        tuple: (model, tokenizer, sampler)
    """
    if model_name not in _model_cache:
        print(f"Loading model: {model_name}...")
        model, tokenizer = load(model_name)
        sampler = make_sampler(temp=0.1)  # Low temperature for deterministic SQL
        _model_cache[model_name] = (model, tokenizer, sampler)
        print("Model loaded!")

    return _model_cache[model_name]


def generate_sql(prompt: str, model=None, tokenizer=None, sampler=None,
                 model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit") -> str:
    """
    Generate SQL from a prompt using MLX.

    Args:
        prompt: The formatted prompt with schema and question
        model, tokenizer, sampler: Pre-loaded model components (optional)
        model_name: Model to load if components not provided

    Returns:
        str: Generated SQL query
    """
    # Load model if not provided
    if model is None or tokenizer is None or sampler is None:
        model, tokenizer, sampler = load_model(model_name)

    # Generate
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=300,
        sampler=sampler
    )

    # Clean response
    sql = clean_sql_response(response)
    return sql


def clean_sql_response(response: str) -> str:
    """
    Clean LLM output to extract valid SQL.
    """
    sql = response.strip()

    # Remove markdown code blocks if present
    if sql.startswith('```'):
        lines = sql.split('\n')
        # Find content between ``` markers
        in_block = False
        sql_lines = []
        for line in lines:
            if line.startswith('```'):
                in_block = not in_block
                continue
            if in_block or not line.startswith('```'):
                sql_lines.append(line)
        sql = '\n'.join(sql_lines).strip()

    # Remove 'sql' prefix if present
    if sql.lower().startswith('sql\n'):
        sql = sql[4:].strip()
    if sql.lower().startswith('sql '):
        sql = sql[4:].strip()

    # The prompt ends with "SELECT", so prepend if missing
    if not sql.upper().startswith('SELECT'):
        sql = 'SELECT ' + sql

    # Remove anything after semicolon (explanations, etc.)
    if ';' in sql:
        sql = sql[:sql.index(';') + 1]
    else:
        # Add semicolon if missing
        sql = sql.strip() + ';'

    return sql


# Safety validation
BLOCKED_KEYWORDS = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'CREATE', 'GRANT', 'REVOKE']


def validate_sql(sql: str) -> tuple:
    """
    Validate SQL for safety (block dangerous operations).

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    sql_upper = sql.upper()

    # Check for blocked keywords
    for keyword in BLOCKED_KEYWORDS:
        # Check for keyword as a word boundary (not part of column name)
        if f' {keyword} ' in f' {sql_upper} ':
            return False, f"Blocked: {keyword} operations not allowed"

    # Must start with SELECT
    if not sql_upper.strip().startswith('SELECT'):
        return False, "Only SELECT queries are allowed"

    return True, ""
