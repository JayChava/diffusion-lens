"""
SQL Copilot Sample - Quick demo of the copilot.

Usage:
    uv run python -m src.copilot.sample
"""

import time
from pathlib import Path

import duckdb

from .llm import load_model, generate_sql, validate_sql
from .prompt import construct_sql_prompt
from .schema import get_full_schema

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"

# Example questions to demo
EXAMPLES = [
    "Count users by tier",
    "Show error rate by user tier",
    "Top 5 users by total session cost",
    "Average latency by status",
]


def main():
    # Load model
    print("Loading model...")
    model, tokenizer, sampler = load_model()

    # Connect to database
    con = duckdb.connect(str(DB_PATH), read_only=True)
    schema = get_full_schema(con)

    # Run examples
    for question in EXAMPLES:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)

        # Generate SQL
        start = time.time()
        prompt = construct_sql_prompt(question, schema)
        sql = generate_sql(prompt, model, tokenizer, sampler)
        print(f"\nSQL ({time.time() - start:.1f}s):\n{sql}\n")

        # Validate and execute
        is_valid, error = validate_sql(sql)
        if not is_valid:
            print(f"❌ {error}")
            continue

        try:
            result = con.execute(sql).fetchdf()
            print(result.to_string())
        except Exception as e:
            print(f"❌ {e}")

    con.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
