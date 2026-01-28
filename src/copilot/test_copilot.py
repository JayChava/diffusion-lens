"""
Tests for SQL Copilot modules.

Run with: uv run pytest src/copilot/test_copilot.py -v
Run fast tests only: uv run pytest src/copilot/test_copilot.py -v -m "not slow"
"""

import pytest
from pathlib import Path

from .llm import clean_sql_response, validate_sql
from .prompt import construct_sql_prompt


# =============================================================================
# Tests for clean_sql_response (no model needed)
# =============================================================================

class TestCleanSqlResponse:
    """Test SQL response cleaning logic."""

    def test_basic_query(self):
        """Basic query passes through."""
        response = "user_id, COUNT(*) FROM fct_sessions GROUP BY user_id"
        result = clean_sql_response(response)
        assert result == "SELECT user_id, COUNT(*) FROM fct_sessions GROUP BY user_id;"

    def test_prepends_select(self):
        """SELECT is prepended if missing."""
        response = "* FROM dim_users LIMIT 10"
        result = clean_sql_response(response)
        assert result.startswith("SELECT")

    def test_removes_markdown_blocks(self):
        """Markdown code blocks are stripped."""
        response = "```sql\nuser_id FROM dim_users\n```"
        result = clean_sql_response(response)
        assert "```" not in result
        assert "SELECT" in result

    def test_adds_semicolon(self):
        """Semicolon is added if missing."""
        response = "* FROM dim_users"
        result = clean_sql_response(response)
        assert result.endswith(";")

    def test_truncates_after_semicolon(self):
        """Content after semicolon is removed (explanations, etc.)."""
        response = "* FROM dim_users; This query returns all users."
        result = clean_sql_response(response)
        assert result == "SELECT * FROM dim_users;"

    def test_strips_sql_prefix(self):
        """Removes 'sql' language prefix."""
        response = "sql\n* FROM dim_users"
        result = clean_sql_response(response)
        assert not result.lower().startswith("sql")


# =============================================================================
# Tests for validate_sql (no model needed)
# =============================================================================

class TestValidateSql:
    """Test SQL validation/safety logic."""

    def test_valid_select(self):
        """Valid SELECT query passes."""
        is_valid, error = validate_sql("SELECT * FROM dim_users;")
        assert is_valid is True
        assert error == ""

    def test_blocks_drop(self):
        """DROP is blocked."""
        is_valid, error = validate_sql("DROP TABLE dim_users;")
        assert is_valid is False
        assert "DROP" in error

    def test_blocks_delete(self):
        """DELETE is blocked."""
        is_valid, error = validate_sql("DELETE FROM dim_users;")
        assert is_valid is False
        assert "DELETE" in error

    def test_blocks_update(self):
        """UPDATE is blocked."""
        is_valid, error = validate_sql("UPDATE dim_users SET name = 'x';")
        assert is_valid is False
        assert "UPDATE" in error

    def test_blocks_insert(self):
        """INSERT is blocked."""
        is_valid, error = validate_sql("INSERT INTO dim_users VALUES (1);")
        assert is_valid is False
        assert "INSERT" in error

    def test_allows_keyword_in_column_name(self):
        """Keywords in column names are allowed (e.g., 'last_update')."""
        is_valid, error = validate_sql("SELECT last_update FROM dim_users;")
        assert is_valid is True

    def test_rejects_non_select(self):
        """Non-SELECT queries are rejected."""
        is_valid, error = validate_sql("SHOW TABLES;")
        assert is_valid is False
        assert "SELECT" in error


# =============================================================================
# Tests for prompt construction (no model needed)
# =============================================================================

class TestPromptConstruction:
    """Test prompt template construction."""

    def test_includes_question(self):
        """User question is included in prompt."""
        prompt = construct_sql_prompt("Show me all users", "CREATE TABLE users...")
        assert "Show me all users" in prompt

    def test_includes_schema(self):
        """Schema is included in prompt."""
        schema = "CREATE TABLE dim_users (user_id INT);"
        prompt = construct_sql_prompt("Get users", schema)
        assert "dim_users" in prompt

    def test_includes_date_context(self):
        """December 2025 date context is included."""
        prompt = construct_sql_prompt("Get data", "schema")
        assert "December 2025" in prompt

    def test_ends_with_select(self):
        """Prompt ends with SELECT for model to continue."""
        prompt = construct_sql_prompt("Get users", "schema")
        assert prompt.strip().endswith("SELECT")

    def test_includes_duckdb_hints(self):
        """DuckDB-specific syntax hints are included."""
        prompt = construct_sql_prompt("Get users", "schema")
        assert "LIMIT" in prompt
        assert "ILIKE" in prompt


# =============================================================================
# Integration test (requires model - slow)
# =============================================================================

@pytest.mark.slow
class TestSqlGeneration:
    """Integration tests that require loading the LLM model.

    Run with: uv run pytest src/copilot/test_copilot.py -v -m slow
    """

    @pytest.fixture(scope="class")
    def model_components(self):
        """Load model once for all tests in class."""
        from .llm import load_model
        return load_model()

    def test_generates_valid_sql(self, model_components):
        """Model generates valid SELECT query."""
        from .llm import generate_sql
        model, tokenizer, sampler = model_components

        prompt = construct_sql_prompt(
            "Count all users",
            "CREATE TABLE dim_users (user_id INT, user_tier VARCHAR);"
        )

        sql = generate_sql(prompt, model, tokenizer, sampler)

        assert sql.upper().startswith("SELECT")
        assert sql.endswith(";")
        is_valid, _ = validate_sql(sql)
        assert is_valid

    def test_handles_aggregation(self, model_components):
        """Model handles aggregation queries."""
        from .llm import generate_sql
        model, tokenizer, sampler = model_components

        prompt = construct_sql_prompt(
            "Show count of users by tier",
            "CREATE TABLE dim_users (user_id INT, user_tier VARCHAR);"
        )

        sql = generate_sql(prompt, model, tokenizer, sampler)

        assert "COUNT" in sql.upper()
        assert "GROUP BY" in sql.upper()


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    # Run fast tests by default
    pytest.main([__file__, "-v", "-m", "not slow"])
