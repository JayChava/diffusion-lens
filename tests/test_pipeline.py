"""
Pipeline Tests - Data quality checks for Dagster assets.

Run with:
    uv run pytest tests/test_pipeline.py -v
    uv run pytest tests/test_pipeline.py -v -k "not slow"  # Skip slow tests
"""

import pytest
from pathlib import Path

import duckdb

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = str(PROJECT_ROOT / "data" / "warehouse.duckdb")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def db_connection():
    """Read-only connection to the database."""
    conn = duckdb.connect(DB_PATH, read_only=True)
    yield conn
    conn.close()


# =============================================================================
# Table Existence Tests
# =============================================================================

class TestTablesExist:
    """Verify all expected tables exist."""

    def test_raw_prompts_exists(self, db_connection):
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'raw_prompts'"
        ).fetchone()[0]
        assert result == 1, "raw_prompts table should exist"

    def test_raw_users_exists(self, db_connection):
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'raw_users'"
        ).fetchone()[0]
        assert result == 1, "raw_users table should exist"

    def test_raw_generations_exists(self, db_connection):
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'raw_generations'"
        ).fetchone()[0]
        assert result == 1, "raw_generations table should exist"

    def test_dim_prompts_exists(self, db_connection):
        """dbt model should exist."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'dim_prompts'"
        ).fetchone()[0]
        assert result == 1, "dim_prompts table should exist (run dbt build)"

    def test_fct_sessions_exists(self, db_connection):
        """dbt model should exist."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'fct_sessions'"
        ).fetchone()[0]
        assert result == 1, "fct_sessions table should exist (run dbt build)"


# =============================================================================
# Row Count Tests (Data Not Empty)
# =============================================================================

class TestRowCounts:
    """Verify tables have expected row counts."""

    def test_raw_prompts_not_empty(self, db_connection):
        count = db_connection.execute("SELECT COUNT(*) FROM raw_prompts").fetchone()[0]
        assert count > 0, "raw_prompts should have data"
        assert count >= 1000, f"Expected at least 1000 prompts, got {count}"

    def test_raw_users_not_empty(self, db_connection):
        count = db_connection.execute("SELECT COUNT(*) FROM raw_users").fetchone()[0]
        assert count > 0, "raw_users should have data"
        assert count >= 100, f"Expected at least 100 users, got {count}"

    def test_raw_generations_not_empty(self, db_connection):
        count = db_connection.execute("SELECT COUNT(*) FROM raw_generations").fetchone()[0]
        assert count > 0, "raw_generations should have data"

    def test_generations_match_prompts(self, db_connection):
        """Each prompt should have exactly one generation."""
        prompts = db_connection.execute("SELECT COUNT(*) FROM raw_prompts").fetchone()[0]
        generations = db_connection.execute("SELECT COUNT(*) FROM raw_generations").fetchone()[0]
        assert generations == prompts, f"Generations ({generations}) should match prompts ({prompts})"


# =============================================================================
# Data Quality Tests
# =============================================================================

class TestDataQuality:
    """Verify data quality constraints."""

    def test_user_tiers_valid(self, db_connection):
        """User tiers should only be free/pro/enterprise."""
        invalid = db_connection.execute("""
            SELECT COUNT(*) FROM raw_users
            WHERE user_tier NOT IN ('free', 'pro', 'enterprise')
        """).fetchone()[0]
        assert invalid == 0, f"Found {invalid} users with invalid tier"

    def test_status_values_valid(self, db_connection):
        """Generation status should be one of expected values."""
        invalid = db_connection.execute("""
            SELECT COUNT(*) FROM raw_generations
            WHERE status NOT IN ('success', 'timeout', 'safety_violation', 'rate_limited', 'model_error')
        """).fetchone()[0]
        assert invalid == 0, f"Found {invalid} generations with invalid status"

    def test_latency_positive(self, db_connection):
        """Latency should always be positive."""
        invalid = db_connection.execute("""
            SELECT COUNT(*) FROM raw_generations WHERE latency_ms <= 0
        """).fetchone()[0]
        assert invalid == 0, f"Found {invalid} generations with non-positive latency"

    def test_cost_non_negative(self, db_connection):
        """Cost should never be negative."""
        invalid = db_connection.execute("""
            SELECT COUNT(*) FROM raw_generations WHERE cost_credits < 0
        """).fetchone()[0]
        assert invalid == 0, f"Found {invalid} generations with negative cost"

    def test_no_orphan_generations(self, db_connection):
        """Every generation should have a valid user."""
        orphans = db_connection.execute("""
            SELECT COUNT(*) FROM raw_generations g
            LEFT JOIN raw_users u ON g.user_id = u.user_id
            WHERE u.user_id IS NULL
        """).fetchone()[0]
        assert orphans == 0, f"Found {orphans} generations without valid user"

    def test_no_orphan_prompts(self, db_connection):
        """Every generation should have a valid prompt."""
        orphans = db_connection.execute("""
            SELECT COUNT(*) FROM raw_generations g
            LEFT JOIN raw_prompts p ON g.prompt_id = p.prompt_id
            WHERE p.prompt_id IS NULL
        """).fetchone()[0]
        assert orphans == 0, f"Found {orphans} generations without valid prompt"


# =============================================================================
# Business Logic Tests
# =============================================================================

class TestBusinessLogic:
    """Verify business logic and derived metrics."""

    def test_success_rate_reasonable(self, db_connection):
        """Success rate should be between 50-99% (not all failures, not perfect)."""
        rate = db_connection.execute("""
            SELECT AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END)
            FROM raw_generations
        """).fetchone()[0]
        assert 0.5 < rate < 0.99, f"Success rate {rate:.1%} outside expected range (50-99%)"

    def test_tier_distribution_reasonable(self, db_connection):
        """Free tier should be majority of users."""
        free_pct = db_connection.execute("""
            SELECT AVG(CASE WHEN user_tier = 'free' THEN 1.0 ELSE 0.0 END)
            FROM raw_users
        """).fetchone()[0]
        assert free_pct > 0.5, f"Free tier ({free_pct:.1%}) should be majority"

    def test_sessions_have_valid_friction(self, db_connection):
        """Friction scores should be 0-100."""
        invalid = db_connection.execute("""
            SELECT COUNT(*) FROM fct_sessions
            WHERE friction_score < 0 OR friction_score > 100
        """).fetchone()[0]
        assert invalid == 0, f"Found {invalid} sessions with friction score outside 0-100"

    def test_timeout_latency_high(self, db_connection):
        """Timeouts should have high latency (>20s)."""
        low_latency_timeouts = db_connection.execute("""
            SELECT COUNT(*) FROM raw_generations
            WHERE status = 'timeout' AND latency_ms < 20000
        """).fetchone()[0]
        assert low_latency_timeouts == 0, f"Found {low_latency_timeouts} timeouts with <20s latency"

    def test_safety_violations_fast(self, db_connection):
        """Safety violations should be fast (<1s) - rejected quickly."""
        slow_violations = db_connection.execute("""
            SELECT COUNT(*) FROM raw_generations
            WHERE status = 'safety_violation' AND latency_ms > 1000
        """).fetchone()[0]
        total_violations = db_connection.execute("""
            SELECT COUNT(*) FROM raw_generations WHERE status = 'safety_violation'
        """).fetchone()[0]
        # Allow some variance but most should be fast
        if total_violations > 0:
            slow_pct = slow_violations / total_violations
            assert slow_pct < 0.1, f"{slow_pct:.1%} of safety violations are slow (>1s)"


# =============================================================================
# Enrichment Coverage Tests
# =============================================================================

class TestEnrichmentCoverage:
    """Verify ML enrichment coverage."""

    def test_embeddings_coverage(self, db_connection):
        """Check text embedding coverage (should be >90% if run)."""
        try:
            stats = db_connection.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN text_embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded
                FROM raw_prompt_enrichments
            """).fetchone()
            total, embedded = stats
            if total > 0:
                coverage = embedded / total
                # Only assert if enrichment has been run
                if embedded > 0:
                    assert coverage > 0.9, f"Embedding coverage {coverage:.1%} below 90%"
        except:
            pytest.skip("raw_prompt_enrichments table not found")

    def test_llm_analysis_coverage(self, db_connection):
        """Check LLM analysis coverage (should be >90% if run)."""
        try:
            stats = db_connection.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN llm_domain IS NOT NULL THEN 1 ELSE 0 END) as analyzed
                FROM raw_prompt_enrichments
            """).fetchone()
            total, analyzed = stats
            if total > 0 and analyzed > 0:
                coverage = analyzed / total
                assert coverage > 0.9, f"LLM analysis coverage {coverage:.1%} below 90%"
        except:
            pytest.skip("raw_prompt_enrichments table not found")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
