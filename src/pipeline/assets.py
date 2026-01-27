"""
Dagster Assets - Data pipeline assets for GenAI Session Analyzer.

Asset DAG:
    raw_prompts (DiffusionDB) ──┬──> raw_generations ──> sessions_built
                                │
    raw_users (synthetic) ──────┘
                                │
    raw_prompts ────────────────┼──> text_embeddings (semantic search)
                                │
                                ├──> llm_prompt_analysis (domain/style/complexity)
                                │
                                └──> prompt_images (blob storage)
"""

from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue

from src.ingestion.diffusiondb_loader import load_diffusiondb_prompts
from src.simulation.user_generator import generate_users
from src.simulation.telemetry_enricher import enrich_generations
from src.simulation.session_builder import build_sessions

import duckdb
from pathlib import Path


# Configuration - use absolute path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = str(PROJECT_ROOT / "data" / "warehouse.duckdb")
NUM_PROMPTS = 10_000
NUM_USERS = 500
ACTIVITY_YEAR = 2025
ACTIVITY_MONTH = 12


@asset(
    group_name="ingestion",
    description="Raw prompts loaded from DiffusionDB (HuggingFace)",
)
def raw_prompts(context: AssetExecutionContext) -> MaterializeResult:
    """Load prompts from DiffusionDB into DuckDB."""

    context.log.info(f"Loading {NUM_PROMPTS:,} prompts from DiffusionDB...")
    row_count = load_diffusiondb_prompts(num_rows=NUM_PROMPTS, db_path=DB_PATH)

    # Get sample for metadata
    conn = duckdb.connect(DB_PATH, read_only=True)
    sample = conn.execute("SELECT prompt FROM raw_prompts LIMIT 3").fetchall()
    conn.close()

    return MaterializeResult(
        metadata={
            "row_count": row_count,
            "source": "poloclub/diffusiondb",
            "sample_prompts": MetadataValue.md("\n".join([f"- {s[0][:80]}..." for s in sample])),
        }
    )


@asset(
    group_name="simulation",
    description="Synthetic users with tier distribution and growth pattern (free/pro/enterprise)",
)
def raw_users(context: AssetExecutionContext) -> MaterializeResult:
    """Generate synthetic users with exponential signup growth throughout the month."""

    context.log.info(f"Generating {NUM_USERS} synthetic users with growth pattern...")
    row_count = generate_users(
        num_users=NUM_USERS,
        db_path=DB_PATH,
        year=ACTIVITY_YEAR,
        month=ACTIVITY_MONTH
    )

    # Get tier distribution for metadata
    conn = duckdb.connect(DB_PATH, read_only=True)
    tiers = conn.execute("""
        SELECT user_tier, COUNT(*) as cnt
        FROM raw_users
        GROUP BY user_tier
        ORDER BY cnt DESC
    """).fetchall()

    # Get weekly signup growth
    weekly = conn.execute("""
        SELECT
            CASE
                WHEN DAY(signup_date) <= 7 THEN 'week1'
                WHEN DAY(signup_date) <= 14 THEN 'week2'
                WHEN DAY(signup_date) <= 21 THEN 'week3'
                ELSE 'week4'
            END as week,
            COUNT(*) as cnt
        FROM raw_users
        GROUP BY week
        ORDER BY week
    """).fetchall()
    conn.close()

    tier_breakdown = {tier: cnt for tier, cnt in tiers}
    weekly_signups = {w: c for w, c in weekly}

    return MaterializeResult(
        metadata={
            "row_count": row_count,
            "tier_distribution": MetadataValue.json(tier_breakdown),
            "weekly_signup_growth": MetadataValue.json(weekly_signups),
            "activity_period": f"{ACTIVITY_YEAR}-{ACTIVITY_MONTH:02d}",
        }
    )


@asset(
    group_name="simulation",
    deps=[raw_prompts, raw_users],
    description="Enriched generations with telemetry and growth pattern (latency, status, feedback, downloads)",
)
def raw_generations(context: AssetExecutionContext) -> MaterializeResult:
    """Enrich prompts with synthetic telemetry following platform growth pattern."""

    context.log.info(f"Enriching generations for {ACTIVITY_YEAR}-{ACTIVITY_MONTH:02d} with growth pattern...")
    row_count = enrich_generations(
        db_path=DB_PATH,
        year=ACTIVITY_YEAR,
        month=ACTIVITY_MONTH,
    )

    # Get stats for metadata
    conn = duckdb.connect(DB_PATH, read_only=True)
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
            SUM(CASE WHEN feedback IS NOT NULL THEN 1 ELSE 0 END) as feedback_count,
            SUM(CASE WHEN downloaded THEN 1 ELSE 0 END) as downloads,
            ROUND(SUM(cost_credits), 2) as total_cost
        FROM raw_generations
    """).fetchone()

    # Get weekly generation growth
    weekly = conn.execute("""
        SELECT
            CASE
                WHEN DAY(session_date) <= 7 THEN 'week1'
                WHEN DAY(session_date) <= 14 THEN 'week2'
                WHEN DAY(session_date) <= 21 THEN 'week3'
                ELSE 'week4'
            END as week,
            COUNT(*) as cnt
        FROM raw_generations
        GROUP BY week
        ORDER BY week
    """).fetchall()
    conn.close()

    weekly_gens = {w: c for w, c in weekly}

    return MaterializeResult(
        metadata={
            "row_count": row_count,
            "success_count": stats[1],
            "success_rate": f"{100 * stats[1] / stats[0]:.1f}%",
            "feedback_count": stats[2],
            "download_count": stats[3],
            "total_cost_credits": stats[4],
            "weekly_generation_growth": MetadataValue.json(weekly_gens),
            "activity_period": f"{ACTIVITY_YEAR}-{ACTIVITY_MONTH:02d}",
        }
    )


@asset(
    group_name="simulation",
    deps=[raw_generations],
    description="Sessions built from generations (30-min timeout window)",
)
def sessions_built(context: AssetExecutionContext) -> MaterializeResult:
    """Build sessions from raw generations."""

    context.log.info("Building sessions from generations...")
    session_count = build_sessions(db_path=DB_PATH)

    # Get session stats
    conn = duckdb.connect(DB_PATH, read_only=True)
    stats = conn.execute("""
        SELECT
            COUNT(DISTINCT session_id) as sessions,
            AVG(gens) as avg_gens_per_session,
            MAX(gens) as max_gens_per_session
        FROM (
            SELECT session_id, COUNT(*) as gens
            FROM raw_generations
            GROUP BY session_id
        )
    """).fetchone()
    conn.close()

    return MaterializeResult(
        metadata={
            "session_count": session_count,
            "avg_generations_per_session": round(stats[1], 2),
            "max_generations_per_session": stats[2],
        }
    )


# =============================================================================
# ML Enrichment Assets (pre-computed features)
# =============================================================================

@asset(
    group_name="ml_enrichment",
    deps=[raw_prompts],
    description="Text embeddings for semantic search (all-MiniLM-L6-v2, 384-dim)",
)
def text_embeddings(context: AssetExecutionContext) -> MaterializeResult:
    """
    Pre-computed text embeddings for semantic prompt search.

    Run manually: uv run python -m src.enrichment.precompute_embeddings
    """
    conn = duckdb.connect(DB_PATH, read_only=True)

    total = conn.execute("SELECT COUNT(*) FROM dim_prompts").fetchone()[0]

    # Check prompt_enrichments table for embeddings
    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_name = 'prompt_enrichments'").fetchall()
    has_enrichment_table = len(tables) > 0

    embedded = 0
    if has_enrichment_table:
        embedded = conn.execute("""
            SELECT COUNT(*) FROM prompt_enrichments WHERE text_embedding IS NOT NULL
        """).fetchone()[0] or 0

    conn.close()

    coverage = embedded / total * 100 if total > 0 else 0
    context.log.info(f"Text embeddings: {embedded:,}/{total:,} ({coverage:.1f}%)")

    return MaterializeResult(
        metadata={
            "total_prompts": total,
            "embedded_count": embedded,
            "coverage_pct": f"{coverage:.1f}%",
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "index_type": "HNSW (DuckDB VSS)",
            "run_command": "uv run python -m src.enrichment.precompute_embeddings",
        }
    )


@asset(
    group_name="ml_enrichment",
    deps=[raw_prompts],
    description="LLM-extracted features: domain, art_style, complexity (Qwen2.5 via MLX)",
)
def llm_prompt_analysis(context: AssetExecutionContext) -> MaterializeResult:
    """
    Pre-computed LLM analysis of prompts.

    Extracts:
    - domain: portrait, character, animal, environment, object, fantasy, scifi, fanart
    - art_style: photography, digital art, oil painting, watercolor, sketch, 3d render, anime, pixel art, concept art, pop art
    - complexity_score: 1-5

    Run manually: uv run python -m src.enrichment.precompute_llm
    """
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Check prompt_enrichments table (separate from dbt-managed tables)
    total = conn.execute("SELECT COUNT(*) FROM dim_prompts").fetchone()[0]

    # Check if prompt_enrichments table exists
    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_name = 'prompt_enrichments'").fetchall()
    has_enrichment_table = len(tables) > 0

    enriched = 0
    coverage = 0.0
    domain_dist = {}
    style_dist = {}

    if has_enrichment_table:
        # Check enrichment status from prompt_enrichments
        enriched = conn.execute("""
            SELECT COUNT(*) FROM prompt_enrichments WHERE llm_domain IS NOT NULL
        """).fetchone()[0] or 0
        coverage = enriched / total * 100 if total > 0 else 0

        if enriched > 0:
            domains = conn.execute("""
                SELECT llm_domain, COUNT(*) as cnt
                FROM prompt_enrichments
                WHERE llm_domain IS NOT NULL
                GROUP BY llm_domain
                ORDER BY cnt DESC
            """).fetchall()
            domain_dist = {d[0]: d[1] for d in domains}

            styles = conn.execute("""
                SELECT llm_art_style, COUNT(*) as cnt
                FROM prompt_enrichments
                WHERE llm_art_style IS NOT NULL
                GROUP BY llm_art_style
                ORDER BY cnt DESC
            """).fetchall()
            style_dist = {s[0]: s[1] for s in styles}

    conn.close()

    context.log.info(f"LLM analysis: {enriched:,}/{total:,} ({coverage:.1f}%)")

    return MaterializeResult(
        metadata={
            "total_prompts": total,
            "enriched_count": enriched,
            "coverage_pct": f"{coverage:.1f}%",
            "model": "Qwen2.5-1.5B-Instruct-4bit (MLX)",
            "device": "Apple Silicon GPU",
            "domain_distribution": MetadataValue.json(domain_dist),
            "style_distribution": MetadataValue.json(style_dist),
            "run_command": "uv run python -m src.enrichment.precompute_llm",
        }
    )


@asset(
    group_name="ml_enrichment",
    deps=[raw_prompts],
    description="Images from DiffusionDB stored in blob storage (simulated S3)",
)
def prompt_images(context: AssetExecutionContext) -> MaterializeResult:
    """
    Pre-downloaded images from DiffusionDB for Session Explorer.

    Storage: data/blob/images/generations/gen_XXXXX.png
    Pattern: Simulates S3 blob storage for production migration.

    Run manually: uv run python -m src.ingestion.download_images --limit 10000
    """
    blob_path = PROJECT_ROOT / "data" / "blob" / "images" / "generations"

    # Count downloaded images
    if blob_path.exists():
        image_files = list(blob_path.glob("*.png"))
        image_count = len(image_files)
    else:
        image_count = 0

    # Check database for linked images
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        stats = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN image_path IS NOT NULL THEN 1 ELSE 0 END) as linked
            FROM dim_prompts
        """).fetchone()
        total_prompts, linked_count = stats
    except:
        total_prompts, linked_count = 0, 0
    conn.close()

    coverage = linked_count / total_prompts * 100 if total_prompts > 0 else 0

    context.log.info(f"Images: {image_count} files, {linked_count}/{total_prompts} linked ({coverage:.1f}%)")

    return MaterializeResult(
        metadata={
            "image_files_on_disk": image_count,
            "linked_in_database": linked_count,
            "total_prompts": total_prompts,
            "coverage_pct": f"{coverage:.1f}%",
            "storage_path": str(blob_path),
            "storage_pattern": "data/blob/images/generations/gen_XXXXX.png",
            "production_equivalent": "s3://bucket/images/generations/",
            "run_command": "uv run python -m src.ingestion.download_images --limit 10000",
        }
    )
