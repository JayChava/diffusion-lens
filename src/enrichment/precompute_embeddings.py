"""
Semantic Search Embeddings - Pre-compute text embeddings using sentence-transformers.

Generates 384-dim embeddings for each prompt text, stored in raw_prompt_enrichments
(separate from dbt-managed tables) for fast vector similarity search with DuckDB VSS.

Uses HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor
search - O(log n) lookup instead of O(n) brute force.

Usage:
    uv run python -m src.enrichment.precompute_embeddings
    uv run python -m src.enrichment.precompute_embeddings --limit 100  # Test with 100 prompts
"""

import argparse
import time
from pathlib import Path

import duckdb
from sentence_transformers import SentenceTransformer

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"


def ensure_enrichment_table(con: duckdb.DuckDBPyConnection):
    """Ensure raw_prompt_enrichments table exists (separate from dbt-managed tables)."""
    con.execute('''
        CREATE TABLE IF NOT EXISTS raw_prompt_enrichments (
            prompt_id BIGINT PRIMARY KEY,
            -- LLM analysis (Qwen2.5 via MLX)
            llm_domain VARCHAR,
            llm_art_style VARCHAR,
            llm_complexity_score INTEGER,
            -- Text embeddings (all-MiniLM-L6-v2)
            text_embedding FLOAT[384],
            -- Metadata
            llm_enriched_at TIMESTAMP,
            embedding_enriched_at TIMESTAMP
        )
    ''')
    print("  raw_prompt_enrichments table ready")


def create_vss_index(con: duckdb.DuckDBPyConnection):
    """Create HNSW index for fast vector search."""
    try:
        con.execute("INSTALL vss")
        con.execute("LOAD vss")

        # Enable experimental persistence for HNSW indexes on disk
        con.execute("SET hnsw_enable_experimental_persistence = true")

        # Check if index exists
        indexes = con.execute("SELECT * FROM duckdb_indexes() WHERE index_name = 'enrichment_embedding_idx'").fetchdf()

        if len(indexes) == 0:
            print("  Creating HNSW index on raw_prompt_enrichments...")
            con.execute("""
                CREATE INDEX enrichment_embedding_idx ON raw_prompt_enrichments
                USING HNSW (text_embedding)
                WITH (metric = 'cosine')
            """)
            print("  HNSW index created!")
        else:
            print("  HNSW index already exists")
    except Exception as e:
        print(f"  Warning: Could not create VSS index: {e}")
        print("  Vector search will still work (brute force ~10ms for 10K vectors)")


def run_embeddings(limit: int = None, batch_size: int = 100):
    """Generate embeddings for all prompts."""
    print("=" * 60)
    print("Semantic Search Embeddings - sentence-transformers")
    print("=" * 60)

    # Load model
    print("\n[1/5] Loading embedding model...")
    start = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  Model loaded in {time.time() - start:.1f}s")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Connect to database
    print("\n[2/5] Connecting to DuckDB...")
    con = duckdb.connect(str(DB_PATH))

    # Ensure enrichment table exists
    print("\n[3/5] Ensuring enrichment table exists...")
    ensure_enrichment_table(con)

    # Get prompts that need embeddings (not yet in raw_prompt_enrichments or missing embedding)
    print("\n[4/5] Generating embeddings...")

    query = """
        SELECT p.prompt_id, p.prompt_text
        FROM dim_prompts p
        LEFT JOIN raw_prompt_enrichments e ON p.prompt_id = e.prompt_id
        WHERE e.text_embedding IS NULL
    """
    if limit:
        query += f" LIMIT {limit}"

    prompts_df = con.execute(query).fetchdf()
    total = len(prompts_df)

    if total == 0:
        print("  All prompts already have embeddings!")
    else:
        print(f"  Found {total} prompts to process")

        # Process in batches
        start_time = time.time()

        for i in range(0, total, batch_size):
            batch_df = prompts_df.iloc[i:i + batch_size]
            batch_texts = batch_df['prompt_text'].tolist()
            batch_ids = batch_df['prompt_id'].tolist()

            # Generate embeddings for batch
            embeddings = model.encode(batch_texts, show_progress_bar=False)

            # Batch upsert to raw_prompt_enrichments
            data = [(int(pid), emb.tolist()) for pid, emb in zip(batch_ids, embeddings)]
            con.executemany("""
                INSERT INTO raw_prompt_enrichments (prompt_id, text_embedding, embedding_enriched_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (prompt_id) DO UPDATE SET
                    text_embedding = EXCLUDED.text_embedding,
                    embedding_enriched_at = EXCLUDED.embedding_enriched_at
            """, data)

            # Progress
            processed = min(i + batch_size, total)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  Progress: {processed}/{total} ({100*processed/total:.1f}%) | "
                  f"Rate: {rate:.0f}/s | ETA: {eta:.0f}s")

        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.1f}s ({total/elapsed:.0f} prompts/sec)")

    # Create VSS index
    print("\n[5/5] Setting up DuckDB VSS index...")
    create_vss_index(con)

    # Show sample
    print("\n[Sample embeddings]")
    sample = con.execute("""
        SELECT e.prompt_id, p.prompt_text,
               e.text_embedding[1:5] as embedding_preview
        FROM raw_prompt_enrichments e
        JOIN dim_prompts p ON e.prompt_id = p.prompt_id
        WHERE e.text_embedding IS NOT NULL
        LIMIT 3
    """).fetchdf()
    print(sample.to_string())

    # Test vector search
    print("\n[Test vector search]")
    test_query = "fantasy landscape with mountains"
    test_embedding = model.encode([test_query])[0].tolist()

    results = con.execute("""
        SELECT
            p.prompt_text,
            array_cosine_similarity(e.text_embedding, ?::FLOAT[384]) AS similarity
        FROM raw_prompt_enrichments e
        JOIN dim_prompts p ON e.prompt_id = p.prompt_id
        WHERE e.text_embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT 5
    """, [test_embedding]).fetchdf()

    print(f"  Query: '{test_query}'")
    print(f"  Top matches:")
    for _, row in results.iterrows():
        print(f"    {row['similarity']:.3f} | {row['prompt_text'][:60]}...")

    con.close()
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute text embeddings")
    parser.add_argument("--limit", type=int, help="Limit number of prompts to process")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for encoding")
    args = parser.parse_args()

    run_embeddings(limit=args.limit, batch_size=args.batch_size)
