"""
Session Explorer Sample - Demo semantic search over prompts.

Uses sentence-transformers embeddings + DuckDB vector similarity.

Usage:
    uv run python -m src.explorer.sample
    uv run python -m src.explorer.sample "cyberpunk city"
    uv run python -m src.explorer.sample "cute cat" --limit 10
"""

import argparse
import time
from pathlib import Path

import duckdb
from sentence_transformers import SentenceTransformer

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"

# Example searches
EXAMPLES = [
    "fantasy landscape with mountains",
    "cyberpunk city at night",
    "portrait of a woman",
    "cute animal",
]


def search(query: str, model: SentenceTransformer, con, limit: int = 5):
    """Search for similar prompts using vector similarity."""
    print(f"\n{'='*60}")
    print(f"Search: {query}")
    print('='*60)

    # Embed query
    start = time.time()
    query_embedding = model.encode([query])[0].tolist()
    embed_time = time.time() - start

    # Search with cosine similarity
    start = time.time()
    results = con.execute("""
        SELECT
            p.prompt_id,
            p.prompt_text,
            g.status,
            g.latency_ms,
            u.user_tier,
            array_cosine_similarity(e.text_embedding, ?::FLOAT[384]) AS similarity
        FROM raw_prompt_enrichments e
        JOIN dim_prompts p ON e.prompt_id = p.prompt_id
        JOIN fct_generations g ON p.prompt_id = g.prompt_id
        JOIN dim_users u ON g.user_id = u.user_id
        WHERE e.text_embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT ?
    """, [query_embedding, limit]).fetchdf()
    search_time = time.time() - start

    print(f"\nTiming: embed={embed_time*1000:.0f}ms, search={search_time*1000:.0f}ms")
    print(f"\nTop {limit} results:\n")

    for _, row in results.iterrows():
        status_emoji = {'success': '✅', 'timeout': '⏱️', 'safety_violation': '🛑'}.get(row['status'], '❓')
        print(f"  {row['similarity']:.3f} | {status_emoji} {row['latency_ms']:>5}ms | {row['prompt_text'][:50]}...")

    return results


def main():
    parser = argparse.ArgumentParser(description="Session Explorer Sample")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    # Load embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Connect to database
    con = duckdb.connect(str(DB_PATH), read_only=True)

    if args.query:
        # Single search
        search(args.query, model, con, args.limit)
    else:
        # Run examples
        for query in EXAMPLES:
            search(query, model, con, args.limit)

    con.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
