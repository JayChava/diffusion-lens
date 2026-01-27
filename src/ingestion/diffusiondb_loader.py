"""
DiffusionDB Loader - Ingests prompt metadata from HuggingFace.

Downloads metadata.parquet directly (no images) and samples N rows.
This avoids the deprecated dataset script issue.
"""

import duckdb
from huggingface_hub import hf_hub_download
from pathlib import Path


def load_diffusiondb_prompts(
    num_rows: int = 10_000,
    db_path: str = "data/warehouse.duckdb",
    seed: int = 42,
) -> int:
    """
    Load DiffusionDB prompts into DuckDB.

    Args:
        num_rows: Number of rows to sample (default 10K)
        db_path: Path to DuckDB file
        seed: Random seed for reproducible sampling

    Returns:
        Number of rows loaded
    """
    print(f"Downloading DiffusionDB metadata.parquet from HuggingFace...")

    # Download metadata parquet (contains all prompt data, no images)
    parquet_path = hf_hub_download(
        repo_id="poloclub/diffusiondb",
        filename="metadata.parquet",
        repo_type="dataset",
    )

    print(f"Downloaded to: {parquet_path}")

    # Connect to DuckDB and sample directly from parquet
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)

    # Check total rows available
    total_rows = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')"
    ).fetchone()[0]
    print(f"Total rows in metadata.parquet: {total_rows:,}")

    # Sample and load into DuckDB
    # Use setseed() for reproducible random sampling
    conn.execute("DROP TABLE IF EXISTS raw_prompts")
    conn.execute(f"SELECT setseed({seed / 1000000})")  # setseed takes 0-1 range
    conn.execute(f"""
        CREATE TABLE raw_prompts AS
        SELECT
            ROW_NUMBER() OVER () - 1 AS prompt_id,
            prompt,
            seed AS generation_seed,
            step,
            cfg,
            sampler,
            width,
            height
        FROM read_parquet('{parquet_path}')
        ORDER BY RANDOM()
        LIMIT {num_rows}
    """)

    row_count = conn.execute("SELECT COUNT(*) FROM raw_prompts").fetchone()[0]
    print(f"Sampled {row_count:,} rows into raw_prompts table")

    # Show sample
    print("\nSample prompts:")
    samples = conn.execute(
        "SELECT prompt_id, LEFT(prompt, 80) as prompt_preview FROM raw_prompts LIMIT 3"
    ).fetchall()
    for pid, preview in samples:
        print(f"  [{pid}] {preview}...")

    conn.close()
    return row_count


def get_sample_prompts(db_path: str = "data/warehouse.duckdb", limit: int = 5) -> list:
    """Quick utility to peek at loaded prompts."""
    conn = duckdb.connect(db_path, read_only=True)
    results = conn.execute(f"SELECT prompt FROM raw_prompts LIMIT {limit}").fetchall()
    conn.close()
    return [r[0] for r in results]


if __name__ == "__main__":
    # CLI usage: python -m src.ingestion.diffusiondb_loader [num_rows]
    # Examples:
    #   python -m src.ingestion.diffusiondb_loader          # 10K rows (default)
    #   python -m src.ingestion.diffusiondb_loader 1000     # 1K rows (fast test)
    #   python -m src.ingestion.diffusiondb_loader 100000   # 100K rows (full demo)
    import sys
    num_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    load_diffusiondb_prompts(num_rows=num_rows)
