"""
Link Images to Prompts - Connect downloaded images to raw_prompt_enrichments table.

This script scans the blob storage directory and links existing images
to their corresponding prompts in raw_prompt_enrichments (not dim_prompts,
so links survive dbt runs).

Usage:
    uv run python -m src.ingestion.link_images
"""

import duckdb
from pathlib import Path
from tqdm import tqdm
import argparse
import re


# Storage paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"
BLOB_PATH = PROJECT_ROOT / "data" / "blob" / "images" / "generations"


def ensure_enrichment_table(conn):
    """Ensure raw_prompt_enrichments table exists with image_path column."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS raw_prompt_enrichments (
            prompt_id BIGINT PRIMARY KEY,
            llm_domain VARCHAR,
            llm_art_style VARCHAR,
            llm_complexity_score INTEGER,
            text_embedding FLOAT[384],
            image_path VARCHAR,
            llm_enriched_at TIMESTAMP,
            embedding_enriched_at TIMESTAMP
        )
    ''')

    # Add image_path if missing
    cols = conn.execute('DESCRIBE raw_prompt_enrichments').fetchdf()
    if 'image_path' not in cols['column_name'].values:
        conn.execute('ALTER TABLE raw_prompt_enrichments ADD COLUMN image_path VARCHAR')


def link_images():
    """
    Link downloaded images to prompts in raw_prompt_enrichments.

    Images are named gen_00000.png to gen_09999.png (sequential).
    Prompts are linked by their order in the database (sorted by prompt_id).
    """
    print("=" * 60)
    print("Link Images to Prompts")
    print("=" * 60)

    if not BLOB_PATH.exists():
        print(f"No blob directory found at: {BLOB_PATH}")
        print("Run download_images.py first to download images.")
        return 0

    # Find all image files (sorted by number)
    image_files = sorted(BLOB_PATH.glob("gen_*.png"),
                        key=lambda x: int(re.search(r'(\d+)', x.name).group(1)))
    print(f"\nFound {len(image_files)} images in blob storage")

    if len(image_files) == 0:
        print("No images to link.")
        return 0

    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    # Load VSS extension (required for tables with HNSW indexes)
    conn.execute("INSTALL vss; LOAD vss;")

    ensure_enrichment_table(conn)

    # Get prompt_ids in order (matches image download order)
    prompt_ids = [
        row[0] for row in
        conn.execute("SELECT prompt_id FROM dim_prompts ORDER BY prompt_id").fetchall()
    ]
    print(f"Found {len(prompt_ids)} prompts in database")

    # Link images by order (image 0 -> first prompt, image 1 -> second prompt, etc.)
    linked = 0
    max_links = min(len(image_files), len(prompt_ids))

    for idx in tqdm(range(max_links), desc="Linking images"):
        image_file = image_files[idx]
        prompt_id = prompt_ids[idx]

        # Relative path for portability
        relative_path = f"data/blob/images/generations/{image_file.name}"

        # Upsert to raw_prompt_enrichments
        conn.execute("""
            INSERT INTO raw_prompt_enrichments (prompt_id, image_path)
            VALUES (?, ?)
            ON CONFLICT (prompt_id) DO UPDATE SET
                image_path = EXCLUDED.image_path
        """, [prompt_id, relative_path])

        linked += 1

    conn.close()

    print(f"\nLinking complete:")
    print(f"  Linked: {linked}")

    return linked


def check_status():
    """Show current image linking status."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    stats = conn.execute("""
        SELECT
            COUNT(*) as total_prompts,
            COUNT(e.image_path) as linked_images
        FROM dim_prompts p
        LEFT JOIN raw_prompt_enrichments e ON p.prompt_id = e.prompt_id
    """).fetchone()

    print(f"Total prompts: {stats[0]}")
    print(f"Linked images: {stats[1]}")
    print(f"Coverage: {100*stats[1]/stats[0]:.1f}%")

    # Check blob storage
    if BLOB_PATH.exists():
        image_count = len(list(BLOB_PATH.glob("gen_*.png")))
        print(f"Images in blob: {image_count}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link images to prompts")
    parser.add_argument("--status", action="store_true", help="Show linking status")
    args = parser.parse_args()

    if args.status:
        check_status()
    else:
        link_images()
