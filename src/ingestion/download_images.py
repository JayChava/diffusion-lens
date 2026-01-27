"""
DiffusionDB Image Downloader - Downloads images for prompts in our database.

Downloads images from HuggingFace DiffusionDB dataset and stores them locally
in the blob storage pattern (data/blob/images/generations/).

Usage:
    uv run python -m src.ingestion.download_images [--limit 1000]
"""

import duckdb
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import argparse


# Storage paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"
BLOB_PATH = PROJECT_ROOT / "data" / "blob" / "images" / "generations"


def download_images(limit: int = 10_000, batch_size: int = 100) -> int:
    """
    Download images from DiffusionDB and save to local blob storage.

    Strategy:
    1. Get prompt texts from our database
    2. Stream DiffusionDB dataset
    3. Match prompts and save images
    4. Update database with image_path

    Args:
        limit: Max images to download
        batch_size: Batch size for dataset streaming

    Returns:
        Number of images downloaded
    """
    print(f"Setting up blob storage at: {BLOB_PATH}")
    BLOB_PATH.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    # Get prompts we need images for
    prompts_df = conn.execute("""
        SELECT prompt_id, prompt
        FROM raw_prompts
        WHERE prompt_id < ?
        ORDER BY prompt_id
    """, [limit]).fetchdf()

    print(f"Found {len(prompts_df)} prompts to match")

    # Create a lookup dict for fast matching
    prompt_to_id = {row['prompt']: row['prompt_id'] for _, row in prompts_df.iterrows()}

    # Add image_path column if not exists
    try:
        conn.execute("ALTER TABLE raw_prompts ADD COLUMN image_path VARCHAR")
    except:
        pass  # Column already exists

    # Stream DiffusionDB (2m_first_10k subset for speed)
    print("Loading DiffusionDB dataset (2m_first_10k subset)...")
    dataset = load_dataset(
        "poloclub/diffusiondb",
        "2m_first_10k",
        split="train",
        trust_remote_code=True
    )

    downloaded = 0
    matched = 0

    print(f"Scanning dataset for matching prompts...")
    for item in tqdm(dataset, desc="Downloading images"):
        prompt_text = item['prompt']

        if prompt_text in prompt_to_id:
            prompt_id = prompt_to_id[prompt_text]

            # Save image
            image_filename = f"gen_{prompt_id:05d}.png"
            image_path = BLOB_PATH / image_filename

            if not image_path.exists():
                item['image'].save(str(image_path))
                downloaded += 1

            # Update database with relative path
            relative_path = f"data/blob/images/generations/{image_filename}"
            conn.execute("""
                UPDATE raw_prompts
                SET image_path = ?
                WHERE prompt_id = ?
            """, [relative_path, prompt_id])

            matched += 1

            # Remove from lookup (already processed)
            del prompt_to_id[prompt_text]

            if matched >= limit:
                break

    conn.close()

    print(f"\nDownload complete:")
    print(f"  Matched prompts: {matched}")
    print(f"  New images downloaded: {downloaded}")
    print(f"  Images saved to: {BLOB_PATH}")

    return downloaded


def download_images_streaming(limit: int = 10_000) -> int:
    """
    Alternative: Download images by streaming the full dataset.

    This is more reliable but slower - it doesn't require exact prompt matching,
    just takes the first N images.
    """
    print(f"Setting up blob storage at: {BLOB_PATH}")
    BLOB_PATH.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(DB_PATH))

    # Add image_path column if not exists
    try:
        conn.execute("ALTER TABLE raw_prompts ADD COLUMN image_path VARCHAR")
    except:
        pass

    # Stream dataset
    print(f"Loading DiffusionDB dataset (streaming first {limit} images)...")
    dataset = load_dataset(
        "poloclub/diffusiondb",
        "2m_first_10k",
        split="train",
        trust_remote_code=True
    )

    downloaded = 0

    for i, item in enumerate(tqdm(dataset, total=min(limit, len(dataset)), desc="Downloading")):
        if i >= limit:
            break

        prompt_id = i
        image_filename = f"gen_{prompt_id:05d}.png"
        image_path = BLOB_PATH / image_filename

        # Save image
        if not image_path.exists():
            item['image'].save(str(image_path))

        # Update database
        relative_path = f"data/blob/images/generations/{image_filename}"
        conn.execute("""
            UPDATE raw_prompts
            SET image_path = ?
            WHERE prompt_id = ?
        """, [relative_path, prompt_id])

        downloaded += 1

    conn.close()

    print(f"\nDownload complete: {downloaded} images")
    print(f"Saved to: {BLOB_PATH}")

    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DiffusionDB images")
    parser.add_argument("--limit", type=int, default=10_000, help="Max images to download")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (simpler, takes first N)")
    args = parser.parse_args()

    if args.streaming:
        download_images_streaming(limit=args.limit)
    else:
        download_images(limit=args.limit)
