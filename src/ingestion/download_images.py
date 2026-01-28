"""
DiffusionDB Image Downloader - Downloads images to local blob storage.

Downloads images from HuggingFace DiffusionDB dataset (2m_first_10k subset)
and stores them locally in blob storage pattern (data/blob/images/generations/).

Images are downloaded in order - image[i] corresponds to prompt[i] since both
come from the same dataset subset. Use link_images.py to connect them to the DB.

Usage:
    uv run python -m src.ingestion.download_images [--limit 10000]
"""

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import argparse


# Storage path
PROJECT_ROOT = Path(__file__).parent.parent.parent
BLOB_PATH = PROJECT_ROOT / "data" / "blob" / "images" / "generations"


def download_images(limit: int = 10_000) -> int:
    """
    Download images from DiffusionDB and save to local blob storage.

    Images are downloaded in order - image[i] matches prompt[i] since
    both come from the same 2m_first_10k subset.

    Args:
        limit: Max images to download (default 10K)

    Returns:
        Number of images downloaded
    """
    print(f"Setting up blob storage at: {BLOB_PATH}")
    BLOB_PATH.mkdir(parents=True, exist_ok=True)

    # Load dataset (same subset used for prompts)
    print(f"Loading DiffusionDB dataset (first {limit} images)...")
    dataset = load_dataset(
        "poloclub/diffusiondb",
        "2m_first_10k",
        split="train",
        trust_remote_code=True
    )

    downloaded = 0
    skipped = 0

    for i, item in enumerate(tqdm(dataset, total=min(limit, len(dataset)), desc="Downloading")):
        if i >= limit:
            break

        image_path = BLOB_PATH / f"gen_{i:05d}.png"

        # Save image (skip if exists)
        if not image_path.exists():
            item['image'].save(str(image_path))
            downloaded += 1
        else:
            skipped += 1

    print(f"\nDownload complete:")
    print(f"  New images: {downloaded}")
    print(f"  Skipped (exist): {skipped}")
    print(f"  Saved to: {BLOB_PATH}")

    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DiffusionDB images")
    parser.add_argument("--limit", type=int, default=10_000, help="Max images to download")
    args = parser.parse_args()

    download_images(limit=args.limit)
