"""
LLM Prompt Analysis - Pre-compute enrichment features using Qwen via MLX.

Uses Pydantic to enforce valid domain and art_style values.

Extracts from each prompt:
- domain: portrait, character, animal, environment, object, fantasy, scifi, fanart
- art_style: photography, digital art, oil painting, watercolor, sketch, 3d render, anime, pixel art, concept art, pop art
- complexity_score: 1-5 based on prompt detail level

Usage:
    uv run python -m src.enrichment.precompute_llm
    uv run python -m src.enrichment.precompute_llm --limit 100
    uv run python -m src.enrichment.precompute_llm --reset --limit 20  # Re-process with new prompt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

import duckdb
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"

# Predefined options (enforced by Pydantic)
DOMAINS = Literal[
    "portrait",    # faces, headshots, close-up of people/creatures
    "character",   # full body people, figures, creatures with personality
    "animal",      # pets, wildlife, creatures (non-humanoid focus)
    "environment", # landscapes, cityscapes, interiors, architecture, scenes
    "object",      # products, items, vehicles, food, still life
    "fantasy",     # mythical, magical, dragons, fairies, supernatural
    "scifi",       # cyberpunk, futuristic, space, robots, tech
    "fanart",      # celebrities, known characters, movie/game/anime IP
]

ART_STYLES = Literal[
    "photography",    # realistic photos, film stills, CCTV, candid
    "digital art",    # general digital illustration, artstation style
    "oil painting",   # classical painting, brush strokes, canvas texture
    "watercolor",     # soft edges, color bleeding, paper texture
    "sketch",         # pencil, line art, drawings, charcoal
    "3d render",      # CGI, Blender, 3D modeling, rendered
    "anime",          # anime/manga style, Japanese animation
    "pixel art",      # retro gaming, 8-bit, pixelated
    "concept art",    # game/film concept art, artstation, detailed designs
    "pop art",        # bold colors, Warhol style, graphic design
]


class PromptAnalysis(BaseModel):
    """Pydantic model enforcing valid values."""
    domain: DOMAINS
    art_style: ART_STYLES
    complexity_score: int = Field(ge=1, le=5)


# LLM prompt template with explicit options and examples
EXTRACTION_PROMPT = '''Classify this image generation prompt. Return ONLY valid JSON.

Prompt: "{prompt}"

DOMAIN options (pick the BEST fit):
- portrait: faces, headshots, close-ups of people → "portrait of a woman", "headshot"
- character: full body people/figures → "warrior standing", "girl in dress"
- animal: pets, wildlife, creatures → "dog laying down", "cat", "eagle"
- environment: landscapes, cities, interiors → "mountain scene", "city street", "bedroom"
- object: products, vehicles, food, items → "sports car", "glass bottle", "pizza"
- fantasy: magical, mythical, supernatural → "dragon", "wizard", "fairy", "magical forest"
- scifi: cyberpunk, futuristic, space, robots → "cyborg", "spaceship", "neon city"
- fanart: celebrities, known characters/IP → "Batman", "Ellie from Last of Us", "Taylor Swift"

ART_STYLE options (pick based on visual style requested):
- photography: realistic photo, film still, CCTV → "photo of", "realistic", "hasselblad"
- digital art: general illustration → "digital painting", "artstation"
- oil painting: classical painting → "oil painting", "canvas", "brush strokes"
- watercolor: soft painted style → "watercolor"
- sketch: line art, pencil → "sketch", "pencil drawing", "line art"
- 3d render: CGI, rendered → "3d render", "blender", "octane"
- anime: Japanese animation style → "anime", "manga style"
- pixel art: retro/8-bit → "pixel art", "8-bit"
- concept art: detailed game/film art → "concept art", "by greg rutkowski"
- pop art: bold graphic style → "pop art", "warhol style"

COMPLEXITY: 1=very simple ("a cat"), 3=moderate detail, 5=very detailed (many modifiers, artists, styles)

Return JSON: {{"domain": "X", "art_style": "Y", "complexity_score": N}}
JSON:'''


def parse_and_validate(response: str) -> PromptAnalysis:
    """Parse LLM response and validate with Pydantic."""
    # Find JSON in response
    start = response.find('{')
    end = response.rfind('}') + 1

    if start < 0 or end <= start:
        raise ValueError("No JSON found in response")

    json_str = response[start:end]
    data = json.loads(json_str)

    # Normalize values before validation
    if 'domain' in data:
        data['domain'] = data['domain'].lower().strip()
        # Map common domain variations
        domain_map = {
            'landscape': 'environment',
            'scene': 'environment',
            'interior': 'environment',
            'cityscape': 'environment',
            'nature': 'environment',
            'abstract': 'object',  # map abstract to object as catch-all
            'creature': 'animal',
            'pet': 'animal',
        }
        for key, val in domain_map.items():
            if key in data['domain']:
                data['domain'] = val
                break
        # Validate domain
        valid_domains = ['portrait', 'character', 'animal', 'environment', 'object', 'fantasy', 'scifi', 'fanart']
        if data['domain'] not in valid_domains:
            data['domain'] = 'character'  # default fallback

    if 'art_style' in data:
        data['art_style'] = data['art_style'].lower().strip()
        # Map common variations
        style_map = {
            'photorealistic': 'photography',
            'realistic': 'photography',
            'photo': 'photography',
            'film': 'photography',
            'oil': 'oil painting',
            'digital': 'digital art',
            '3d': '3d render',
            'cgi': '3d render',
            'render': '3d render',
            'pencil': 'sketch',
            'drawing': 'sketch',
            'line art': 'sketch',
            'cg': 'digital art',
            'illustration': 'digital art',
            'manga': 'anime',
            'retro': 'pixel art',
            '8-bit': 'pixel art',
            'warhol': 'pop art',
            'graphic': 'pop art',
        }
        for key, val in style_map.items():
            if key in data['art_style']:
                data['art_style'] = val
                break
        # If still not valid, default to 'digital art'
        valid_styles = ['photography', 'digital art', 'oil painting', 'watercolor',
                       'sketch', '3d render', 'anime', 'pixel art', 'concept art', 'pop art']
        if data['art_style'] not in valid_styles:
            data['art_style'] = 'digital art'

    # Validate with Pydantic
    return PromptAnalysis(**data)


def get_defaults() -> PromptAnalysis:
    """Return default values when parsing fails."""
    return PromptAnalysis(domain="character", art_style="digital art", complexity_score=3)


def ensure_enrichment_table(con: duckdb.DuckDBPyConnection):
    """Ensure prompt_enrichments table exists (separate from dbt-managed tables)."""
    con.execute('''
        CREATE TABLE IF NOT EXISTS prompt_enrichments (
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
    print("  prompt_enrichments table ready")


def run_enrichment(limit: int = None, reset: bool = False):
    """Run LLM enrichment on all prompts."""
    print("=" * 70)
    print("LLM Prompt Analysis - Qwen2.5 via MLX (with Pydantic validation)")
    print("=" * 70)

    # Show valid options
    print("\nValid domains:", "portrait, character, animal, environment, object, fantasy, scifi, fanart")
    print("Valid styles:", "photography, digital art, oil painting, watercolor, sketch, 3d render, anime, pixel art, concept art, pop art")

    # Load model
    print(f"\n[1/4] Loading model... (MLX {mx.__version__}, Device: {mx.default_device()})")
    model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    sampler = make_sampler(temp=0.1)
    print("  Model ready!")

    # Connect to database
    print("\n[2/4] Connecting to DuckDB...")
    con = duckdb.connect(str(DB_PATH))
    # Load VSS extension (required because table has HNSW index)
    con.execute("INSTALL vss; LOAD vss;")

    # Ensure enrichment table exists
    print("\n[3/4] Ensuring enrichment table exists...")
    ensure_enrichment_table(con)

    # Reset if requested
    if reset:
        print("  Resetting existing LLM enrichment data...")
        con.execute("UPDATE prompt_enrichments SET llm_domain = NULL, llm_art_style = NULL, llm_complexity_score = NULL, llm_enriched_at = NULL")

    # Get prompts that need enrichment (not yet in prompt_enrichments or missing LLM data)
    query = """
        SELECT p.prompt_id, p.prompt_text
        FROM dim_prompts p
        LEFT JOIN prompt_enrichments e ON p.prompt_id = e.prompt_id
        WHERE e.llm_domain IS NULL
    """
    if limit:
        query += f" LIMIT {limit}"

    prompts_df = con.execute(query).fetchdf()
    total = len(prompts_df)

    if total == 0:
        print("  All prompts already enriched!")
        return

    print(f"\n[4/4] Processing {total} prompts...")
    print("-" * 70)

    start_time = time.time()
    success = 0
    errors = 0

    # Use tqdm for progress bar
    pbar = tqdm(
        prompts_df.iterrows(),
        total=total,
        desc="Enriching",
        unit="prompt",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for i, row in pbar:
        prompt_id = row['prompt_id']
        prompt_text = row['prompt_text'][:300]

        llm_prompt = EXTRACTION_PROMPT.format(prompt=prompt_text)

        try:
            response = generate(model, tokenizer, prompt=llm_prompt, max_tokens=60, sampler=sampler)
            analysis = parse_and_validate(response)
            success += 1
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            analysis = get_defaults()
            errors += 1

        # Upsert to prompt_enrichments table
        con.execute("""
            INSERT INTO prompt_enrichments (prompt_id, llm_domain, llm_art_style, llm_complexity_score, llm_enriched_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (prompt_id) DO UPDATE SET
                llm_domain = EXCLUDED.llm_domain,
                llm_art_style = EXCLUDED.llm_art_style,
                llm_complexity_score = EXCLUDED.llm_complexity_score,
                llm_enriched_at = EXCLUDED.llm_enriched_at
        """, [prompt_id, analysis.domain, analysis.art_style, analysis.complexity_score])

        # Update progress bar description
        pbar.set_postfix({
            "domain": analysis.domain[:8],
            "style": analysis.art_style[:10],
            "ok": success,
            "err": errors
        })

    # Close progress bar
    pbar.close()

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "-" * 70)
    print(f"Done! {success} success, {errors} defaulted | {elapsed:.1f}s ({total/elapsed:.1f} prompts/s)")
    print(f"Estimated time for 10K: {10000/total * elapsed / 60:.0f} minutes")

    # Show distribution
    print("\nDomain distribution:")
    dist = con.execute("""
        SELECT llm_domain, COUNT(*) as cnt FROM prompt_enrichments
        WHERE llm_domain IS NOT NULL GROUP BY llm_domain ORDER BY cnt DESC
    """).fetchdf()
    for _, r in dist.iterrows():
        print(f"  {r['llm_domain']:12s}: {r['cnt']}")

    print("\nArt style distribution:")
    dist = con.execute("""
        SELECT llm_art_style, COUNT(*) as cnt FROM prompt_enrichments
        WHERE llm_art_style IS NOT NULL GROUP BY llm_art_style ORDER BY cnt DESC
    """).fetchdf()
    for _, r in dist.iterrows():
        print(f"  {r['llm_art_style']:15s}: {r['cnt']}")

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit prompts to process")
    parser.add_argument("--reset", action="store_true", help="Reset and re-process already enriched prompts")
    args = parser.parse_args()
    run_enrichment(limit=args.limit, reset=args.reset)
