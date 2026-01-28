"""
LLM Prompt Analysis - Pre-compute enrichment features using Qwen via MLX.

Uses Pydantic to enforce valid domain and art_style values with automatic normalization.

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
import time
from pathlib import Path
from typing import Literal

import duckdb
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"

# Valid options
VALID_DOMAINS = ["portrait", "character", "animal", "environment", "object", "fantasy", "scifi", "fanart"]
VALID_STYLES = ["photography", "digital art", "oil painting", "watercolor", "sketch", "3d render", "anime", "pixel art", "concept art", "pop art"]

DOMAINS = Literal["portrait", "character", "animal", "environment", "object", "fantasy", "scifi", "fanart"]
ART_STYLES = Literal["photography", "digital art", "oil painting", "watercolor", "sketch", "3d render", "anime", "pixel art", "concept art", "pop art"]


class PromptAnalysis(BaseModel):
    """Pydantic model with automatic normalization via field validators."""
    domain: DOMAINS
    art_style: ART_STYLES
    complexity_score: int = Field(ge=1, le=5)

    @field_validator('domain', mode='before')
    @classmethod
    def normalize_domain(cls, v):
        if not isinstance(v, str):
            return "character"
        v = v.lower().strip()
        # Map common LLM variations to valid values
        mapping = {
            'landscape': 'environment', 'scene': 'environment', 'interior': 'environment',
            'cityscape': 'environment', 'nature': 'environment', 'abstract': 'object',
            'creature': 'animal', 'pet': 'animal',
        }
        for key, val in mapping.items():
            if key in v:
                return val
        return v if v in VALID_DOMAINS else "character"

    @field_validator('art_style', mode='before')
    @classmethod
    def normalize_art_style(cls, v):
        if not isinstance(v, str):
            return "digital art"
        v = v.lower().strip()
        # Map common LLM variations to valid values
        mapping = {
            'photorealistic': 'photography', 'realistic': 'photography', 'photo': 'photography',
            'film': 'photography', 'oil': 'oil painting', 'digital': 'digital art',
            '3d': '3d render', 'cgi': '3d render', 'render': '3d render',
            'pencil': 'sketch', 'drawing': 'sketch', 'line art': 'sketch',
            'cg': 'digital art', 'illustration': 'digital art', 'manga': 'anime',
            'retro': 'pixel art', '8-bit': 'pixel art', 'warhol': 'pop art', 'graphic': 'pop art',
        }
        for key, val in mapping.items():
            if key in v:
                return val
        return v if v in VALID_STYLES else "digital art"

    @field_validator('complexity_score', mode='before')
    @classmethod
    def normalize_complexity(cls, v):
        try:
            v = int(v)
            return max(1, min(5, v))  # Clamp to 1-5
        except (ValueError, TypeError):
            return 3


def parse_llm_response(response: str) -> PromptAnalysis:
    """Extract JSON from LLM response and validate with Pydantic."""
    start = response.find('{')
    end = response.rfind('}') + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON found")
    return PromptAnalysis(**json.loads(response[start:end]))


# LLM prompt template
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


def ensure_enrichment_table(con: duckdb.DuckDBPyConnection):
    """Ensure raw_prompt_enrichments table exists."""
    con.execute('''
        CREATE TABLE IF NOT EXISTS raw_prompt_enrichments (
            prompt_id BIGINT PRIMARY KEY,
            llm_domain VARCHAR,
            llm_art_style VARCHAR,
            llm_complexity_score INTEGER,
            text_embedding FLOAT[384],
            llm_enriched_at TIMESTAMP,
            embedding_enriched_at TIMESTAMP
        )
    ''')


def run_enrichment(limit: int = None, reset: bool = False):
    """Run LLM enrichment on all prompts."""
    print("=" * 70)
    print("LLM Prompt Analysis - Qwen2.5 via MLX (Pydantic validation)")
    print("=" * 70)

    # Load model
    print(f"\n[1/4] Loading model... (MLX {mx.__version__})")
    model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    sampler = make_sampler(temp=0.1)
    print("  Model ready!")

    # Connect to database
    print("\n[2/4] Connecting to DuckDB...")
    con = duckdb.connect(str(DB_PATH))
    con.execute("INSTALL vss; LOAD vss;")
    ensure_enrichment_table(con)

    # Reset if requested
    if reset:
        print("  Resetting existing LLM enrichment data...")
        con.execute("UPDATE raw_prompt_enrichments SET llm_domain = NULL, llm_art_style = NULL, llm_complexity_score = NULL, llm_enriched_at = NULL")

    # Get prompts needing enrichment
    print("\n[3/4] Finding prompts to process...")
    query = """
        SELECT p.prompt_id, p.prompt_text
        FROM dim_prompts p
        LEFT JOIN raw_prompt_enrichments e ON p.prompt_id = e.prompt_id
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
    start_time = time.time()
    success = errors = 0

    pbar = tqdm(prompts_df.iterrows(), total=total, desc="Enriching", unit="prompt")

    for _, row in pbar:
        prompt_id = row['prompt_id']
        llm_prompt = EXTRACTION_PROMPT.format(prompt=row['prompt_text'][:300])

        try:
            response = generate(model, tokenizer, prompt=llm_prompt, max_tokens=60, sampler=sampler)
            analysis = parse_llm_response(response)
            success += 1
        except (json.JSONDecodeError, ValidationError, ValueError):
            analysis = PromptAnalysis(domain="character", art_style="digital art", complexity_score=3)
            errors += 1

        con.execute("""
            INSERT INTO raw_prompt_enrichments (prompt_id, llm_domain, llm_art_style, llm_complexity_score, llm_enriched_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (prompt_id) DO UPDATE SET
                llm_domain = EXCLUDED.llm_domain, llm_art_style = EXCLUDED.llm_art_style,
                llm_complexity_score = EXCLUDED.llm_complexity_score, llm_enriched_at = EXCLUDED.llm_enriched_at
        """, [prompt_id, analysis.domain, analysis.art_style, analysis.complexity_score])

        pbar.set_postfix({"ok": success, "err": errors})

    pbar.close()

    # Summary
    elapsed = time.time() - start_time
    print(f"\nDone! {success} success, {errors} defaulted | {elapsed:.1f}s ({total/elapsed:.1f}/s)")

    # Show distributions
    for col, label in [("llm_domain", "Domain"), ("llm_art_style", "Art style")]:
        print(f"\n{label} distribution:")
        dist = con.execute(f"SELECT {col}, COUNT(*) as cnt FROM raw_prompt_enrichments WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY cnt DESC").fetchdf()
        for _, r in dist.iterrows():
            print(f"  {r[col]:15s}: {r['cnt']}")

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit prompts to process")
    parser.add_argument("--reset", action="store_true", help="Reset and re-process")
    args = parser.parse_args()
    run_enrichment(limit=args.limit, reset=args.reset)
