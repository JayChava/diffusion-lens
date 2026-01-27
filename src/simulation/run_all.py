"""
Run the complete simulation pipeline.

Usage:
    uv run python -m src.simulation.run_all                    # Defaults: 10K prompts, 500 users
    uv run python -m src.simulation.run_all --prompts 1000     # Quick test
    uv run python -m src.simulation.run_all --prompts 100000   # Full scale
"""

import argparse
from src.ingestion.diffusiondb_loader import load_diffusiondb_prompts
from src.simulation.user_generator import generate_users
from src.simulation.telemetry_enricher import enrich_generations
from src.simulation.session_builder import build_sessions


def run_pipeline(num_prompts: int = 10_000, num_users: int = 500, year: int = 2025, month: int = 12):
    """Run the full ingestion and simulation pipeline."""

    print("=" * 60)
    print("GenAI Session Analyzer - Data Pipeline")
    print("=" * 60)

    # Step 1: Load prompts from DiffusionDB
    print("\n[1/4] Loading prompts from DiffusionDB...")
    load_diffusiondb_prompts(num_rows=num_prompts)

    # Step 2: Generate synthetic users
    print("\n[2/4] Generating synthetic users...")
    generate_users(num_users)

    # Step 3: Enrich with telemetry
    print("\n[3/4] Enriching with telemetry (latency, status, feedback, downloads)...")
    enrich_generations(year=year, month=month)

    # Step 4: Build sessions
    print("\n[4/4] Building sessions...")
    build_sessions()

    print("\n" + "=" * 60)
    print("Pipeline complete! Data available in data/warehouse.duckdb")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GenAI session simulation pipeline")
    parser.add_argument("--prompts", type=int, default=10_000, help="Number of prompts to load")
    parser.add_argument("--users", type=int, default=500, help="Number of users to generate")
    parser.add_argument("--year", type=int, default=2025, help="Year for activity")
    parser.add_argument("--month", type=int, default=12, help="Month for activity (1-12)")

    args = parser.parse_args()
    run_pipeline(num_prompts=args.prompts, num_users=args.users, year=args.year, month=args.month)
