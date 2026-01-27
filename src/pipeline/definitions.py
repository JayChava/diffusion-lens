"""
Dagster Definitions - Entry point for the GenAI Session Analyzer pipeline.

Run with:
    dagster dev -m src.pipeline.definitions
"""

from dagster import Definitions, load_assets_from_modules

from src.pipeline import assets


defs = Definitions(
    assets=load_assets_from_modules([assets]),
)
