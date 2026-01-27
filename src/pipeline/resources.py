"""
Dagster Resources - Shared resources for the pipeline.
"""

from dagster import ConfigurableResource
import duckdb
from pathlib import Path


class DuckDBResource(ConfigurableResource):
    """DuckDB connection resource."""

    db_path: str = "data/warehouse.duckdb"

    def get_connection(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(self.db_path)
