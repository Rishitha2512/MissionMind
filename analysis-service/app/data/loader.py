"""
MissionMind — MEDA Data Loader.

Loads processed MEDA data (parquet) and provides convenient access methods.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional


class MEDADataLoader:
    """Loads and provides access to processed MEDA data."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self._loaded = False

    def load(self) -> pd.DataFrame:
        """Load the processed MEDA parquet file."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"MEDA data not found at {self.data_path}. "
                f"Run 'python scripts/generate_meda_data.py' first."
            )
        self.df = pd.read_parquet(self.data_path)
        self._loaded = True
        return self.df

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.df is not None

    def get_sol_range(self) -> tuple[int, int]:
        """Return (min_sol, max_sol)."""
        if not self.is_loaded:
            raise RuntimeError("Data not loaded.")
        return int(self.df["sol"].min()), int(self.df["sol"].max())

    def get_sol_data(self, sol: int) -> pd.DataFrame:
        """Get all data for a specific sol."""
        if not self.is_loaded:
            raise RuntimeError("Data not loaded.")
        return self.df[self.df["sol"] == sol].copy()

    def get_variable_stats(self, variable: str) -> dict:
        """Get global stats for a variable."""
        if not self.is_loaded:
            raise RuntimeError("Data not loaded.")
        col = self.df[variable]
        return {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "count": int(col.count()),
        }

    def estimate_ls(self, sol: int) -> float:
        """
        Estimate solar longitude (Ls) from sol number.

        Perseverance landed at Ls ≈ 5.2° on Sol 0 (Feb 18, 2021).
        Mars year ≈ 668.6 sols, so Ls increases ~0.539°/sol.
        """
        ls_at_landing = 5.2
        ls_rate = 360.0 / 668.6  # ~0.539°/sol
        ls = (ls_at_landing + sol * ls_rate) % 360.0
        return round(ls, 2)
