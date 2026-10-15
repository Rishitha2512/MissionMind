"""
MissionMind — Seasonal-Baseline Z-Score Anomaly Detection.

Method:
  1. Bin MEDA data by Local True Solar Time (1-hour bins) and season (Ls quadrant).
  2. Compute per-bin mean and std for each variable (t_air, pressure).
  3. For a new observation, compute z-score = |value - bin_mean| / bin_std.
  4. Flag as anomalous if z-score > threshold (default 2.5).

All numbers are computed here; the LLM never assigns scores.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class BaselineStats:
    """Per-bin (ltst_bin, season_bin) statistics for a variable."""
    mean: float
    std: float
    count: int


@dataclass
class AnomalyReport:
    detected: bool
    score: float                # max z-score across variables
    method: str = "seasonal_zscore"
    reasons: List[str] = field(default_factory=list)
    variable_scores: Dict[str, float] = field(default_factory=dict)
    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)
    season_name: str = "Northern Spring"
    time_of_sol: str = "Midday"
    severity: str = "Normal"


class SeasonalZScoreDetector:
    """Seasonal-baseline z-score anomaly detector for MEDA data."""

    VARIABLES = ["t_air", "pressure"]
    SEASON_BOUNDARIES = [0, 90, 180, 270, 360]  # Ls boundaries

    def __init__(
        self,
        time_bin_hours: int = 1,
        threshold: float = 2.5,
        min_bin_count: int = 5,
    ):
        self.time_bin_hours = time_bin_hours
        self.threshold = threshold
        self.min_bin_count = min_bin_count
        # baselines[variable][(ltst_bin, season_bin)] = BaselineStats
        self.baselines: Dict[str, Dict[Tuple[int, int], BaselineStats]] = {}
        self._fitted = False

    @staticmethod
    def ls_to_season(ls: float) -> int:
        """Convert solar longitude (0-360) to season bin (0-3)."""
        if ls < 90:
            return 0   # Northern spring
        elif ls < 180:
            return 1   # Northern summer
        elif ls < 270:
            return 2   # Northern fall
        else:
            return 3   # Northern winter

    def ltst_to_bin(self, ltst: float) -> int:
        """Convert LTST (0-24 hours) to time-of-day bin."""
        return int(ltst // self.time_bin_hours) % (24 // self.time_bin_hours)

    def fit(self, df: pd.DataFrame) -> "SeasonalZScoreDetector":
        """
        Fit baselines from MEDA data.

        Expected columns: ltst (float, 0-24), ls (float, 0-360),
                          t_air (°C), pressure (Pa)
        """
        df = df.copy()
        df["ltst_bin"] = df["ltst"].apply(self.ltst_to_bin)
        df["season_bin"] = df["ls"].apply(self.ls_to_season)

        for var in self.VARIABLES:
            if var not in df.columns:
                continue
            self.baselines[var] = {}
            grouped = df.groupby(["ltst_bin", "season_bin"])[var]
            for (ltst_bin, season_bin), group in grouped:
                if len(group) >= self.min_bin_count:
                    self.baselines[var][(ltst_bin, season_bin)] = BaselineStats(
                        mean=float(group.mean()),
                        std=float(group.std()) if group.std() > 0 else 1.0,
                        count=len(group),
                    )

        self._fitted = True
        return self

    def score_observation(
        self,
        ltst: float,
        ls: float,
        values: Dict[str, float],
    ) -> AnomalyReport:
        """
        Score a single observation against the seasonal baseline.

        Args:
            ltst: Local True Solar Time (0-24)
            ls: Solar longitude (0-360)
            values: dict of variable_name → measured value
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        ltst_bin = self.ltst_to_bin(ltst)
        season_bin = self.ls_to_season(ls)
        season_names = ["Northern Spring", "Northern Summer",
                        "Northern Fall", "Northern Winter"]

        variable_scores: Dict[str, float] = {}
        baseline_details: Dict[str, Dict[str, float]] = {}
        reasons: List[str] = []

        for var, val in values.items():
            if var not in self.baselines:
                continue
            key = (ltst_bin, season_bin)
            if key not in self.baselines[var]:
                reasons.append(
                    f"No baseline for {var} at LTST bin {ltst_bin}, "
                    f"{season_names[season_bin]} — skipping."
                )
                continue

            stats = self.baselines[var][key]
            z = float(abs(val - stats.mean) / stats.std)
            variable_scores[var] = float(round(z, 3))
            baseline_details[var] = {
                "mean": float(round(stats.mean, 2)),
                "std": float(round(stats.std, 2)),
                "measured": float(round(val, 2)),
                "z_score": float(round(z, 2)),
                "delta": float(round(val - stats.mean, 2)),
                "unit": "°C" if var == "t_air" else "Pa",
            }

            if z > self.threshold:
                direction = "above" if val > stats.mean else "below"
                reasons.append(
                    f"{var} = {val:.2f} is {z:.2f}σ {direction} the "
                    f"seasonal baseline (mean={stats.mean:.2f}, "
                    f"std={stats.std:.2f}) for LTST {ltst_bin:02d}:00–"
                    f"{ltst_bin + self.time_bin_hours:02d}:00, "
                    f"{season_names[season_bin]}."
                )

        max_z = max(variable_scores.values()) if variable_scores else 0.0
        detected = max_z > self.threshold

        if not detected and not reasons:
            reasons.append("All variables within seasonal norms.")

        # Compute human-friendly diurnal time description
        if 0 <= ltst < 6:
            time_of_sol = "Pre-dawn (00:00–06:00 LTST)"
        elif 6 <= ltst < 12:
            time_of_sol = "Morning (06:00–12:00 LTST)"
        elif 12 <= ltst < 18:
            time_of_sol = "Afternoon (12:00–18:00 LTST)"
        else:
            time_of_sol = "Night (18:00–24:00 LTST)"

        # Compute severity
        if max_z > 3.5:
            severity = "Critical Anomaly (>3.5σ)"
        elif max_z > self.threshold:
            severity = "Moderate Anomaly (2.5σ–3.5σ)"
        else:
            severity = "Normal Mars Climatology (<2.5σ)"

        return AnomalyReport(
            detected=detected,
            score=round(max_z, 3),
            method="seasonal_zscore",
            reasons=reasons,
            variable_scores=variable_scores,
            baselines=baseline_details,
            season_name=season_names[season_bin],
            time_of_sol=time_of_sol,
            severity=severity,
        )

    def score_series(
        self,
        series: List[Dict],
        ls: float,
    ) -> AnomalyReport:
        """
        Score a time series. Returns the observation with the highest z-score.
        Each entry: {"t": "sol_ltst_string", "value": float}
        We assume the value is t_air by default.
        """
        if not series:
            return AnomalyReport(
                detected=False, score=0.0,
                reasons=["Empty series provided."]
            )

        best_report: Optional[AnomalyReport] = None

        for point in series:
            # Parse LTST from the 't' field — format: "sol_XXX_LTST_HH.HH"
            ltst = self._parse_ltst(point["t"])
            if ltst is None:
                ltst = 12.0  # default to noon if unparseable

            report = self.score_observation(
                ltst=ltst, ls=ls,
                values={"t_air": point["value"]},
            )
            if best_report is None or report.score > best_report.score:
                best_report = report

        return best_report or AnomalyReport(
            detected=False, score=0.0, reasons=["No scoreable points."]
        )

    @staticmethod
    def _parse_ltst(t_str: str) -> Optional[float]:
        """Try to extract LTST from various string formats."""
        import re
        # Try "LTST_HH.HH" pattern
        m = re.search(r"LTST[_\s]*([\d.]+)", t_str, re.IGNORECASE)
        if m:
            return float(m.group(1))
        # Try plain float
        try:
            v = float(t_str)
            if 0 <= v <= 24:
                return v
        except ValueError:
            pass
        return None

    def export_baselines(self, path: Path) -> None:
        """Export baselines to JSON for inspection."""
        out = {}
        for var, bins in self.baselines.items():
            out[var] = {
                f"{k[0]}_{k[1]}": {
                    "mean": v.mean, "std": v.std, "count": v.count
                }
                for k, v in bins.items()
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
