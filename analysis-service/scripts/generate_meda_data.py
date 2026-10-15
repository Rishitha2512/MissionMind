"""
Generate synthetic MEDA-like data for MissionMind.

Produces realistic Mars surface temperature and pressure data modeled on
actual MEDA characteristics at Jezero Crater (~18.4°N).

Key characteristics modeled:
  - Diurnal temperature cycle (coldest ~06:00 LTST, warmest ~14:00 LTST)
  - Air temperature range: approx -80°C to +5°C
  - Pressure range: approx 700-850 Pa with seasonal CO2 cycle
  - Perseverance landed at Ls ≈ 5.2° (early northern spring)
  - Known anomalies injected: dust events, cold fronts, pressure waves

Output: data/processed/meda_clean.parquet
"""
import sys
from pathlib import Path

# Add parent to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# ── Mars constants ───────────────────────────────────────────────────────────
SOLS = range(1, 101)            # 100 sols
HOURS_PER_SOL = 24              # measurements every hour
LS_AT_LANDING = 5.2             # Ls at sol 0
LS_RATE = 360.0 / 668.6         # degrees Ls per sol

# Temperature model parameters (Jezero Crater)
T_AIR_MEAN = -40.0              # global mean air temp (°C)
T_AIR_DIURNAL_AMP = 35.0        # diurnal amplitude
T_AIR_SEASONAL_AMP = 8.0        # seasonal amplitude
T_AIR_NOISE_STD = 3.0           # natural variability

# Pressure model parameters
P_MEAN = 750.0                  # mean pressure (Pa)
P_SEASONAL_AMP = 60.0           # seasonal CO2 cycle amplitude
P_DIURNAL_AMP = 15.0            # thermal tide amplitude
P_NOISE_STD = 5.0               # natural variability

# Anomaly injection config
ANOMALIES = [
    # (sol, start_hour, end_hour, type, t_air_offset, pressure_offset)
    (12, 10, 16, "dust_event", 15.0, -30.0),
    (25, 0, 6, "cold_front", -20.0, 10.0),
    (37, 14, 18, "hot_anomaly", 25.0, -5.0),
    (45, 2, 8, "nighttime_warming", 30.0, 0.0),
    (52, 8, 14, "pressure_wave", 5.0, 50.0),
    (60, 12, 20, "dust_lifting", 12.0, -25.0),
    (68, 4, 10, "cold_air_intrusion", -25.0, 15.0),
    (73, 16, 22, "late_afternoon_spike", 20.0, -10.0),
    (82, 0, 24, "regional_dust_storm", 10.0, -40.0),
    (91, 6, 12, "morning_pressure_drop", 3.0, -35.0),
    (95, 20, 24, "evening_cooling_anomaly", -18.0, 8.0),
    (15, 12, 14, "brief_hot_spike", 30.0, -8.0),
    (33, 3, 5, "pre_dawn_warming", 25.0, 5.0),
    (55, 10, 12, "pressure_surge", 2.0, 55.0),
    (78, 18, 22, "sunset_temp_anomaly", 18.0, -12.0),
]

np.random.seed(42)


def sol_to_ls(sol: int) -> float:
    """Convert sol number to solar longitude (Ls)."""
    return (LS_AT_LANDING + sol * LS_RATE) % 360.0


def generate_meda_data() -> pd.DataFrame:
    """Generate synthetic MEDA temperature and pressure data."""
    records = []

    for sol in SOLS:
        ls = sol_to_ls(sol)
        ls_rad = np.radians(ls)

        for hour in range(HOURS_PER_SOL):
            ltst = float(hour) + np.random.uniform(0, 0.5)  # add jitter
            ltst_rad = 2 * np.pi * (ltst - 14.0) / 24.0  # peak at 14:00

            # ── Air temperature model ────────────────────────────────────
            t_seasonal = T_AIR_SEASONAL_AMP * np.sin(ls_rad)
            t_diurnal = -T_AIR_DIURNAL_AMP * np.cos(ltst_rad)
            t_noise = np.random.normal(0, T_AIR_NOISE_STD)
            t_air = T_AIR_MEAN + t_seasonal + t_diurnal + t_noise

            # ── Pressure model ───────────────────────────────────────────
            p_seasonal = -P_SEASONAL_AMP * np.cos(ls_rad - np.pi / 4)
            p_diurnal = P_DIURNAL_AMP * np.sin(2 * ltst_rad)  # semi-diurnal tide
            p_noise = np.random.normal(0, P_NOISE_STD)
            pressure = P_MEAN + p_seasonal + p_diurnal + p_noise

            records.append({
                "sol": sol,
                "ltst": round(ltst, 2),
                "ls": round(ls, 2),
                "t_air": round(t_air, 2),
                "pressure": round(pressure, 2),
                "is_anomaly": False,
                "anomaly_type": "none",
            })

    df = pd.DataFrame(records)

    # ── Inject anomalies ─────────────────────────────────────────────────
    for sol, start_h, end_h, atype, t_offset, p_offset in ANOMALIES:
        mask = (df["sol"] == sol) & (df["ltst"] >= start_h) & (df["ltst"] < end_h)
        df.loc[mask, "t_air"] += t_offset
        df.loc[mask, "pressure"] += p_offset
        df.loc[mask, "is_anomaly"] = True
        df.loc[mask, "anomaly_type"] = atype

    print(f"Generated {len(df)} data points across {len(SOLS)} sols")
    print(f"Anomalous points: {df['is_anomaly'].sum()}")
    print(f"Temperature range: {df['t_air'].min():.1f} to {df['t_air'].max():.1f} °C")
    print(f"Pressure range: {df['pressure'].min():.1f} to {df['pressure'].max():.1f} Pa")
    print(f"Ls range: {df['ls'].min():.1f} to {df['ls'].max():.1f}°")

    return df


def main():
    output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_meda_data()
    output_path = output_dir / "meda_clean.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\n[OK] Saved to {output_path}")

    # Also save CSV for easy inspection
    csv_path = output_dir / "meda_clean.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV copy at {csv_path}")


if __name__ == "__main__":
    main()
