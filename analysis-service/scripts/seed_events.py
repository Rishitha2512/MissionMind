"""
Seed historical events for MissionMind similarity search.

These events are generated from the synthetic MEDA data anomalies,
providing labeled historical events for similarity matching and evaluation.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def generate_events(data_path: Path) -> list:
    """Generate event records from the anomaly-labeled MEDA data."""

    df = pd.read_parquet(data_path)
    anomaly_df = df[df["is_anomaly"] == True].copy()

    events = []
    seen_sols = set()

    for _, group in anomaly_df.groupby(["sol", "anomaly_type"]):
        sol = int(group["sol"].iloc[0])
        atype = group["anomaly_type"].iloc[0]

        if (sol, atype) in seen_sols:
            continue
        seen_sols.add((sol, atype))

        ltst_mean = float(group["ltst"].mean())
        ls = float(group["ls"].iloc[0])
        duration = float(group["ltst"].max() - group["ltst"].min()) + 1.0

        # Compute magnitude as max deviation from overall mean
        t_air_global_mean = float(df["t_air"].mean())
        t_air_max_dev = float((group["t_air"] - t_air_global_mean).abs().max())

        p_global_mean = float(df["pressure"].mean())
        p_max_dev = float((group["pressure"] - p_global_mean).abs().max())

        magnitude = max(t_air_max_dev / df["t_air"].std(), p_max_dev / df["pressure"].std())

        # Human-readable summaries
        summaries = {
            "dust_event": f"Dust event on Sol {sol}: elevated temperatures (+{t_air_max_dev:.1f}°C deviation) with pressure drop, lasting ~{duration:.0f} hours during daytime.",
            "cold_front": f"Cold front passage on Sol {sol}: sharp temperature decrease ({t_air_max_dev:.1f}°C below normal) with pressure increase, pre-dawn hours.",
            "hot_anomaly": f"Anomalous heating on Sol {sol}: air temperature {t_air_max_dev:.1f}°C above seasonal baseline during afternoon hours.",
            "nighttime_warming": f"Unusual nighttime warming on Sol {sol}: temperature {t_air_max_dev:.1f}°C warmer than expected during 02:00-08:00 LTST.",
            "pressure_wave": f"Pressure wave event on Sol {sol}: {p_max_dev:.1f} Pa above normal during morning hours, minimal temperature impact.",
            "dust_lifting": f"Dust lifting event on Sol {sol}: temperature elevation with pressure decrease during afternoon, consistent with local dust activity.",
            "cold_air_intrusion": f"Cold air intrusion on Sol {sol}: sudden {t_air_max_dev:.1f}°C temperature drop during morning hours with pressure increase.",
            "late_afternoon_spike": f"Late afternoon temperature spike on Sol {sol}: {t_air_max_dev:.1f}°C above baseline between 16:00-22:00 LTST.",
            "regional_dust_storm": f"Regional dust storm on Sol {sol}: sustained temperature elevation and pressure depression across entire sol, strongest signal of the dataset.",
            "morning_pressure_drop": f"Morning pressure drop on Sol {sol}: {p_max_dev:.1f} Pa decrease during 06:00-12:00 LTST with mild temperature increase.",
            "evening_cooling_anomaly": f"Anomalous evening cooling on Sol {sol}: temperature {t_air_max_dev:.1f}°C below normal during 20:00-24:00 LTST.",
            "brief_hot_spike": f"Brief high-temperature spike on Sol {sol}: {t_air_max_dev:.1f}°C above baseline during midday, lasting ~2 hours.",
            "pre_dawn_warming": f"Pre-dawn warming event on Sol {sol}: unexpected temperature increase during 03:00-05:00 LTST.",
            "pressure_surge": f"Pressure surge on Sol {sol}: {p_max_dev:.1f} Pa above normal during mid-morning.",
            "sunset_temp_anomaly": f"Sunset temperature anomaly on Sol {sol}: elevated temperatures persisting past expected cooling after 18:00 LTST.",
        }

        event = {
            "event_id": f"EVT-{sol:03d}-{atype.upper()[:4]}",
            "summary": summaries.get(atype, f"{atype} event on Sol {sol}"),
            "sol": sol,
            "ltst": round(ltst_mean, 2),
            "ls": round(ls, 2),
            "magnitude": round(magnitude, 3),
            "duration_hours": round(duration, 1),
            "variable": "t_air" if t_air_max_dev > p_max_dev else "pressure",
            "anomaly_type": atype,
            "t_air_deviation": round(t_air_max_dev, 2),
            "pressure_deviation": round(p_max_dev, 2),
        }
        events.append(event)

    return events


def main():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "meda_clean.parquet"

    if not data_path.exists():
        print("Error: MEDA data not found. Run generate_meda_data.py first.")
        sys.exit(1)

    events = generate_events(data_path)
    print(f"Generated {len(events)} events:")
    for e in events:
        print(f"  {e['event_id']}: {e['anomaly_type']} (sol {e['sol']}, mag={e['magnitude']:.2f})")

    output_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "events.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"\n[OK] Saved {len(events)} events to {output_path}")


if __name__ == "__main__":
    main()
