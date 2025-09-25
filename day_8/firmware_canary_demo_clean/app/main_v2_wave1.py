from pathlib import Path
import pandas as pd
from simulator import Device, Fleet
from rollout import make_cohorts, wave_name

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Fleet
fleet = Fleet([Device(f"D{i:05d}", model=("SensorX-200" if i % 4 == 0 else "SensorX-100")) for i in range(2000)])

# Baseline over a few epochs
vals = [fleet.tick()["crash_rate"] for _ in range(5)]
baseline = sum(vals) / len(vals)
print(f"Baseline ≈ {baseline:.4f}")

# Plan cohorts and apply 1%
cohorts = make_cohorts([f"D{i:05d}" for i in range(2000)])
wave = cohorts[0]
print("Wave 1% size:", len(wave))

fleet.apply_firmware(wave, version="1.0.1")

# Observe 10 epochs and store KPIs
rows = []
for epoch in range(10):
    k = fleet.tick()
    rows.append({"epoch": epoch, "tag": "wave_1pct", "crash_rate": k["crash_rate"], "rolled_back": 0})

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(OUT / "kpis_wave1.csv", index=False)
print("Saved → outputs/kpis_wave1.csv")
