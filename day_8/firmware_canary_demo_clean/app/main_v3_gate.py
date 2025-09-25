from pathlib import Path
import pandas as pd
from simulator import Device, Fleet
from rollout import make_cohorts, wave_name
from anomaly import gate_pass

OUT = Path(__file__).resolve().parents[1] / "outputs"

# Fleet & baseline
fleet = Fleet([Device(f"D{i:05d}", model=("SensorX-200" if i % 4 == 0 else "SensorX-100")) for i in range(2000)])
vals = [fleet.tick()["crash_rate"] for _ in range(5)]
baseline = sum(vals) / len(vals)

# Apply 1% and watch
cohorts = make_cohorts([d.device_id for d in fleet.devices])
fleet.apply_firmware(cohorts[0], version="1.0.1")
rows = []
for epoch in range(10):
    k = fleet.tick()
    rows.append({"epoch": epoch, "crash_rate": k["crash_rate"], "rolled_back": 0})

current = rows[-1]["crash_rate"]
print(f"Gate check → baseline={baseline:.4f}, current={current:.4f}")
print("PASS" if gate_pass(baseline, current) else "FAIL")
