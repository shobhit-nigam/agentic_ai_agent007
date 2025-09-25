from pathlib import Path
from simulator import Device, Fleet

OUT = Path(__file__).resolve().parents[1] / "outputs"

# Build a fleet of 2,000 devices
fleet = Fleet([Device(f"D{i:05d}", model=("SensorX-200" if i % 4 == 0 else "SensorX-100")) for i in range(2000)])

# Measure baseline crash rate over 5 epochs
vals = [fleet.tick()["crash_rate"] for _ in range(5)]
baseline = sum(vals) / len(vals)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "baseline.txt").write_text(f"Baseline crash rate ≈ {baseline:.4f}\n")
print("Baseline:", baseline)
