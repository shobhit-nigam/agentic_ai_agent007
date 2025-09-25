import json
from pathlib import Path
import pandas as pd
from simulator import Device, Fleet
from rollout import make_cohorts, wave_name
from anomaly import gate_pass
from report import make_report

OUT = Path(__file__).resolve().parents[1] / "outputs"
MAN = Path(__file__).resolve().parents[1] / "manifests" / "1.0.1.json"

# (1) Manifest (pretend we just received it)
MAN.parent.mkdir(parents=True, exist_ok=True)
MAN.write_text(json.dumps({
    "id": "fw-1.0.1", "version": "1.0.1",
    "sha256": "deadbeefdeadbeefdeadbeefdeadbeef",
    "target_models": ["SensorX-100", "SensorX-200"],
    "notes": "Driver patch for radio stability; minor scheduler tweak"
}, indent=2))
print("Manifest saved → manifests/1.0.1.json")

# (2) Fleet & baseline
fleet = Fleet([Device(f"D{i:05d}", model=("SensorX-200" if i % 4 == 0 else "SensorX-100")) for i in range(2000)])
vals = [fleet.tick()["crash_rate"] for _ in range(5)]
baseline = sum(vals) / len(vals)
print(f"Baseline ≈ {baseline:.4f}")

# (3) Plan cohorts and roll waves
cohorts = make_cohorts([d.device_id for d in fleet.devices])
rows = []

def record(tag):
    k = fleet.tick()
    rows.append({
        "epoch": len(rows),
        "tag": tag,
        "crash_rate": k["crash_rate"],
        "rolled_back": 0
    })

# A little baseline run for spacing
for _ in range(5):
    record("baseline")

rolled_back = False
for i, cohort in enumerate(cohorts):
    if rolled_back:
        break
    name = wave_name(i)
    print(f"Applying wave {name} (size={len(cohort)})")
    fleet.apply_firmware(cohort, version="1.0.1")

    for _ in range(10):
        record(f"wave_{name}")

    current = rows[-1]["crash_rate"]
    if not gate_pass(baseline, current, threshold=0.0020):
        print(f"Gate FAILED at wave {name} → rolling back this cohort")
        fleet.apply_firmware(cohort, version="1.0.0")
        rows[-1]["rolled_back"] = 1
        rolled_back = True
        break
    else:
        print(f"Gate PASS at wave {name}")

# (4) Save KPIs and build report
OUT.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "kpis.csv", index=False)

decision = make_report(
    kpi_csv=str(OUT / "kpis.csv"),
    plot_path=str(OUT / "report.png"),
    summary_path=str(OUT / "summary.txt"),
    baseline=baseline,
)
print("Report ready → outputs/report.png, outputs/summary.txt")
print("Decision:", decision)
