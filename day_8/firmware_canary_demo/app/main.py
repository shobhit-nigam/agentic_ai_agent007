
import json, random, time
from pathlib import Path
import pandas as pd

from simulator import Device, Fleet
from rollout import make_cohorts, wave_name
from anomaly import gate_pass
from report import make_report

OUT = Path(__file__).resolve().parents[1] / "outputs"
MAN = Path(__file__).resolve().parents[1] / "manifests" / "1.0.1.json"

def log(msg):
    (OUT / "rollout_log.txt").open("a").write(msg + "\n")
    print(msg)

def build_fleet(n=2000):
    devices = []
    for i in range(n):
        model = "SensorX-200" if i % 4 == 0 else "SensorX-100"
        devices.append(Device(device_id=f"D{i:05d}", model=model))
    return Fleet(devices=devices)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rollout_log.txt").write_text("")

    manifest = json.loads(MAN.read_text())
    version = manifest["version"]
    log(f"Received manifest {manifest['id']} version {version}")

    # Create fleet & measure baseline for a few epochs
    fleet = build_fleet()
    baseline_epochs = 5
    baseline_vals = []
    for e in range(baseline_epochs):
        kpi = fleet.tick()
        baseline_vals.append(kpi["crash_rate"])
    baseline = sum(baseline_vals) / len(baseline_vals)
    log(f"Baseline crash rate ≈ {baseline:.4f}")

    # Plan cohorts
    cohorts = make_cohorts([d.device_id for d in fleet.devices])
    log(f"Planned {len(cohorts)} waves; first sizes: {[len(c) for c in cohorts[:4]]}")

    # Dataframe for KPIs
    rows = []
    epoch = 0
    def record(tag):
        nonlocal epoch
        k = fleet.tick()
        rows.append({
            "epoch": epoch,
            "tag": tag,
            "crash_rate": k["crash_rate"],
            "crashes": k["crashes"],
            "population": k["population"],
            "rolled_back": 0,
        })
        epoch += 1
        return k

    # Run baseline a bit more for chart separation
    for _ in range(5):
        record("baseline")

    # Rollout waves
    rolled_back = False
    for i, cohort in enumerate(cohorts):
        if rolled_back:
            break
        name = wave_name(i)
        log(f"--- Applying wave {name} ({len(cohort)} devices) ---")
        fleet.apply_firmware(cohort_ids=cohort, version=version)

        # Stabilize and monitor for 10 epochs
        for _ in range(10):
            k = record(f"wave_{name}")
        current_cr = rows[-1]["crash_rate"]

        # Gate check vs baseline
        if not gate_pass(baseline, current_cr, threshold=0.0020):
            log(f"Gate FAILED after wave {name}: baseline={baseline:.4f}, current={current_cr:.4f}")
            log("Rolling back affected cohort...")
            # Rollback: set firmware back to 1.0.0 for just-applied cohort
            fleet.apply_firmware(cohort_ids=cohort, version="1.0.0")
            # Mark rolled_back in last row for visibility
            rows[-1]["rolled_back"] = 1
            rolled_back = True
            break
        else:
            log(f"Gate PASS after wave {name}: {current_cr:.4f} (Δ={current_cr - baseline:.4f})")

    # If not rolled back, run a few consolidation epochs
    if not rolled_back:
        for _ in range(10):
            record("post_rollout")

    # Save KPIs
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kpis.csv", index=False)

    # Report (plot + summary)
    decision = make_report(
        kpi_csv=str(OUT / "kpis.csv"),
        plot_path=str(OUT / "report.png"),
        summary_path=str(OUT / "summary.txt"),
        baseline=baseline,
    )
    log(f"Report generated: decision={decision}")
    log(f"Artifacts: kpis.csv, report.png, summary.txt, rollout_log.txt")

if __name__ == "__main__":
    main()
