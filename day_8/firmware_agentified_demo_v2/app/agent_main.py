import json
from pathlib import Path
from simulator import Device, Fleet
from rollout import make_cohorts, wave_name
from agents.base import Message
from agents.tracer import Tracer
from agents.monitor import MonitorAgent
from agents.planner import PlannerAgent
from agents.guard import SafetyGuardAgent
from agents.actuator import ActuatorAgent
from agents.reporter import ReporterAgent

OUT = (Path(__file__).resolve().parents[1] / "outputs")
OUT.mkdir(parents=True, exist_ok=True)

def build_fleet(n=2000):
    return Fleet([Device(f"D{i:05d}", model=("SensorX-200" if i % 4 == 0 else "SensorX-100")) for i in range(n)])

def main():
    trace = Tracer(str(OUT / "agent_trace.jsonl"))

    # (0) Manifest arrives
    man = {"id":"fw-1.0.1","version":"1.0.1"}
    (Path(__file__).resolve().parents[1] / "manifests" / "1.0.1.json").write_text(json.dumps(man, indent=2))

    # (1) Fleet + baseline via MonitorAgent
    fleet = build_fleet()
    monitor = MonitorAgent("monitor", trace=trace)
    baseline_vals = []
    for epoch in range(5):
        obs = monitor.handle(Message("Tick", {"fleet": fleet, "epoch": epoch}))
        baseline_vals.append(obs.payload["kpi"]["crash_rate"])
    baseline = sum(baseline_vals)/len(baseline_vals)

    # (2) Instantiate agents
    device_ids = [d.device_id for d in fleet.devices]
    planner = PlannerAgent("planner", device_ids=device_ids, trace=trace)
    guard   = SafetyGuardAgent("guard", baseline=baseline, threshold=0.0020, trace=trace)
    act     = ActuatorAgent("actuator", trace=trace)
    report  = ReporterAgent("reporter", out_dir=str(OUT), trace=trace)

    rows = []
    def record(tag, kpi, rolled_back=0):
        rows.append({"epoch": len(rows), "tag": tag, "crash_rate": kpi["crash_rate"], "rolled_back": rolled_back})

    # (3) Pre-wave baseline ticks for spacing
    for _ in range(5):
        obs = monitor.handle(Message("Tick", {"fleet": fleet, "epoch": len(rows)}))
        record("baseline", obs.payload["kpi"])

    # (4) Wave loop
    cohorts = make_cohorts(device_ids)
    rolled_back = False
    for i, cohort in enumerate(cohorts):
        if rolled_back: break

        # Planner proposes next wave
        proposal = planner.handle(Message("PlanNext", {}))
        cmd = {"type": proposal.type, **proposal.payload}

        # Actuator applies
        applied = act.handle(Message("Cmd", {"cmd": cmd, "fleet": fleet}))

        # Observe 10 epochs
        for _ in range(10):
            obs = monitor.handle(Message("Tick", {"fleet": fleet, "epoch": len(rows)}))
            record(f"wave_{cmd['wave_name']}", obs.payload["kpi"])

        # Guard checks latest KPI
        gate = guard.handle(Message("KPI", {"kpi": rows[-1]}))
        if not gate.payload["pass"]:
            # rollback just-applied cohort
            rb = act.handle(Message("Cmd", {"cmd": {"type":"Rollback","cohort": cmd["cohort"],"wave_name": cmd["wave_name"]}, "fleet": fleet}))
            rows[-1]["rolled_back"] = 1
            rolled_back = True
            break

    # (5) Reporter builds outputs
    rep = report.handle(Message("Rows", {"rows": rows, "baseline": baseline}))
    print("Decision:", rep.payload["decision"])
    print("Artifacts in outputs/: kpis.csv, report.png, summary.txt, agent_trace.jsonl")

if __name__ == "__main__":
    main()
