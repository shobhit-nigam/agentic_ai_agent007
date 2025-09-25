# Firmware Canary Rollout — Mini Agentic Demo

A small, **output-driven** end-to-end project you can run live in class.
It simulates a firmware rollout in **waves** (1% → 10% → 50% → 100%),
monitors device KPIs (crash rate), **auto-rolls back** on anomaly, and generates a plot/report.

## TL;DR (Live Demo)
```bash
# 1) Create venv (optional)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
python app/main.py

# 4) See outputs
ls outputs/
# -> kpis.csv, rollout_log.txt, report.png, summary.txt
```

## What This Shows (Wow Moments)
- **Staged canary waves** with guardrails
- **Automated anomaly detection** (crash-rate delta gate)
- **Immediate rollback** if gate fails
- **Auto-generated plot** (report.png) and **plain-English summary**

## Project Layout
```
firmware_canary_demo/
  app/
    main.py         # Orchestrates rollout
    simulator.py    # Device & KPI simulation
    rollout.py      # Cohort planning & apply
    anomaly.py      # Gate logic
    report.py       # Generates plot + summary
  manifests/
    1.0.1.json      # Example firmware manifest
  outputs/          # Generated: logs, KPIs, plots
  requirements.txt
  README.md
  run_demo.sh
```
