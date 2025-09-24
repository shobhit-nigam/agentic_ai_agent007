
from __future__ import annotations
import json, csv, sys
from pathlib import Path
from typing import List, Dict, Any

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # best-effort
                yield {"_raw": line}

def summary(report_json: str):
    data = _load_json(report_json)
    s = data.get("summary", {})
    print("=== SUMMARY ===")
    for k in ["run_id","trace_path","total","passed","pass_rate"]:
        print(f"{k}: {s.get(k)}")
    print()

def list_cases(report_json: str):
    data = _load_json(report_json)
    print("=== CASES ===")
    for item in data.get("results", []):
        case = item.get("case", {})
        cid = case.get("case_id")
        prompt = case.get("prompt", "").strip().replace("\n"," ")[:120]
        print(f"- {cid}: {prompt}")
    print()

def _group_trace_by_case(trace_jsonl: str) -> Dict[str, List[dict]]:
    """
    Groups events between meta/before and meta/after for each case_id.
    Assumes tracer logged 'meta' with phases 'before' and 'after' containing detail.case_id.
    """
    groups: Dict[str, List[dict]] = {}
    current = None
    for evt in _iter_jsonl(trace_jsonl):
        node = evt.get("node")
        phase = evt.get("phase")
        detail = evt.get("detail", {})
        if node == "meta" and phase == "before":
            current = detail.get("case_id")
            if current and current not in groups:
                groups[current] = []
            continue
        if node == "meta" and phase == "after":
            current = None
            continue
        if current:
            groups[current].append(evt)
    return groups

def show_case(report_json: str, trace_jsonl: str, case_id: str):
    data = _load_json(report_json)
    # find result row
    row = None
    for item in data.get("results", []):
        if item.get("case", {}).get("case_id") == case_id:
            row = item
            break
    if not row:
        print(f"[!] Case '{case_id}' not found in report.")
        return
    # header
    print(f"=== CASE: {case_id} ===")
    print("Prompt:", row["case"]["prompt"])
    print("Simulate:", row["case"]["simulate"])
    print("Expect:", row["case"]["expect"])
    print("\nResult:",
          {k: row["result"][k] for k in ["final_answer","planner_turns","tool_calls","violations","breaker_open"]})
    print("Analysis:", row["result"]["analysis"])
    print("\nChecks:")
    for chk in row["score"]["checks"]:
        status = "PASS" if chk["pass"] else "FAIL"
        print(f" - [{status}] {chk['name']}: {chk['msg']}")

    # timeline from JSONL
    groups = _group_trace_by_case(trace_jsonl)
    evts = groups.get(case_id, [])
    print("\n--- Timeline (planner → validator → executor → tools) ---")
    if not evts:
        print("(no events found in trace for this case id)")
    for e in evts:
        ts = e.get("ts_iso","")
        node = e.get("node","")
        phase = e.get("phase","")
        d = e.get("detail",{})
        # abbreviate big fields
        raw = d.get("raw")
        if isinstance(raw, str) and len(raw) > 140:
            d = dict(d)
            d["raw"] = raw[:140] + "…"
        print(f"{ts} | {node}.{phase} | {d}")

def to_csv_from_jsonl(trace_jsonl: str, out_csv: str):
    # flatten a few fields for spreadsheet viewing
    rows = []
    for e in _iter_jsonl(trace_jsonl):
        rows.append({
            "ts": e.get("ts"),
            "ts_iso": e.get("ts_iso"),
            "node": e.get("node"),
            "phase": e.get("phase"),
            "detail": json.dumps(e.get("detail", {}), ensure_ascii=False)
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts","ts_iso","node","phase","detail"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Lab18 log/report reader")
    ap.add_argument("--report", required=True, help="Path to *_report.json")
    ap.add_argument("--trace", required=False, help="Path to *.jsonl trace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    sub.add_parser("cases")
    p_show = sub.add_parser("case"); p_show.add_argument("case_id")
    p_csv = sub.add_parser("trace2csv"); p_csv.add_argument("out_csv")
    args = ap.parse_args()

    if args.cmd == "summary":
        summary(args.report)
    elif args.cmd == "cases":
        list_cases(args.report)
    elif args.cmd == "case":
        if not args.trace:
            print("[!] --trace is required for 'case'")
            sys.exit(2)
        show_case(args.report, args.trace, args.case_id)
    elif args.cmd == "trace2csv":
        if not args.trace:
            print("[!] --trace is required for 'trace2csv'")
            sys.exit(2)
        to_csv_from_jsonl(args.trace, args.out_csv)

if __name__ == "__main__":
    main()
