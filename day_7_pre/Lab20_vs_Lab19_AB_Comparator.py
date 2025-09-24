# Lab20_vs_Lab19_AB_Comparator.py
# Compare Lab 20 (HITL/autonomous process metrics) with Lab 19 (quality scores).
from __future__ import annotations
import argparse, json, csv, statistics as stats
from typing import Dict, Any, List, Optional

def read_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def to_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def default_id_map(l20_ids: List[str], l19_ids: List[str]) -> Dict[str,str]:
    # Heuristic defaults for our labs
    mapping = {}
    l19_join = "|".join(l19_ids).lower()
    for cid in l20_ids:
        cl = cid.lower()
        if "paris" in cl:
            mapping[cid] = "happy_paris" if "happy_paris" in l19_join else next((k for k in l19_ids if "paris" in k.lower()), "")
        elif "chicago" in cl:
            mapping[cid] = "timeout_chicago_fallback" if "timeout_chicago_fallback" in l19_join else next((k for k in l19_ids if "chicago" in k.lower()), "")
        elif "london" in cl or "math" in cl:
            mapping[cid] = "math_london" if "math_london" in l19_join else next((k for k in l19_ids if "london" in k.lower() or "math" in k.lower()), "")
        else:
            mapping[cid] = ""
    return mapping

def group_avg(rows: List[Dict[str,Any]], field: str) -> Optional[float]:
    xs = [to_float(r.get(field)) for r in rows if to_float(r.get(field)) is not None]
    return round(stats.mean(xs), 3) if xs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab20_csv", required=True, help="Path to lab20_report.csv")
    ap.add_argument("--lab19_csv", required=True, help="Path to lab19_report.csv")
    ap.add_argument("--id_map_json", default=None, help="Optional JSON mapping {lab20_case_id: lab19_case_id}")
    ap.add_argument("--out_prefix", default="./logs/l20_vs_l19")
    args = ap.parse_args()

    lab20 = read_csv(args.lab20_csv)
    lab19 = read_csv(args.lab19_csv)

    l20_by_id = {r["case_id"]: r for r in lab20}
    l19_by_id = {r["case_id"]: r for r in lab19}

    if args.id_map_json:
        with open(args.id_map_json, "r", encoding="utf-8") as f:
            id_map = json.load(f)
    else:
        id_map = default_id_map(list(l20_by_id.keys()), list(l19_by_id.keys()))

    # Merge per-case
    merged_rows = []
    for l20_id, r20 in l20_by_id.items():
        l19_id = id_map.get(l20_id, "")
        r19 = l19_by_id.get(l19_id) if l19_id else None
        merged_rows.append({
            "lab20_case_id": l20_id,
            "mode": r20.get("mode",""),
            "planner_turns": r20.get("planner_turns",""),
            "tool_calls": r20.get("tool_calls",""),
            "approvals": r20.get("approvals",""),
            "edits": r20.get("edits",""),
            "rejects": r20.get("rejects",""),
            "added_latency_ms": r20.get("added_latency_ms",""),
            "final_answer": r20.get("final_answer",""),
            "lab19_case_id": l19_id,
            "coherence": r19.get("coherence","") if r19 else "",
            "correctness": r19.get("correctness","") if r19 else "",
            "sim_score": r19.get("sim_score","") if r19 else "",
            "success_type": r19.get("success_type","") if r19 else "",
            "passed_v18": r19.get("passed_v18","") if r19 else "",
        })

    # Write rows CSV
    out_rows = args.out_prefix + "_rows.csv"
    Path(out_rows).parent.mkdir(parents=True, exist_ok=True)
    with open(out_rows, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
        w.writeheader(); w.writerows(merged_rows)

    # Summaries per mode
    auto = [r for r in merged_rows if (r.get("mode") or "").lower() == "autonomous"]
    hitl = [r for r in merged_rows if (r.get("mode") or "").lower() == "hitl"]

    def mode_avg(rows, field): return group_avg(rows, field)

    summary = {
        "counts": {"autonomous": len(auto), "hitl": len(hitl), "total": len(merged_rows)},
        "autonomous": {
            "avg_planner_turns": mode_avg(auto, "planner_turns"),
            "avg_tool_calls": mode_avg(auto, "tool_calls"),
            "avg_coherence": mode_avg(auto, "coherence"),
            "avg_correctness": mode_avg(auto, "correctness"),
            "avg_latency_ms": mode_avg(auto, "added_latency_ms"),
        },
        "hitl": {
            "avg_planner_turns": mode_avg(hitl, "planner_turns"),
            "avg_tool_calls": mode_avg(hitl, "tool_calls"),
            "avg_coherence": mode_avg(hitl, "coherence"),
            "avg_correctness": mode_avg(hitl, "correctness"),
            "avg_latency_ms": mode_avg(hitl, "added_latency_ms"),
        }
    }

    def delta(a, b):
        if a is None or b is None: return None
        return round(a - b, 3)

    summary["delta_hitl_minus_auto"] = {
        "planner_turns": delta(summary["hitl"]["avg_planner_turns"], summary["autonomous"]["avg_planner_turns"]),
        "tool_calls": delta(summary["hitl"]["avg_tool_calls"], summary["autonomous"]["avg_tool_calls"]),
        "coherence": delta(summary["hitl"]["avg_coherence"], summary["autonomous"]["avg_coherence"]),
        "correctness": delta(summary["hitl"]["avg_correctness"], summary["autonomous"]["avg_correctness"]),
        "latency_ms": delta(summary["hitl"]["avg_latency_ms"], summary["autonomous"]["avg_latency_ms"]),
    }

    out_json = args.out_prefix + "_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"id_map": id_map, "summary": summary}, f, ensure_ascii=False, indent=2)

    print("=== Lab 20 vs Lab 19 — A/B Summary ===")
    print(json.dumps(summary, indent=2))
    print("Rows CSV :", out_rows)
    print("Summary  :", out_json)

if __name__ == "__main__":
    main()
