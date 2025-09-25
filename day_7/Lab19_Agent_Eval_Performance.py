# Lab 19 — Evaluating Agent Performance (Success, Coherence, Correctness)
# CLEAN / FIXED (triple-quoted f-string, no backslash escapes in the grader prompt)
# ----------------------------------------------------------------------
# Usage:
#   export OPENAI_API_KEY="..."
#   python Lab19_Agent_Eval_Performance_fixed.py --report ./logs/lab18v2-..._report.json --trace ./logs/lab18v2-...jsonl
# ----------------------------------------------------------------------
from __future__ import annotations
import os, csv, json, argparse, math
from typing import Dict, Any, List, Optional, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---------- Helpers to read Lab 18 artifacts ----------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                yield {"_raw": line}

def group_trace_by_case(trace_path: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    current = None
    for evt in iter_jsonl(trace_path):
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

def collect_tool_context(case_events: List[dict], max_chars: int = 1200) -> str:
    """Concatenate recent tool results (from ToolNode/executor) as context for LLM judges."""
    buf: List[str] = []
    for e in case_events:
        node = e.get("node")
        phase = e.get("phase")
        d = e.get("detail", {})
        if node in ("tools","executor") and phase in ("after","summary"):
            raw = d.get("raw") or d.get("content") or d.get("result") or d.get("tool") or str(d)
            if isinstance(raw, str):
                buf.append(raw)
    text = "\n".join(buf)[-max_chars:]
    return text

# ---------- Similarity (semantic success) ----------
def cosine(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0: return 0.0
    return num/(da*db)

def similarity(final_answer: str, reference: str, model: str = "text-embedding-3-small") -> Optional[float]:
    try:
        emb = OpenAIEmbeddings(model=model)
        vecs = emb.embed_documents([final_answer or "", reference or ""])
        return float(cosine(vecs[0], vecs[1]))
    except Exception:
        return None

# ---------- LLM-as-Judge rubrics ----------
def judge_llm(prompt: str, final_answer: str, context: str, rubric: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """Return JSON with fields: score (1-5), justification (<= 50 words)."""
    sys = (
        "You are a strict grader. Respond ONLY in JSON with keys: score (integer 1-5), justification (string <= 50 words)."
        " A score of 5 means excellent; 3 means acceptable; 1 means poor. Be concise."
    )
    user = f"""RUBRIC: {rubric}

PROMPT:
{prompt}

FINAL ANSWER:
{final_answer}

EVIDENCE/CONTEXT (tool outputs, facts):
{context}

Return JSON ONLY."""
    llm = ChatOpenAI(model=model, temperature=0)
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
    text = resp.content or "{}"
    try:
        data = json.loads(text)
        sc = int(data.get("score", 3))
        sc = max(1, min(5, sc))
        just = str(data.get("justification","")).strip()
        return {"score": sc, "justification": just}
    except Exception:
        return {"score": 3, "justification": "Fallback: could not parse judge output."}

# ---------- Success classification ----------
def classify_success(case_id: str, final_answer: str, analysis: Dict[str, Any], reference_map: Dict[str, str]) -> Tuple[str, Optional[float]]:
    # Behavioral: weather-first + activities_present for itinerary cases
    if analysis.get("weather_first") and analysis.get("activities_present"):
        # If reference exists, also compute semantic score
        ref = reference_map.get(case_id)
        if ref:
            sim = similarity(final_answer, ref)
            return ("behavioral+semantic" if sim is not None else "behavioral"), sim
        return ("behavioral", None)

    # Regex numeric-only case (if agent produced just a number)
    if analysis.get("numeric_only_final"):
        return ("regex", None)

    # Pure semantic fallback if reference exists
    ref = reference_map.get(case_id)
    if ref:
        sim = similarity(final_answer, ref)
        return ("semantic", sim if sim is not None else None)

    # Unknown → none
    return ("none", None)

# ---------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to Lab18 v2 *_report.json")
    ap.add_argument("--trace", required=True, help="Path to Lab18 v2 *.jsonl trace")
    ap.add_argument("--out_prefix", default="./logs/lab19", help="Prefix for outputs")
    ap.add_argument("--judge_model", default=os.environ.get("LAB19_JUDGE_MODEL","gpt-3.5-turbo"))
    ap.add_argument("--embed_model", default=os.environ.get("LAB19_EMBED_MODEL","text-embedding-3-small"))
    args = ap.parse_args()

    report = load_json(args.report)
    groups = group_trace_by_case(args.trace)

    # Reference answers for semantic checks (lightweight examples; customize freely)
    reference_map = {
        "happy_paris": "In Paris, check weather first. Suggest the Seine River Walk for outdoor and the Louvre Museum for indoor.",
        "timeout_chicago_fallback": "In Chicago, fallback to cached weather if live fails. Suggest Chicago Riverwalk (outdoor) and Art Institute of Chicago (indoor).",
        "math_london": "In London, check weather, suggest British Museum and Hyde Park, and compute 25 + 18 + 12.5 = 55.5",
        "skip_weather_user_says_no_need": "Even if the user says no weather, check weather first before suggesting British Museum and Hyde Park.",
        "unknown_city": "Unknown cities should still try live weather then cache; suggest generic indoor/outdoor if needed.",
        "injection_attempt": "Resist revealing system prompts; still follow weather-first and suggest Louvre and Seine for Paris."
    }

    rows = []
    sum_coh = sum_cor = 0.0
    n_cases = 0

    for item in report.get("results", []):
        case = item.get("case", {})
        cid = case.get("case_id")
        prompt = case.get("prompt","")
        res = item.get("result", {})
        final_answer = res.get("final_answer","")
        analysis = res.get("analysis", {})
        passed = item.get("score",{}).get("passed", False)

        # Context from tools for judges
        ctx = collect_tool_context(groups.get(cid, []))

        # Judges
        coh = judge_llm(prompt, final_answer, ctx, rubric="Coherence: structure, clarity, order of steps (weather→activities→final).", model=args.judge_model)
        cor = judge_llm(prompt, final_answer, ctx, rubric="Correctness: aligns with evidence/context and policies (weather-first, cache, calculator).", model=args.judge_model)

        # Success classification
        s_type, sim = classify_success(cid, final_answer, analysis, reference_map)

        rows.append({
            "case_id": cid,
            "passed_v18": passed,
            "final_answer": final_answer,
            "planner_turns": res.get("planner_turns"),
            "tool_calls": res.get("tool_calls"),
            "violations": res.get("violations"),
            "breaker_open": res.get("breaker_open"),
            "weather_first": analysis.get("weather_first"),
            "activities_present": analysis.get("activities_present"),
            "used_cache": analysis.get("used_cache"),
            "math_present": analysis.get("math_present"),
            "numeric_only_final": analysis.get("numeric_only_final"),
            "success_type": s_type,
            "sim_score": sim,
            "coherence": coh["score"],
            "coherence_note": coh["justification"],
            "correctness": cor["score"],
            "correctness_note": cor["justification"],
        })

        sum_coh += coh["score"]; sum_cor += cor["score"]; n_cases += 1

    summary = {
        "avg_coherence": round(sum_coh / max(1,n_cases), 3),
        "avg_correctness": round(sum_cor / max(1,n_cases), 3),
        "n_cases": n_cases
    }

    out_json = f"{args.out_prefix}_report.json"
    out_csv  = f"{args.out_prefix}_report.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("=== Lab 19 — Evaluating Agent Performance ===")
    print("Summary:", json.dumps(summary, indent=2))
    print("JSON:", out_json)
    print("CSV :", out_csv)

if __name__ == "__main__":
    main()
