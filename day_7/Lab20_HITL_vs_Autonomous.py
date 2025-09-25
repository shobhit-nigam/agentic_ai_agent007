# Lab 20 — Human-in-the-Loop (HITL) vs. Autonomous Agents
# ----------------------------------------------------------------------------
# What this lab teaches
# - Where and why to place a Human-in-the-Loop (HITL) approval gate
# - How to compare HITL vs Autonomous modes on quality, safety and speed
# - How to log interventions (approve / edit / reject) into the same trace format
# - How to compute simple metrics for decision-making trade-offs
#
# Runs fully offline (no API key needed). If OPENAI_API_KEY is set, the planner
# can optionally be driven by an LLM; otherwise a heuristic planner is used.
# ----------------------------------------------------------------------------
from __future__ import annotations

import os, re, csv, json, time, random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# Optional LLM planner (falls back to heuristic if not available)
USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))
try:
    if USE_LLM:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
    else:
        ChatOpenAI = None
except Exception:
    USE_LLM = False
    ChatOpenAI = None

# ----------------------------- Tracing ---------------------------------------
class Tracer:
    def __init__(self, run_id: str, out_dir: str = "./logs"):
        self.run_id = run_id
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = f"{out_dir}/{run_id}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, node: str, phase: str, detail: Dict[str, Any]):
        evt = {
            "run_id": self.run_id,
            "ts_iso": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "node": node,
            "phase": phase,
            "detail": detail,
        }
        self._fh.write(json.dumps(evt, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

# ----------------------------- Tools -----------------------------------------
def tool_weather_live(input_city: str, simulate_fail: bool=False) -> Tuple[str, Dict[str, Any]]:
    """Simulated live weather (risky). Returns (status, payload)."""
    # Simulate occasional failures/timeouts to motivate HITL
    if simulate_fail:
        return "ERROR|WEATHER", {"code": "TIMEOUT", "message": "Live weather API timed out."}
    # Deterministic pseudo-weather for demo
    city = input_city.strip()
    catalog = {
        "Paris": ("sunny", 24.0),
        "Chicago": ("cloudy", 18.0),
        "London": ("overcast", 17.0),
    }
    cond, temp = catalog.get(city, ("clear", 22.0))
    return "OK|WEATHER", {"city": city, "temp_c": temp, "condition": cond, "source": "live"}

def tool_weather_cache(input_city: str) -> Tuple[str, Dict[str, Any]]:
    """Fast, reliable cached weather (safe)."""
    city = input_city.strip()
    # Slightly different temps to show it's a separate source
    catalog = {
        "Paris": ("sunny", 23.0),
        "Chicago": ("cloudy", 18.0),
        "London": ("overcast", 17.0),
    }
    cond, temp = catalog.get(city, ("clear", 21.5))
    return "OK|WEATHER", {"city": city, "temp_c": temp, "condition": cond, "source": "cache"}

def tool_suggest_city_activities(city: str, weather: Optional[str]=None) -> Tuple[str, Dict[str, Any]]:
    """Simple weather-aware suggestions."""
    city_l = city.lower()
    if "paris" in city_l:
        indoor, outdoor = "Louvre Museum", "Seine River Walk"
    elif "chicago" in city_l:
        indoor, outdoor = "Art Institute of Chicago", "Chicago Riverwalk"
    elif "london" in city_l:
        indoor, outdoor = "British Museum", "Hyde Park"
    else:
        indoor, outdoor = "Local museum", "Central park/riverfront"

    summary = f"City: {city}. Indoor: {indoor}. Outdoor: {outdoor}."
    if weather:
        summary += f" (Weather: {weather})"
    return "OK|SUGGEST", {"city": city, "indoor": indoor, "outdoor": outdoor, "text": summary}

def tool_calculator_strict(expr: str) -> Tuple[str, Dict[str, Any]]:
    """Strict numeric evaluation of simple arithmetic 'a [+|-|*|/] b [+ ...]'. Returns numeric-only string in 'value'."""
    # Security: only digits, spaces, and operators
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", expr):
        return "ERROR|CALC", {"code": "VALIDATION", "message": "Invalid characters in expression."}
    try:
        # Very restricted eval
        value = eval(expr, {"__builtins__": {}}, {})
        if not isinstance(value, (int, float)):
            return "ERROR|CALC", {"code": "TYPE", "message": "Expression did not evaluate to a number."}
        return "OK|CALC", {"value": float(value)}
    except Exception as e:
        return "ERROR|CALC", {"code": "EVAL", "message": f"Evaluation error: {e}"}

# ----------------------------- Planner ---------------------------------------
PLANNER_SYS = (
    "You are a planner. Output ONLY JSON with either:\n"
    '{ "done": false, "next_step": {"tool": "<tool_name>", "input": "<string or JSON>"} }\n'
    'or { "done": true, "final_answer": "<concise final answer>" }.\n'
    "Never call tools yourself; just propose the next step. Tools: weather_live(city), weather_cache(city), "
    "suggest_city_activities({city, weather}), calculator_strict(expression)."
)

def parse_city_from_prompt(text: str) -> Optional[str]:
    for c in ["Paris", "Chicago", "London"]:
        if c.lower() in text.lower():
            return c
    return None

def heuristic_planner(user_prompt: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic planner used when no LLM available or as fallback."""
    has_weather = state.get("has_weather", False)
    math_expr = state.get("math_expr")
    city = state.get("city") or parse_city_from_prompt(user_prompt) or "Paris"

    # 1) If math requested and not computed yet, plan calculator
    if math_expr and not state.get("math_done"):
        return {"done": False, "next_step": {"tool": "calculator_strict", "input": math_expr}}

    # 2) If we need weather first
    if not has_weather and any(k in user_prompt.lower() for k in ["weather", "plan", "indoor", "outdoor"]):
        return {"done": False, "next_step": {"tool": "weather_live", "input": city}}

    # 3) If we have weather and no suggestions yet → suggest
    if has_weather and not state.get("suggest_done"):
        w = state.get("weather", {})
        weather_str = f"{w.get('condition','unknown')}, {w.get('temp_c','?')}°C"
        return {"done": False, "next_step": {"tool": "suggest_city_activities", "input": {"city": city, "weather": weather_str}}}

    # 4) Finalize
    if state.get("suggest_done"):
        city = state.get("city") or city
        indoor = state.get("indoor") or "an indoor place"
        outdoor = state.get("outdoor") or "an outdoor place"
        parts = [f"In {city}, consider:", f"Indoor — {indoor}.", f"Outdoor — {outdoor}."]
        if state.get("math_done"):
            parts.append(f"Result — {state.get('math_value')}")
        return {"done": True, "final_answer": " ".join(parts)}

    # Default: try weather
    return {"done": False, "next_step": {"tool": "weather_live", "input": city}}

def llm_planner(user_prompt: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Calls an LLM to produce the same JSON as heuristic_planner."""
    if not USE_LLM or ChatOpenAI is None:
        return heuristic_planner(user_prompt, state)
    messages = [
        SystemMessage(content=PLANNER_SYS),
        HumanMessage(content=f"User: {user_prompt}\nState: {json.dumps(state)}\nReturn JSON ONLY.")
    ]
    llm = ChatOpenAI(model=os.environ.get("LAB20_PLANNER_MODEL", "gpt-4o-mini"), temperature=0)
    resp = llm.invoke(messages)
    txt = (resp.content or "").strip()
    try:
        data = json.loads(txt)
        if isinstance(data, dict) and "done" in data:
            return data
    except Exception:
        pass
    # Fallback
    return heuristic_planner(user_prompt, state)

# ----------------------------- HITL Gate -------------------------------------
@dataclass
class HitlDecision:
    action: str  # "approve" | "edit" | "reject"
    next_step: Optional[Dict[str, Any]] = None
    note: str = ""

def assess_risk(next_step: Dict[str, Any], state: Dict[str, Any]) -> int:
    """Simple risk score: live calls risky, finalize without math risky if math requested, unknown city risky."""
    tool = (next_step or {}).get("tool")
    risk = 0
    if tool == "weather_live":
        risk += 2
    if tool == "finalize" and state.get("math_expr") and not state.get("math_done"):
        risk += 3
    if (state.get("city") or "Paris") not in ["Paris", "Chicago", "London"]:
        risk += 2
    return risk

def hitl_gate(next_step: Dict[str, Any], state: Dict[str, Any], mode: str, tracer: Tracer,
              simulate_live_failure: bool=False) -> HitlDecision:
    """Simulated Human-in-the-Loop. In AUTONOMOUS mode, passes through. In HITL, approves/edits/rejects by policy."""
    tracer.log("hitl", "before", {"mode": mode, "next_step": next_step, "state_keys": list(state.keys())})
    if mode.lower() == "autonomous":
        tracer.log("hitl", "after", {"decision": "approve"})
        return HitlDecision("approve")

    # HITL mode: inspect risk & recent outcomes
    risk = assess_risk(next_step, state)
    tool = next_step.get("tool")
    # Example policy: if attempting weather_live and we know live tends to fail (simulated), edit to weather_cache
    if tool == "weather_live" and simulate_live_failure:
        edited = {"tool": "weather_cache", "input": next_step.get("input")}
        tracer.log("hitl", "after", {"decision": "edit", "note": "Switch to cache to avoid flakiness.", "edited": edited})
        return HitlDecision("edit", next_step=edited, note="Switch to cache")

    # If finalizing too early, reject
    if tool == "finalize" and (state.get("math_expr") and not state.get("math_done")):
        tracer.log("hitl", "after", {"decision": "reject", "note": "Compute math before finalizing."})
        return HitlDecision("reject", note="Compute math first")

    # Otherwise approve
    tracer.log("hitl", "after", {"decision": "approve", "risk": risk})
    return HitlDecision("approve")

# ----------------------------- Executor --------------------------------------
def execute_step(step: Dict[str, Any], state: Dict[str, Any], tracer: Tracer,
                 simulate_live_failure: bool=False) -> None:
    tool = step.get("tool")
    inp = step.get("input")
    tracer.log("executor", "before", {"tool": tool, "input": inp})
    if tool == "weather_live":
        status, payload = tool_weather_live(str(inp), simulate_fail=simulate_live_failure)
        tracer.log("tools", "after", {"tool": tool, "status": status, "payload": payload})
        if status.startswith("OK"):
            state["has_weather"] = True
            state["weather"] = payload
            state["city"] = payload.get("city")
        else:
            state.setdefault("errors", []).append({"tool": tool, "status": status, "payload": payload})
    elif tool == "weather_cache":
        status, payload = tool_weather_cache(str(inp))
        tracer.log("tools", "after", {"tool": tool, "status": status, "payload": payload})
        if status.startswith("OK"):
            state["has_weather"] = True
            state["weather"] = payload
            state["city"] = payload.get("city")
    elif tool == "suggest_city_activities":
        if isinstance(inp, dict):
            city = inp.get("city") or (state.get("city") or "Paris")
            weather = inp.get("weather") or (state.get("weather", {}).get("condition"))
        else:
            city = str(inp)
            weather = state.get("weather", {}).get("condition")
        status, payload = tool_suggest_city_activities(city, weather)
        tracer.log("tools", "after", {"tool": tool, "status": status, "payload": payload})
        if status.startswith("OK"):
            state["suggest_done"] = True
            state["indoor"] = payload.get("indoor")
            state["outdoor"] = payload.get("outdoor")
    elif tool == "calculator_strict":
        status, payload = tool_calculator_strict(str(inp))
        tracer.log("tools", "after", {"tool": tool, "status": status, "payload": payload})
        if status.startswith("OK"):
            state["math_done"] = True
            state["math_value"] = payload.get("value")
        else:
            state.setdefault("errors", []).append({"tool": tool, "status": status, "payload": payload})
    elif tool == "finalize":
        # No-op: planner will produce final text; we just mark intent
        tracer.log("tools", "after", {"tool": tool, "status": "OK|FINALIZE", "payload": {}})
    else:
        tracer.log("tools", "after", {"tool": tool, "status": "ERROR|UNKNOWN", "payload": {"message": "Unknown tool"}})

# ----------------------------- Runner ----------------------------------------
@dataclass
class RunConfig:
    mode: str = "autonomous"              # "autonomous" | "hitl"
    simulate_live_failure: bool = False   # force weather_live failure to trigger edit
    recursion_limit: int = 20             # max planner turns

@dataclass
class RunResult:
    case_id: str
    mode: str
    planner_turns: int = 0
    tool_calls: int = 0
    approvals: int = 0
    edits: int = 0
    rejects: int = 0
    added_latency_ms: int = 0
    final_answer: str = ""
    state: Dict[str, Any] = field(default_factory=dict)

def detect_math_expr(prompt: str) -> Optional[str]:
    # naive: look for numbers and + - * / in prompt
    m = re.findall(r"[0-9\.\s\+\-\*\/\(\)]+", prompt)
    # Filter to something meaningful
    for cand in m:
        if any(op in cand for op in ["+","-","*","/"]) and any(ch.isdigit() for ch in cand):
            return cand.strip()
    return None

def run_case(case_id: str, prompt: str, cfg: RunConfig, tracer: Tracer) -> RunResult:
    tracer.log("meta", "before", {"case_id": case_id, "prompt": prompt, "mode": cfg.mode})
    state: Dict[str, Any] = {
        "city": parse_city_from_prompt(prompt) or None,
        "math_expr": detect_math_expr(prompt)
    }
    turns = 0; tool_calls = 0; approvals = 0; edits = 0; rejects = 0; added_latency = 0

    while turns < cfg.recursion_limit:
        # Planner proposes a step (LLM or heuristic)
        plan = llm_planner(prompt, state)
        turns += 1
        tracer.log("planner", "after", {"plan": plan, "turn": turns})

        if plan.get("done"):
            # Finalize
            if not state.get("suggest_done") and state.get("has_weather"):
                # ensure reasonable final
                w = state.get("weather", {})
                city = state.get("city") or "the city"
                indoor = state.get("indoor") or "an indoor place"
                outdoor = state.get("outdoor") or "an outdoor place"
                plan["final_answer"] = (
                    f"Weather in {city}: {w.get('condition')} {w.get('temp_c')}°C. "
                    f"Indoor: {indoor}. Outdoor: {outdoor}."
                )
            result_text = plan.get("final_answer", "")
            tracer.log("executor", "summary", {"final_answer": result_text})
            tracer.log("meta", "after", {"case_id": case_id})
            return RunResult(
                case_id=case_id, mode=cfg.mode,
                planner_turns=turns, tool_calls=tool_calls,
                approvals=approvals, edits=edits, rejects=rejects,
                added_latency_ms=added_latency, final_answer=result_text, state=state
            )

        next_step = plan.get("next_step") or {}
        # HITL decision
        decision = hitl_gate(next_step, state, cfg.mode, tracer, simulate_live_failure=cfg.simulate_live_failure)
        if decision.action == "approve":
            approvals += 1
        elif decision.action == "edit":
            edits += 1
            next_step = decision.next_step or next_step
        elif decision.action == "reject":
            rejects += 1
            # Simple replan: force weather first if math pending; otherwise switch to cache if live
            if next_step.get("tool") == "finalize" and state.get("math_expr") and not state.get("math_done"):
                next_step = {"tool": "calculator_strict", "input": state["math_expr"]}
            elif next_step.get("tool") == "weather_live":
                next_step = {"tool": "weather_cache", "input": next_step.get("input")}
            else:
                # no-op reject, continue loop
                continue
        # Simulate human latency (only in HITL)
        if cfg.mode.lower() == "hitl":
            time.sleep(0.05)  # small pause for demo
            added_latency += 50

        # Execute
        execute_step(next_step, state, tracer, simulate_live_failure=cfg.simulate_live_failure)
        tool_calls += 1

    tracer.log("meta", "after", {"case_id": case_id, "note": "recursion_limit_reached"})
    return RunResult(case_id=case_id, mode=cfg.mode, planner_turns=turns, tool_calls=tool_calls,
                     approvals=approvals, edits=edits, rejects=rejects, added_latency_ms=added_latency,
                     final_answer="(No final; recursion limit reached)", state=state)

# ----------------------------- Demos & Reporting -----------------------------
def write_report(rows: List[RunResult], prefix: str):
    os.makedirs("./logs", exist_ok=True)
    # JSON
    data = {"summary": {
                "total": len(rows),
                "by_mode": {
                    "autonomous": sum(1 for r in rows if r.mode=="autonomous"),
                    "hitl": sum(1 for r in rows if r.mode=="hitl"),
                },
                "avg_planner_turns": round(sum(r.planner_turns for r in rows)/max(1,len(rows)),2),
                "avg_tool_calls": round(sum(r.tool_calls for r in rows)/max(1,len(rows)),2),
                "total_approvals": sum(r.approvals for r in rows),
                "total_edits": sum(r.edits for r in rows),
                "total_rejects": sum(r.rejects for r in rows),
                "total_added_latency_ms": sum(r.added_latency_ms for r in rows),
            },
            "rows": [asdict(r) for r in rows]}
    jpath = f"./logs/{prefix}_report.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # CSV
    cpath = f"./logs/{prefix}_report.csv"
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return jpath, cpath

def demo():
    # AUTONOMOUS vs HITL comparisons
    tracer = Tracer("lab20")
    rows: List[RunResult] = []

    # Demo A: Autonomous, classic plan Paris
    a1 = run_case("auto_paris", "Plan a short evening in Paris. Weather first, then one indoor and one outdoor.", 
                  RunConfig(mode="autonomous"), tracer)
    print("=== DEMO A (Autonomous) ===")
    print("Final:", a1.final_answer); rows.append(a1)

    # Demo B: HITL, live weather edited to cache to avoid flakiness
    b1 = run_case("hitl_chicago", "Plan a short evening in Chicago. Weather first, then activities.", 
                  RunConfig(mode="hitl", simulate_live_failure=True), tracer)
    print("\n=== DEMO B (HITL: live→cache edit) ===")
    print("Final:", b1.final_answer); rows.append(b1)

    # Demo C: HITL, includes math → human rejects finalize until calculator runs
    c1 = run_case("hitl_london_math", "In London, suggest indoor/outdoor and compute 25 + 18 + 12.5 (only the number).",
                  RunConfig(mode="hitl"), tracer)
    print("\n=== DEMO C (HITL: reject premature finalize; require calculator) ===")
    print("Final:", c1.final_answer); rows.append(c1)

    tracer.close()
    jpath, cpath = write_report(rows, "lab20")
    print("\nReport JSON:", jpath)
    print("Report CSV :", cpath)
    print("Trace JSONL:", tracer.path)

if __name__ == "__main__":
    demo()
