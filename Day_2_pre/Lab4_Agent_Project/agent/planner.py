from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from agent.tools import REGISTRY
from agent.policies import call_with_policies
from agent.memory import LongMemory

@dataclass
class Step:
    name: str
    tool: str                 # 'weather', 'wikipedia', or 'llm'
    args: Dict[str, Any]
    depends_on: Optional[str] = None

def make_plan(user_goal: str, *, preferences: Optional[Dict[str, str]] = None) -> List[Step]:
    goal = user_goal.lower()
    prefs = preferences or {}
    steps: List[Step] = []

    # Trip planning (simple rule-based)
    if "trip" in goal or "itinerary" in goal or "day in" in goal:
        city = None
        for c in ["san francisco","new york","chicago","paris","london","mumbai","tokyo","delhi","bangalore"]:
            if c in goal:
                city = c.title(); break
        if not city:
            city = prefs.get("city", "Chicago")
        steps.append(Step("check_weather", "weather", {"city": city}))
        steps.append(Step("hotel_basics", "wikipedia", {"topic": f"Hotels in {city}"}))
        steps.append(Step("draft_itinerary", "llm",
                          {"prompt": f"Create a 1‑day itinerary for {city}. Consider weather and hotel availability."},
                          depends_on="check_weather"))
        return steps

    # Research brief
    if "research" in goal or "report" in goal:
        topic = user_goal.replace("research", "").replace("report", "").strip() or "Agentic AI"
        steps.append(Step("gather_facts", "wikipedia", {"topic": topic}))
        steps.append(Step("summarize", "llm",
                          {"prompt": f"Summarize key points about {topic} for a 1‑pager."},
                          depends_on="gather_facts"))
        return steps

    # Default
    steps.append(Step("answer", "llm", {"prompt": user_goal}))
    return steps

def _invoke(tool: str, args: Dict[str, Any]) -> Tuple[str, int]:
    import time
    t0 = time.time()
    if tool == "llm":
        out = f"(mock LLM) {args.get('prompt', '')[:160]}..."
    else:
        fn = REGISTRY[tool].call
        out = call_with_policies(
            tool_name=tool,
            tool_fn=fn,
            args=args,
            denylist={"delete database", "transfer all money", "format disk"},
            max_attempts=2,
            backoff_sec=0.2,
            enable_logging=True
        )
    dt = int((time.time() - t0) * 1000)
    return out, dt

@dataclass
class StepResult:
    step: Step
    status: str     # "ok" | "error"
    output: str
    ms: int

def execute_plan(steps: List[Step]) -> Tuple[List[StepResult], Dict[str, str]]:
    trace: List[StepResult] = []
    outputs: Dict[str, str] = {}
    for st in steps:
        try:
            call_args = dict(st.args)
            if st.depends_on:
                upstream = outputs.get(st.depends_on)
                if upstream is None:
                    raise RuntimeError(f"Missing output from dependency '{st.depends_on}'")
                call_args["upstream"] = upstream
            out, ms = _invoke(st.tool, call_args)
            outputs[st.name] = out
            trace.append(StepResult(st, "ok", out, ms))
        except Exception as e:
            outputs[st.name] = f"(error) {e}"
            trace.append(StepResult(st, "error", str(e), 0))
            # policy: continue so user can see the whole trace
    return trace, outputs

def pretty_plan(steps: List[Step]) -> str:
    lines = []
    for i, s in enumerate(steps, 1):
        dep = f" (depends on: {s.depends_on})" if s.depends_on else ""
        lines.append(f"{i}. {s.name} -> tool={s.tool}, args={s.args}{dep}")
    return "\n".join(lines)

def pretty_trace(trace: List[StepResult]) -> str:
    return "\n".join(f"- {r.step.name} [{r.step.tool}] -> {r.status} in {r.ms}ms" for r in trace)
