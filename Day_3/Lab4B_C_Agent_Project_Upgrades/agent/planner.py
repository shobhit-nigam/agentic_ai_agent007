from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from agent.tools import REGISTRY
from agent.policies import call_with_policies

@dataclass
class Step:
    name: str
    tool: str
    args: Dict[str, Any]
    depends_on: Optional[str] = None

def make_plan(user_goal: str, *, preferences: Optional[Dict[str, str]] = None, use_live: bool = False, use_openai: bool = False) -> List[Step]:
    goal = user_goal.lower()
    prefs = preferences or {}
    steps: List[Step] = []

    weather_tool_name = "live_weather" if use_live else "weather"
    wiki_tool_name    = "wiki_summary" if use_live else "wikipedia"
    llm_tool_name     = "llm_openai"   if use_openai else "llm"

    if "trip" in goal or "itinerary" in goal or "day in" in goal:
        city = None
        for c in ["san francisco","new york","chicago","paris","london","mumbai","tokyo","delhi","bangalore"]:
            if c in goal:
                city = c.title(); break
        if not city:
            city = prefs.get("city", "Chicago")
        steps.append(Step("check_weather", weather_tool_name, {"city": city}))
        steps.append(Step("hotel_basics", wiki_tool_name, {"topic": f"Hotels in {city}"}))
        steps.append(Step("draft_itinerary", llm_tool_name, {"prompt": f"Create a 1-day itinerary for {city}. Consider weather and hotels."},
                          depends_on="check_weather"))
        return steps

    if "research" in goal or "report" in goal:
        topic = user_goal.replace("research","").replace("report","").strip() or "Agentic AI"
        steps.append(Step("gather_facts", wiki_tool_name, {"topic": topic}))
        steps.append(Step("summarize", llm_tool_name, {"prompt": f"Summarize key points about {topic} for a 1-pager."},
                          depends_on="gather_facts"))
        return steps

    return [Step("answer", llm_tool_name, {"prompt": user_goal})]

def _invoke(tool: str, args: Dict[str, Any]) -> Tuple[str, int]:
    import time
    t0 = time.time()
    if tool == "llm":
        upstream = args.get("upstream")
        prompt = args.get("prompt","")
        if upstream:
            prompt = prompt + "\n\nContext (from previous step):\n" + str(upstream)
        out = f"(mock LLM) {prompt[:200]}..."
    else:
        fn = REGISTRY[tool].call
        out = call_with_policies(
            tool_name=tool,
            tool_fn=fn,
            args=args,
            denylist={"delete database", "transfer all money", "format disk"},
            max_attempts=2,
            backoff_sec=0.3,
            enable_logging=True
        )
    dt = int((time.time() - t0) * 1000)
    return out, dt

@dataclass
class StepResult:
    step: Step
    status: str
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
    return trace, outputs

def pretty_plan(steps: List[Step]) -> str:
    lines = []
    for i, s in enumerate(steps, 1):
        dep = f" (depends on: {s.depends_on})" if s.depends_on else ""
        lines.append(f"{i}. {s.name} -> tool={s.tool}, args={s.args}{dep}")
    return "\n".join(lines)

def pretty_trace(trace: List[StepResult]) -> str:
    return "\n".join(f"- {r.step.name} [{r.step.tool}] -> {r.status} in {r.ms}ms" for r in trace)
