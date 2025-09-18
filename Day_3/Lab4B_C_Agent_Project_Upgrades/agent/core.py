from agent.planner import make_plan, execute_plan, pretty_plan, pretty_trace

def run_goal(goal: str, *, live: bool = False, use_openai: bool = False) -> str:
    steps = make_plan(goal, preferences={}, use_live=live, use_openai=use_openai)
    plan_text = pretty_plan(steps)
    trace, outputs = execute_plan(steps)
    trace_text = pretty_trace(trace)

    final = ["PLAN:", plan_text, "", "TRACE:", trace_text, "", "FINAL OUTPUTS:"]
    for k, v in outputs.items():
        final.append(f"{k}: {v}")
    return "\n".join(final)
