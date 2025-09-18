from agent.memory import LongMemory, ShortMemory, is_storable
from agent.planner import make_plan, execute_plan, pretty_plan, pretty_trace

SM = ShortMemory(k=5)
LM = LongMemory()

def _preferences_from_memory(query: str):
    # Very simple: search preferences and try to infer a city keyword
    prefs = {}
    mems = LM.retrieve(query, k=5)
    cities = ["san francisco","new york","chicago","paris","london","mumbai","tokyo","delhi","bangalore"]
    for it in mems:
        t = it.text.lower()
        for c in cities:
            if c in t:
                prefs["city"] = c.title()
                return prefs
    return prefs

def run_goal(goal: str) -> str:
    SM.add("user", goal)

    # Write useful statements into memory
    if goal.lower().startswith("remember "):
        ack = LM.remember(goal, kind="preference", meta={"source":"user"})
        SM.add("assistant", ack)
        return ack
    if is_storable(goal):
        LM.remember(goal, kind="episodic", meta={"source":"conversation"})

    # pull the preferences from memory
    prefs = _preferences_from_memory(goal)

    # 1) PLAN
    steps = make_plan(goal, preferences=prefs)
    plan_text = pretty_plan(steps)

    # 2) Execute 
    trace, outputs = execute_plan(steps)
    trace_text = pretty_trace(trace)

    # assemble reply + store outcome 
    final = ["PLAN:", plan_text, "", "TRACE:", trace_text, "", "FINAL OUTPUTS:"]
    for k, v in outputs.items():
        final.append(f"{k}: {v}")
    reply = "\n".join(final)
    
    SM.add("assistant", reply)
    # Store an episodic outcome
    LM.remember(f"goal='{goal}' executed with steps={len(steps)}", kind="episodic", meta={"source":"planner"})
    return reply
