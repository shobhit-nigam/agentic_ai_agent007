from typing import Tuple, Optional
from agent.tools import REGISTRY
from agent.policies import call_with_policies
from agent.memory import ShortMemory, LongMemory, is_storable

SM = ShortMemory(k=5)
LM = LongMemory()

def choose_tool(query: str) -> Tuple[str, str]:
    q = query.lower()
    if "weather" in q or "forecast" in q:
        return "weather", "matched keyword: weather/forecast"
    if any(kw in q for kw in ["wiki", "wikipedia", "who is", "what is"]):
        return "wikipedia", "matched keyword: wiki/who/what"
    return "llm", "no keyword match"

def _find_city(query: str) -> Optional[str]:
    q = query.lower()
    cities = [
        "san francisco", "new york",
        "chicago", "paris", "london", "mumbai", "tokyo", "delhi", "bangalore"
    ]
    for c in cities:
        if c in q:
            return c.title()
    return None

def _find_topic(query: str) -> str:
    q = query.strip()
    lower = q.lower()
    if "about" in lower:
        return q[lower.index("about") + len("about"):].strip() or "Agentic AI"
    if "on " in lower:
        return q[lower.index("on ") + len("on "):].strip() or "Agentic AI"
    return q

def _city_from_memories(mem_items) -> Optional[str]:
    city_list = {"san francisco","new york","chicago","paris","london","mumbai","tokyo","delhi","bangalore"}
    for it in mem_items:
        txt = it.text.lower()
        for c in city_list:
            if c in txt:
                return c.title()
    return None

def run_once(query: str) -> str:
    SM.add("user", query)

    if query.lower().startswith("remember "):
        ack = LM.remember(query, kind="preference", meta={"source":"user"})
        SM.add("assistant", ack)
        return ack

    if is_storable(query):
        LM.remember(query, kind="episodic", meta={"source":"conversation"})

    retrieved = LM.retrieve(query, k=3)
    tool, reason = choose_tool(query)

    if tool == "weather":
        city = _find_city(query)
        if not city:
            mem_city = _city_from_memories(retrieved)
            if mem_city:
                city = mem_city
        if not city:
            reply = "I can check the weather, but I couldn't spot the city."
            SM.add("assistant", reply)
            return reply
        out = call_with_policies(
            tool_name="weather",
            tool_fn=REGISTRY["weather"].call,
            args={"city": city},
            denylist={"delete database", "transfer all money", "format disk"},
            max_attempts=2,
            backoff_sec=0.2,
            enable_logging=True
        )
        reply = f"[weather] {out}  (reason: {reason}; used memory defaults: {'yes' if _find_city(query) is None else 'no'})"
        SM.add("assistant", reply)
        LM.remember(f"weather({city}) -> {out}", kind="episodic", meta={"source":"tool"})
        return reply

    if tool == "wikipedia":
        topic = _find_topic(query)
        out = call_with_policies(
            tool_name="wikipedia",
            tool_fn=REGISTRY["wikipedia"].call,
            args={"topic": topic},
            denylist={"delete database", "transfer all money", "format disk"},
            max_attempts=2,
            backoff_sec=0.2,
            enable_logging=True
        )
        reply = f"[wikipedia] {out}  (reason: {reason})"
        SM.add("assistant", reply)
        LM.remember(f"wiki({topic}) -> {out}", kind="episodic", meta={"source":"tool"})
        return reply

    reply = "(llm) I'd answer conversationally here. (reason: no matching tool)"
    SM.add("assistant", reply)
    return reply
