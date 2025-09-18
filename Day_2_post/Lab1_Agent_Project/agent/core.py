from typing import Tuple, Optional
from agent.tools import REGISTRY

# --- 1) Router: pick a tool name from a query (tiny, explicit) ---
def choose_tool(query: str) -> Tuple[str, str]:
    q = query.lower()
    if "weather" in q or "forecast" in q:
        return "weather", "matched keyword: weather/forecast"
    if any(kw in q for kw in ["wiki", "wikipedia", "who is", "what is"]):
        return "wikipedia", "matched keyword: wiki/who/what"
    return "llm", "no keyword match"

# --- 2) Naive argument extraction (kept simple for Lab 1) ---
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

# --- 3) Orchestrator: route -> extract args -> call tool (or fallback) ---
def run_once(query: str) -> str:
    tool, reason = choose_tool(query)

    if tool == "weather":
        city = _find_city(query)
        if not city:
            return "I can check the weather, but I couldn't spot the city. Try: 'weather in Paris'."
        out = REGISTRY["weather"].call(city=city)  
        return f"[weather] {out}  (reason: {reason})"

    if tool == "wikipedia":
        topic = _find_topic(query)
        out = REGISTRY["wikipedia"].call(topic=topic)  
        return f"[wikipedia] {out}  (reason: {reason})"

    return "(llm) I'd answer conversationally here. (reason: no matching tool)"

