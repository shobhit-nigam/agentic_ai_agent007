from dataclasses import dataclass
from typing import Callable, Dict

@dataclass
class Tool:
    name: str
    description: str
    call: Callable[..., str]   # simple: kwargs in, string out

# Global tool catalog (name -> Tool)
REGISTRY: Dict[str, Tool] = {}

def register(tool: Tool) -> None:
    """Add/replace a tool by name (simple for Lab 1; we can harden later)."""
    REGISTRY[tool.name] = tool

# ---- Mock tools (offline-friendly) ----
def weather_tool(city: str) -> str:
    # Keep it deterministic for the first lab
    return f"(mock) Weather in {city}: 22°C, clear"

def wiki_tool(topic: str) -> str:
    return f"(mock) Quick facts about {topic}: ..."

# Register two tools so the agent can "act"
register(Tool(
    name="weather",
    description="Get current weather by city",
    call=lambda city: weather_tool(city)
))
register(Tool(
    name="wikipedia",
    description="Short summary of a topic",
    call=lambda topic: wiki_tool(topic)
))
