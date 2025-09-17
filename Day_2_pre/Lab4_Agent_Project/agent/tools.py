from dataclasses import dataclass
from typing import Callable, Dict

@dataclass
class Tool:
    name: str
    description: str
    call: Callable[..., str]   # kwargs in, string out

REGISTRY: Dict[str, Tool] = {}

def register(tool: Tool) -> None:
    REGISTRY[tool.name] = tool

# Mock tools
def weather_tool(city: str) -> str:
    return f"(mock) Weather in {city}: 22°C, clear"

def wiki_tool(topic: str) -> str:
    return f"(mock) Quick facts about {topic}: ..."

register(Tool("weather", "Get current weather by city", lambda city: weather_tool(city)))
register(Tool("wikipedia", "Short summary of a topic", lambda topic: wiki_tool(topic)))
