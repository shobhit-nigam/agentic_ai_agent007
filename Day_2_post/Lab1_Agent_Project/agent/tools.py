from dataclasses import dataclass
from typing import Callable, Dict

@dataclass 
class Tool:
    name: str
    description: str
    call: Callable[..., str]  # kwargs in, string out 

REGISTRY : Dict[str, Tool] = {}

# local mock tools (offline-friendly)
def register(tool: Tool) -> None:
    REGISTRY[tool.name] = tool

def weather_tool(city: str) -> str:
    return f"weather in {city}: 22 degrees, clear"

def wiki_tool(topic: str) -> str:
    return f"(mock) Quick facts about {topic}: ..."

#register the weather tool
register(
    Tool(
        name = "weather", 
        description="Get current weather by city", 
        call = lambda city: weather_tool(city)
    )
)
#register the wiki tool
register(
    Tool(
        name = "wikipedia", 
        description="Short summary of a topic", 
        call = lambda topic: wiki_tool(topic)
    )
)