import os
import json
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
from urllib.parse import urlparse, quote
import requests


# ---------------------------
# Tool registry
# ---------------------------

@dataclass
class Tool:
    name: str
    description: str
    # The callable should accept keyword args (we invoke with **args)
    call: Callable[..., str]


REGISTRY: Dict[str, Tool] = {}


def register(tool: Tool) -> None:
    REGISTRY[tool.name] = tool


# ---------------------------
# Mock tools (offline-friendly)
# ---------------------------

def weather_tool(city: str) -> str:
    return f"(mock) Weather in {city}: 22°C, clear"


def wiki_tool(topic: str) -> str:
    return f"(mock) Quick facts about {topic}: ..."


# Register mock tools
register(Tool(
    name="weather",
    description="Mock: weather by city",
    call=lambda city: weather_tool(city),
))
register(Tool(
    name="wikipedia",
    description="Mock: summary of a topic",
    call=lambda topic: wiki_tool(topic),
))


# ---------------------------
# Part B: Live fetch tools
# ---------------------------

ALLOWED_HOSTS = {"api.open-meteo.com", "en.wikipedia.org"}

CITY_TO_COORDS: Dict[str, tuple[float, float]] = {
    "Chicago": (41.8819, -87.6278),
    "Paris": (48.8566, 2.3522),
    "London": (51.5074, -0.1278),
    "Mumbai": (19.0760, 72.8777),
    "Tokyo": (35.6762, 139.6503),
    "Delhi": (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "San Francisco": (37.7749, -122.4194),
    "New York": (40.7128, -74.0060),
}


def http_get_json(url: str, *, timeout: float = 8.0) -> Dict[str, Any]:
    host = urlparse(url).netloc
    if host not in ALLOWED_HOSTS:
        return {"error": f"Host not allowed: {host}"}
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "lab4-agent/1.0"})
        if r.status_code >= 400:
            return {"error": f"http {r.status_code}", "body": r.text[:800]}
        ctype = r.headers.get("Content-Type", "")
        if "application/json" in ctype or url.endswith(".json"):
            return r.json()
        # Try JSON anyway; else return a text snippet
        try:
            return r.json()
        except Exception:
            return {"text": r.text[:1000]}
    except Exception as e:
        return {"error": str(e)}


def live_weather(city: str) -> str:
    coords = CITY_TO_COORDS.get(city)
    if not coords:
        return f"Sorry, I don't have coordinates for {city}."
    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    )
    data = http_get_json(url)
    if "error" in data:
        body = data.get("body", "")[:120]
        return f"(live weather failed: {data['error']} {body}) Falling back to mock: {weather_tool(city)}"
    cw = data.get("current_weather") or {}
    temp = cw.get("temperature")
    wind = cw.get("windspeed")
    code = cw.get("weathercode")
    if temp is None:
        return f"(live weather unavailable) {weather_tool(city)}"
    return f"(live) Weather in {city}: {temp}°C, wind {wind} km/h, code {code}"


def wiki_summary(topic: str) -> str:
    title = quote(topic.strip().replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    data = http_get_json(url)
    if "error" in data:
        body = data.get("body", "")[:120]
        return f"(live wiki failed: {data['error']} {body}) {wiki_tool(topic)}"
    extract = data.get("extract")
    if not extract:
        return f"(live wiki: no extract) {wiki_tool(topic)}"
    return f"(live) {extract[:800]}"


# Register live tools
register(Tool(
    name="live_weather",
    description="Live weather via Open-Meteo (no API key)",
    call=lambda city: live_weather(city),
))
register(Tool(
    name="wiki_summary",
    description="Live Wikipedia summary",
    call=lambda topic: wiki_summary(topic),
))


# ---------------------------
# Part C: OpenAI LLM tool
# ---------------------------

def openai_chat(prompt: str, *, model: Optional[str] = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"(mock LLM; set OPENAI_API_KEY to use OpenAI) {prompt[:200]}..."

    endpoint = "https://api.openai.com/v1/chat/completions"
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    try:
        r = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=25)
        if r.status_code >= 400:
            body = r.text[:600].replace("\n", " ")
            return f"(openai http {r.status_code}) {body}"
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(openai error: {e}) {prompt[:200]}..."


def _merge_prompt_with_upstream(prompt: str, upstream: Optional[str]) -> str:
    if not upstream:
        return prompt
    return f"""{prompt}

Context (from previous step):
{upstream}
"""


# Register OpenAI tool; accept **kwargs so 'upstream' won't crash
register(Tool(
    name="llm_openai",
    description="LLM via OpenAI Chat Completions (needs OPENAI_API_KEY)",
    call=lambda **kw: openai_chat(
        prompt=_merge_prompt_with_upstream(kw.get("prompt", ""), kw.get("upstream")),
        model=kw.get("model"),
    ),
))
