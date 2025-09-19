# Lab 8 (v2, FIXED): Prompt Engineering for Agent Decision-Making
# ---------------------------------------------------------------
# Focus: how SYSTEM prompts shape agent behavior (tool routing, format, cost).
#
# Variants on a Tool-Calling agent (OpenAI function-calling):
#   A) Baseline helper
#   B) JSON Policy (structured output + tools_used)
#   C) Cost-Aware (avoid unnecessary tool calls)
#   D) Math Policy (force calculator, error contract)  <-- braces escaped
#   E) Forbid Wiki (disallow a specific tool)
#
# Setup:
#   pip install langchain openai
#   export OPENAI_API_KEY="your_key"   # mac/linux
#   setx OPENAI_API_KEY "your_key"     # windows powershell
#
# Run:
#   python Lab8_Prompt_Engineering_for_Agents_v2_FIXED.py

from typing import Dict, List, Tuple
import re, json
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

# -----------------------------
# Tools (offline & deterministic)
# -----------------------------

@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """Return simple weather for a given city from a tiny offline database."""
    weather_db: Dict[str, str] = {
        "paris": "sunny, 24°C",
        "chicago": "cloudy, 18°C",
        "mumbai": "rainy, 30°C",
        "london": "overcast, 17°C",
        "tokyo": "clear, 26°C",
    }
    c = city.strip().lower()
    if c in weather_db:
        return f"The weather in {city} is {weather_db[c]}."
    return f"No weather found for '{city}'. Assume mild (22°C) and clear for demo."

@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """Safely evaluate a simple math expression using digits and + - * / ( ) . % ."""
    import math, re
    if not re.fullmatch(r"[0-9+\-*/(). %\s]+", expression):
        return "Calculator error: invalid characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"

@tool("mini_wiki", return_direct=False)
def mini_wiki(topic: str) -> str:
    """Return a short summary from a tiny offline encyclopedia."""
    kb: Dict[str, str] = {
        "alan turing": "Alan Turing (1912–1954) was a mathematician and pioneer of computer science who formalized computation and contributed to codebreaking in WWII.",
        "agentic ai": "Agentic AI refers to systems that can plan, choose tools/actions, and adapt using feedback, rather than only producing text responses.",
        "langchain": "LangChain is a framework that provides building blocks for LLM apps: prompts, chains, tools, memory, and agents.",
    }
    return kb.get(topic.strip().lower(), "No entry found in mini_wiki. Try 'Alan Turing', 'Agentic AI', or 'LangChain'.")

def _parse_city_weather(query: str) -> Tuple[str, str]:
    q = query.strip()
    if ";" in q or "city=" in q.lower():
        parts = [p.strip() for p in q.split(";")]
        city, weather = "", ""
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip().lower(); v = v.strip()
                if k == "city":
                    city = v
                elif k == "weather":
                    weather = v
        if not city:
            city = re.split(r"[;,]", q)[0].replace("city", "").replace("=", "").strip()
        return city, weather
    return q, ""

@tool("suggest_city_activities", return_direct=False)
def suggest_city_activities(query: str) -> str:
    """Recommend ONE indoor and ONE outdoor activity.
    Input: SINGLE string, e.g., "city=Paris; weather=sunny, 24°C" or just "Paris"""
    catalog = {
        "chicago": {
            "indoor": ["Art Institute of Chicago", "Museum of Science and Industry", "Field Museum"],
            "outdoor": ["Chicago Riverwalk", "Millennium Park", "Navy Pier"],
        },
        "paris": {
            "indoor": ["Louvre Museum", "Musée d'Orsay"],
            "outdoor": ["Seine River Walk", "Jardin du Luxembourg"],
        },
        "london": {
            "indoor": ["British Museum", "Tate Modern"],
            "outdoor": ["Hyde Park", "South Bank Walk"],
        },
        "tokyo": {
            "indoor": ["teamLab Planets", "Tokyo National Museum"],
            "outdoor": ["Ueno Park", "Shibuya Crossing Walk"],
        },
        "mumbai": {
            "indoor": ["Chhatrapati Shivaji Maharaj Vastu Sangrahalaya", "Phoenix Mall"],
            "outdoor": ["Marine Drive", "Sanjay Gandhi National Park"],
        },
    }
    city, weather = _parse_city_weather(query)
    c = city.strip().lower()
    if not c:
        return "Please provide a city name (e.g., 'city=Paris; weather=sunny, 24°C')."
    data = catalog.get(c)
    if not data:
        return ("General suggestions (city not in catalog): Indoor - visit a local museum or aquarium. "
                "Outdoor - take a riverfront/park walk if conditions allow.")
    w = weather.lower()
    indoor_first = any(k in w for k in ["rain", "storm"]) or ("overcast" in w and "cold" in w)
    if indoor_first:
        indoor = data["indoor"][0]; outdoor = data["outdoor"][0]
    elif any(k in w for k in ["sunny", "clear"]):
        outdoor = data["outdoor"][0]; indoor = data["indoor"][0]
    else:
        indoor = data["indoor"][0]; outdoor = data["outdoor"][0]
    return f"City: {city}. Indoor: {indoor}. Outdoor: {outdoor}. (Weather-aware heuristics.)"

# -----------------------------
# LLM + Agent factory
# -----------------------------
def make_llm(temp: float = 0.0):
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=temp)

def make_tool_calling_agent(system_message: str, tools, temp: float = 0.0):
    llm = make_llm(temp)
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_message},
    )

# -----------------------------
# System prompt variants (policies)
# -----------------------------
SYS_BASELINE = (
    "You are a helpful assistant. Use tools when relevant. "
    "Finish the final response with the token 'Final Answer:' followed by your concise answer."
)

SYS_JSON_POLICY = (
    "You are a structured assistant. When answering, you MUST:\n"
    "1) Call tools when needed (get_weather first if weather is unknown; then suggest_city_activities).\n"
    "2) Output ONLY a JSON object after 'Final Answer:' with keys: "
    "city, weather, indoor, outdoor, tools_used (array of tool names).\n"
    "3) Do not include extra prose outside the JSON after 'Final Answer:'."
)

SYS_COST_AWARE = (
    "You are a cost-aware assistant. Prefer NOT to call tools unless strictly necessary. "
    "If you can answer confidently without a tool, do so. If the city is unknown, say 'Unknown'. "
    "Finish with 'Final Answer:'."
)

# IMPORTANT: braces escaped with double {{ }}
SYS_MATH_POLICY = (
    "You are a math-safe assistant. For ANY arithmetic that is not trivial, "
    "you MUST call the 'calculator' tool. If the calculator errors, return 'Final Answer:' with "
    "an object {{'error':'calculator_failed'}}. Keep answers concise."
)

SYS_FORBID_WIKI = (
    "You are a concise assistant. You MUST NOT call the 'mini_wiki' tool. "
    "If asked to summarize concepts, respond directly without tools. Finish with 'Final Answer:'."
)

# -----------------------------
# Build agents with different policies
# -----------------------------
TOOLS_ALL = [get_weather, calculator, mini_wiki, suggest_city_activities]
TOOLS_NO_WIKI = [get_weather, calculator, suggest_city_activities]

agent_baseline = make_tool_calling_agent(SYS_BASELINE, TOOLS_ALL, temp=0.0)
agent_json = make_tool_calling_agent(SYS_JSON_POLICY, TOOLS_ALL, temp=0.0)
agent_cost = make_tool_calling_agent(SYS_COST_AWARE, TOOLS_ALL, temp=0.0)
agent_math = make_tool_calling_agent(SYS_MATH_POLICY, TOOLS_ALL, temp=0.0)
agent_no_wiki = make_tool_calling_agent(SYS_FORBID_WIKI, TOOLS_NO_WIKI, temp=0.0)

# -----------------------------
# Demo runner + JSON extractor
# -----------------------------
def run(agent, name: str, query: str):
    print(f"\n===== {name} =====")
    print("User:", query)
    out = agent.invoke({"input": query})
    print("Assistant:", out["output"])
    return out["output"]

def extract_json_after_final_answer(text: str):
    if "Final Answer:" not in text:
        return None
    after = text.split("Final Answer:", 1)[1].strip()
    try:
        return json.loads(after)
    except Exception:
        m = re.search(r"\{.*\}\s*$", after, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def main():
    # A) Baseline
    run(agent_baseline, "A) Baseline", "Plan an evening in London with one indoor and one outdoor option.")

    # B) JSON Policy
    out_b = run(agent_json, "B) JSON Policy", "Plan an evening in Chicago based on the weather.")
    parsed = extract_json_after_final_answer(out_b)
    print("\n[Parsed JSON]", parsed)

    # C) Cost-Aware — city not in our catalog
    run(agent_cost, "C) Cost-Aware", "Suggest evening activities in Zurich (indoor+outdoor). Keep it brief." )

    # D) Math Policy
    run(agent_math, "D) Math Policy", "Compute 19*7 + 2.5. Provide only the numeric result.")

    # E) Forbid Wiki
    run(agent_no_wiki, "E) Forbid Wiki", "What is LangChain in one sentence?")

if __name__ == "__main__":
    main()
