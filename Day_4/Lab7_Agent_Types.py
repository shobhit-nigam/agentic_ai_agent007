# Lab 7: Agent Types — Tool-Calling, ReAct, and Conversational (with Memory)
# --------------------------------------------------------------------------
# Builds on Lab 5 and Lab 6.
# Demonstrates three agent styles side-by-side:
#   1) Tool-Calling agent (OpenAI function-calling)
#   2) ReAct agent (Zero-Shot ReAct Description)
#   3) Conversational ReAct agent with Memory
#
# Notes:
# - ReAct expects single-input tools (text in, text out). We design tools accordingly.
# - We enable handle_parsing_errors=True to make demos robust.
# - For newer projects, LangChain recommends LangGraph. We'll stick to Agents here for learning continuity.
#
# Setup:
#   pip install langchain openai
#   export OPENAI_API_KEY="your_key"   # mac/linux
#   setx OPENAI_API_KEY "your_key"     # windows powershell
#
# Run:
#   python Lab7_Agent_Types.py

from typing import Dict
import re
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory


# (Optional) Silence LangChain deprecation warnings in this learning lab
try:
    from langchain_core._api import LangChainDeprecationWarning
    import warnings as _warnings
    _warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass


# -----------------------------
# Tools (single-input where required for ReAct safety)
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
    c = city.strip().strip("'\"`").lower()
    if c in weather_db:
        return f"The weather in {city} is {weather_db[c]}."
    return f"No weather found for '{city}'. Assume mild (22°C) and clear for demo."


@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """Safely evaluate a simple math expression. Accepts raw text like
    '23*17 + 3.5' or lines such as 'expression = "23*17 + 3.5"'."""
    import math, re

    s = expression.strip()
    # Accept patterns like: expression = "...", expr: ..., calc="..."
    m_eq = re.match(r"^[A-Za-z_][\w\-]*\s*(?:=|:)\s*(.*)$", s)
    if m_eq:
        s = m_eq.group(1).strip()
    # Strip surrounding quotes/backticks if present
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in "'\"`"):
        s = s[1:-1].strip()

    # Basic validation: digits, ops, spaces, parentheses, decimal dot, percent
    if not re.fullmatch(r"[0-9+\-*/(). %\s]+", s):
        return "Calculator error: invalid characters."
    try:
        # Evaluate in a safe namespace (only math allowed)
        result = eval(s, {"__builtins__": {}}, {"math": math})
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

def _parse_city_weather(query: str):
    """Extract city and optional weather from a single string.
    Accepts: "Paris" OR "city=Paris" OR "city=Paris; weather=sunny, 24°C"
    Returns: (city, weather) with weather possibly ''.
    """
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
    Input: SINGLE string. Examples:
      - "Paris"
      - "city=Paris"
      - "city=Paris; weather=sunny, 24°C"
    """
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
    c = city.strip().strip("'\"`").lower()
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
# LLM and Tool Registry
# -----------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # deterministic for stable parsing
tools = [get_weather, calculator, mini_wiki, suggest_city_activities]

# -----------------------------
# 1) Tool-Calling Agent
# -----------------------------
tool_calling_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "system_message": (
            "You are a helpful assistant. Use tools when relevant. "
            "Finish with 'Final Answer:' and a concise result."
        )
    }
)

# -----------------------------
# 2) ReAct Agent (Zero-Shot)
# -----------------------------
react_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": (
            "You are a helpful assistant. Think step-by-step, then use tools. "
            "Use ONLY these section headers: Thought:, Action:, Action Input:, Observation:. "
            "End your final response with a line starting 'Final Answer:'."
        )
    }
)

# -----------------------------
# 3) Conversational Agent with Memory
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conversational_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    agent_kwargs={
        "prefix": (
            "You are a friendly assistant that remembers prior messages in this chat. "
            "Use tools as needed. End your final response with 'Final Answer:'."
        )
    }
)

# -----------------------------
# Demos
# -----------------------------

def demo_tool_calling():
    print("\n=== Demo 1: Tool-Calling Agent ===\n")
    q = "What's the weather in Tokyo and suggest one indoor and one outdoor thing to do this evening."
    print("User:", q)
    r = tool_calling_agent.invoke({"input": q})
    print("Assistant:", r["output"])

def demo_react():
    print("\n=== Demo 2: ReAct Agent ===\n")
    q = "Calculate: 23*17 + 3.5, then check mini_wiki for 'LangChain' and summarize both in one line."
    print("User:", q)
    r = react_agent.invoke({"input": q})
    print("Assistant:", r["output"])

def demo_conversational_memory():
    print("\n=== Demo 3: Conversational Agent with Memory ===\n")
    turns = [
        "Hi, my name is Shobhit",
        "Plan a relaxed evening for me in Paris. Remember my name.",
        "What was my name? Also suggest one indoor and one outdoor activity."
    ]
    for t in turns:
        print("\nUser:", t)
        r = conversational_agent.invoke({"input": t})
        print("Assistant:", r["output"])

if __name__ == "__main__":
    demo_tool_calling()
    demo_react()
    demo_conversational_memory()
