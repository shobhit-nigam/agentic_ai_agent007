# Lab 9 (FIXED): LangGraph ReAct Agent with Tools & Persistence
# -------------------------------------------------------------
# This version fixes the state accumulator so OpenAI tool-calls work reliably:
#   AgentState.messages: Annotated[list[AnyMessage], add]
#
# Setup:
#   pip install -U langgraph langchain langchain-openai typing_extensions
#   export OPENAI_API_KEY="your_key"
#
# Run:
#   python Lab9_LangGraph_ReAct.py

from typing import Dict, Tuple
import re

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# Tools (offline, deterministic)
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

@tool("calculator", return_direct=True)
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
    Input: SINGLE string like: "city=Paris; weather=sunny, 24°C" or just "Paris".
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

TOOLS = [get_weather, calculator, suggest_city_activities]

# -----------------------------
# Agent state & graph
# -----------------------------

class AgentState(TypedDict):
    # ACCUMULATE messages across nodes in order (prevents tool role error)
    messages: Annotated[list[AnyMessage], add]

# The LLM with tool binding (structured tool-calls)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

def agent_node(state: AgentState) -> AgentState:
    """Call the chat model. If tool calls are returned, the router will send us to Tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)  # routes to tools or END
graph.add_edge("tools", "agent")                     # loop back after tools

# Persistence: in-memory checkpointer (swap with SQLite/Redis for prod)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# -----------------------------
# Demo helpers
# -----------------------------

def run_single_turn():
    print("\n=== Single-turn demo ===")
    cfg = {"configurable": {"thread_id": "t-single"}}
    out = app.invoke({"messages": [HumanMessage(content="Plan an evening in Chicago with one indoor and one outdoor, based on the weather.")]}, cfg)
    final_msg = out["messages"][-1]
    print("Assistant:", getattr(final_msg, "content", final_msg))

def run_multi_turn_with_persistence():
    print("\n=== Multi-turn demo with persistence (threaded) ===")
    cfg = {"configurable": {"thread_id": "t-123"}}  # same thread reuses memory
    msgs = [
        HumanMessage(content="Hi, my name is Priya."),
        HumanMessage(content="Please plan a relaxed evening for me in Paris. Remember my name."),
        HumanMessage(content="What was my name? Also suggest one indoor and one outdoor activity."),
    ]
    for m in msgs:
        out = app.invoke({"messages": [m]}, cfg)
        final_msg = out["messages"][-1]
        print("\nUser:", m.content)
        print("Assistant:", getattr(final_msg, "content", final_msg))

def run_math_tool():
    print("\n=== Tool routing for math (calculator) ===")
    cfg = {"configurable": {"thread_id": "t-math"}}
    out = app.invoke({"messages": [HumanMessage(content="Compute 23*17 + 3.5 and return only the number.")]}, cfg)
    final_msg = out["messages"][-1]
    print("Assistant:", getattr(final_msg, "content", final_msg))

if __name__ == "__main__":
    run_single_turn()
    run_multi_turn_with_persistence()
    run_math_tool()
