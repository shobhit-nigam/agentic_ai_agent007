# Lab 10: Memory + Multi-Tool Orchestration (LangGraph)
# -----------------------------------------------------
# This lab builds a *travel assistant* that:
#  - Remembers user preferences across turns (via LangGraph checkpointer + thread_id)
#  - Orchestrates multiple tools (weather, wiki, activities, calculator)
#  - Produces a clear final answer
#
# Setup:
#   pip install -U langgraph langchain langchain-openai typing_extensions
#   export OPENAI_API_KEY="your_key"
#
# Run:
#   python Lab10_Memory_MultiTool_LangGraph.py

from typing import Dict, Tuple
import re
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# Tools (single-input, deterministic)
# -----------------------------

@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """Return a minimal weather string for a city from a tiny offline DB."""
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

@tool("mini_wiki", return_direct=False)
def mini_wiki(topic: str) -> str:
    """Return a short fact from a tiny offline encyclopedia."""
    kb: Dict[str, str] = {
        "paris": "Paris is the capital of France, known for the Eiffel Tower and the Louvre.",
        "london": "London is the capital of the UK, home to the British Museum and the Thames.",
        "tokyo": "Tokyo blends tradition and technology; famous for Shibuya Crossing and Ueno Park.",
        "mumbai": "Mumbai is India's financial hub, known for Marine Drive and film industry.",
        "chicago": "Chicago sits on Lake Michigan; known for the Riverwalk and deep-dish pizza.",
    }
    return kb.get(topic.strip().lower(), "No entry found in mini_wiki. Try city names like 'Paris' or 'Chicago'.")

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
    """Return ONE indoor and ONE outdoor suggestion for a city.
    Input: SINGLE string like 'city=Paris; weather=sunny, 24°C' or just 'Paris'."""
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
        return "General: Indoor - local museum or aquarium. Outdoor - central park or riverfront walk."
    w = weather.lower()
    indoor_first = any(k in w for k in ["rain", "storm"]) or ("overcast" in w and "cold" in w)
    if indoor_first:
        indoor = data["indoor"][0]; outdoor = data["outdoor"][0]
    elif any(k in w for k in ["sunny", "clear"]):
        outdoor = data["outdoor"][0]; indoor = data["indoor"][0]
    else:
        indoor = data["indoor"][0]; outdoor = data["outdoor"][0]
    return f"City: {city}. Indoor: {indoor}. Outdoor: {outdoor}. (Weather-aware heuristics.)"

@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """Evaluate simple math using only digits and + - * / ( ) . %"""
    import math, re
    if not re.fullmatch(r"[0-9+\-*/(). %\s]+", expression):
        return "Calculator error: invalid characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"

TOOLS = [get_weather, mini_wiki, suggest_city_activities, calculator]

# -----------------------------
# Agent state & graph
# -----------------------------

class AgentState(TypedDict):
    # Accumulate messages across nodes in order (prevents tool role error)
    messages: Annotated[list[AnyMessage], add]

# The LLM with tool binding (structured tool-calls)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

def agent_node(state: AgentState) -> AgentState:
    """One LLM step. If tool calls are returned, graph will route to tools."""
    # System policy (inline, could also be a SystemMessage in messages):
    # - You are a travel assistant that remembers user preferences from prior turns in this thread.
    # - Prefer to get weather before activity suggestions if weather not known.
    # - When planning, respect user preferences (e.g., indoor/outdoor, budget hints).
    # - Use mini_wiki for a 1-sentence city fact if helpful.
    # - Finish the final response with 'Final Answer:' and a concise plan.
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

# Custom router: if the assistant asked for tools → go to tools; else END
def route_from_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    # AIMessage in LangChain exposes tool calls on .tool_calls or .additional_kwargs
    tc = getattr(last, "tool_calls", None) or getattr(getattr(last, "additional_kwargs", {}), "get", lambda *_: None)("tool_calls")
    if tc:
        return "tools"
    return END

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_from_agent)
graph.add_edge("tools", "agent")

# Persistence: per-thread memory
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# -----------------------------
# Demo helpers
# -----------------------------

def run_conversation():
    print("\n=== Lab 10: Memory + Multi-Tool Orchestration ===")
    cfg = {"configurable": {"thread_id": "t-trip"}}
    msgs = [
        HumanMessage(content="Hi, I'm Priya. I prefer indoor activities and budget-friendly options."),
        HumanMessage(content="Plan a 1-day trip in Paris considering the weather; include one indoor and one outdoor pick."),
        HumanMessage(content="What did I say my preference was?"),
        HumanMessage(content="Now do the same for Chicago."),
        HumanMessage(content="Quickly compute estimated tickets 25 + 18 + 12.5 and tell me only the number."),
    ]
    for m in msgs:
        out = app.invoke({"messages": [m]}, cfg)
        final_msg = out["messages"][-1]
        print("\nUser:", m.content)
        print("Assistant:", getattr(final_msg, "content", final_msg))

if __name__ == "__main__":
    run_conversation()
