# Lab 11 (or Lab 10): Multi-Agent Systems — Planner + Executor (LangGraph)
# -----------------------------------------------------------------------
# Concepts:
# - PLANNER (no tools): emits STRICT JSON steps or a final answer
# - EXECUTOR (has tools): executes the step and returns EXEC_RESULT
# - Centralized orchestration via a LangGraph state machine.
# - Persistent memory via checkpointer + thread_id.
#
# Setup:
#   pip install -U langgraph langchain langchain-openai typing_extensions
#   export OPENAI_API_KEY="your_key"
#
# Run:
#   python Lab11.py

from typing import Dict, Tuple, Optional
import re, json

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ---------- Tools (reused from earlier labs) ----------

@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """Return a minimal weather string for the given city using a tiny offline map.
    Input: `city` (str), e.g. "Paris".
    Output: A one-line weather string (e.g., "The weather in Paris is sunny, 24°C.")
    """
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
    """Return a one-sentence fact for a known city from a tiny offline knowledge base.
    Input: `topic` (str), e.g. "Paris".
    Output: A single-sentence description or a fallback message if topic not found.
    """
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
    """Recommend ONE indoor and ONE outdoor activity for a city.
    Input: single string, either "Paris" or "city=Paris; weather=sunny, 24°C".
    Output: One line with formatted picks and a short heuristic note.
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
    """Evaluate a simple math expression containing only digits, whitespace, and + - * / ( ) . %.
    Input: `expression` (str), e.g., "25 + 18 + 12.5".
    Output: The numeric result as a string, or an error message for invalid input.
    """
    import math, re
    if not re.fullmatch(r"[0-9+\-*/(). %\s]+", expression):
        return "Calculator error: invalid characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"Calculator error: {e}"

TOOLS = [get_weather, mini_wiki, suggest_city_activities, calculator]

# ---------- State ----------
class MASState(TypedDict):
    messages: Annotated[list[AnyMessage], add]   # accumulate across nodes
    plan: Optional[dict]                         # latest planner JSON
    exec_result: Optional[str]                   # latest executor "EXEC_RESULT: ..."

# ---------- LLMs ----------
planner_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm_with_tools = executor_llm.bind_tools(TOOLS)

PLANNER_SYS = (
    "You are PLANNER. You DO NOT call tools.\n"
    "On each turn, output STRICT JSON ONLY, no extra text:\n"
    "  If more work needed:\n"
    '    {"done": false, "next_step": {"tool": "get_weather"|"suggest_city_activities"|"mini_wiki"|"calculator", "input": "<string>"}, "rationale": "<short>"}\n'
    "  If finished:\n"
    '    {"done": true, "final_answer": "<concise final answer>"}\n'
)

EXECUTOR_SYS = (
    "You are EXECUTOR. You MUST execute the given step using the available tools.\n"
    "If a tool name is given, call exactly that tool with the provided input.\n"
    "If the last message is a tool result, DO NOT call tools again; instead return an EXEC_RESULT summarizing it.\n"
    "After tool execution (or summarizing the latest tool result), respond with ONLY one line:\n"
    "EXEC_RESULT: <concise result>\n"
)

def _extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}\s*$", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

# ---------- Nodes ----------
def planner_node(state: MASState) -> MASState:
    msgs = state["messages"]
    if not msgs or msgs[0].type != "system":
        msgs = [SystemMessage(content=PLANNER_SYS)] + msgs
    response = planner_llm.invoke(msgs)
    plan = _extract_json(response.content) or {}
    return {"messages": [response], "plan": plan}

def executor_node(state: MASState) -> MASState:
    msgs = list(state["messages"])
    last = msgs[-1]
    # If last is a tool result, ask executor to summarize (no more tool calls).
    if isinstance(last, ToolMessage):
        exec_msgs = msgs + [
            SystemMessage(content=EXECUTOR_SYS),
            HumanMessage(content="Summarize the latest tool result and return only 'EXEC_RESULT: <one line>'")
        ]
    else:
        step = (state.get("plan") or {}).get("next_step", {})
        exec_msgs = msgs + [
            SystemMessage(content=EXECUTOR_SYS),
            HumanMessage(content=f"Step to execute: {json.dumps(step)}")
        ]
    response = executor_llm_with_tools.invoke(exec_msgs)
    new_state: MASState = {"messages": [response]}
    # If no tool_calls, capture EXEC_RESULT
    tool_calls = getattr(response, "tool_calls", None) or (getattr(response, "additional_kwargs", {}) or {}).get("tool_calls")
    if not tool_calls:
        m = re.search(r"EXEC_RESULT:\s*(.+)\s*$", (response.content or "").strip())
        if m:
            new_state["exec_result"] = m.group(1).strip()
    return new_state

# Tools node (prebuilt)
tools_node = ToolNode(TOOLS)

# ---------- Graph wiring ----------
graph = StateGraph(MASState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("tools", tools_node)

def route_from_planner(state: MASState):
    plan = state.get("plan") or {}
    return END if plan.get("done") else "executor"

def route_from_executor(state: MASState):
    last = state["messages"][-1]
    tc = getattr(last, "tool_calls", None) or (getattr(last, "additional_kwargs", {}) or {}).get("tool_calls")
    return "tools" if tc else "planner"

graph.set_entry_point("planner")
graph.add_conditional_edges("planner", route_from_planner)   # planner → executor or END
graph.add_conditional_edges("executor", route_from_executor) # executor → tools or planner
graph.add_edge("tools", "executor")                          # tools → executor

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ---------- Demo ----------
if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "mas-thread"}}
    user_task = (
        "Plan a short evening in Paris: first get the current weather, "
        "then pick one indoor and one outdoor activity, include a one-sentence city fact, "
        "and then finalize the plan."
    )
    out = app.invoke({"messages": [HumanMessage(content=user_task)]}, cfg)

    # Print the final planner message (should contain the final_answer JSON)
    final_msg = out["messages"][-1]
    print("\n=== FINAL (Planner) ===")
    print(getattr(final_msg, "content", final_msg))
