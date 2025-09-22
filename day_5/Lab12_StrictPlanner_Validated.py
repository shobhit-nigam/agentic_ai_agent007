
# Lab 12 (Strict Planner) — v2: Fix validator to detect calculator completion (prevents recursion loops)
from typing import Dict, Tuple, Optional
import re, json, os

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ---------- Tools ----------

@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """Return a minimal weather string for the given city using a tiny offline map.
    Input: `city` (str), e.g. "Paris". Output: A one-line weather string.
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
    Input: `topic` (str), e.g. "Paris". Output: One sentence or fallback message.
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
    Input: "Paris" or "city=Paris; weather=sunny, 24°C". Output: one line with picks.
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
    Input: `expression` (str), e.g., "25 + 18 + 12.5". Output: numeric result as string.
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

class MemState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    plan: Optional[dict]
    exec_result: Optional[str]
    summary: Optional[str]
    turn_count: int
    user_profile: Optional[dict]

# ---------- LLMs ----------

planner_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm_with_tools = executor_llm.bind_tools(TOOLS)
summ_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ---------- Policies ----------

PLANNER_SYS = (
    "You are PLANNER. You DO NOT call tools.\n"
    "You MUST NOT set done=true until these steps are executed in order for the CURRENT CITY:\n"
    "  1) get_weather(city)\n"
    "  2) suggest_city_activities(\"city=<CITY>; weather=<WEATHER_SNIPPET>\" if weather is known, else \"<CITY>\")\n"
    "If the latest user message contains a math expression, require step 3:\n"
    "  3) calculator(\"<EXPRESSION>\")\n"
    "Output STRICT JSON ONLY, no extra text:\n"
    "  {\"done\": false, \"next_step\": {\"tool\": \"get_weather\"|\"suggest_city_activities\"|\"calculator\", \"input\": \"<string>\"}, \"rationale\": \"<short>\"}\n"
    "  {\"done\": true, \"final_answer\": \"<concise final answer>\"}\n"
    "Always use the latest city from the user's most recent message; do not reuse older cities.\n"
)

EXECUTOR_SYS = (
    "You are EXECUTOR. You MUST execute the given step using the available tools.\n"
    "If a tool name is given, call exactly that tool with the provided input.\n"
    "If the last message is a tool result, DO NOT call tools again; instead return an EXEC_RESULT summarizing it.\n"
    "After tool execution (or summarizing the latest tool result), respond with ONLY one line:\n"
    "EXEC_RESULT: <concise result>\n"
)

# ---------- Helpers ----------

KNOWN_CITIES = {"paris", "chicago", "london", "tokyo", "mumbai"}

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

def latest_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""

def extract_latest_city(msg: str) -> Optional[str]:
    last = None
    low = msg.lower()
    for c in KNOWN_CITIES:
        if c in low:
            last = c
    m = re.search(r"\bin\s+([A-Z][a-z]+)\b", msg)
    if m:
        cand = m.group(1).lower()
        if cand in KNOWN_CITIES:
            last = cand
    return last

def has_tool_result_for_city(messages: list[AnyMessage], tool_name: str, city: str) -> bool:
    city_low = city.lower()
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == tool_name:
            if city_low in (m.content or "").lower():
                return True
        if isinstance(m, HumanMessage):
            break
    return False

def extract_weather_snippet(messages: list[AnyMessage], city: str) -> Optional[str]:
    city_low = city.lower()
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "get_weather":
            txt = (m.content or "")
            if city_low in txt.lower():
                mm = re.search(r"is\s+([^\.]+)\.", txt)
                if mm:
                    return mm.group(1).strip()
        if isinstance(m, HumanMessage):
            break
    return None

def math_expr_in_text(msg: str) -> Optional[str]:
    m = re.search(r"([0-9][0-9\.\s\+\-\*/\(\)%]+)", msg)
    return m.group(1).strip() if m else None

def has_calculator_result(messages: list[AnyMessage]) -> bool:
    """Detect if calculator tool ran since the last human turn (no city needed)."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "calculator":
            return True
        if isinstance(m, HumanMessage):
            break
    return False

# ---------- Nodes ----------

def planner_node(state: MemState) -> MemState:
    msgs = state["messages"]
    sys_msgs = []
    if not msgs or msgs[0].type != "system":
        sys_msgs.append(SystemMessage(content=PLANNER_SYS))
        profile = state.get("user_profile") or {}
        if profile:
            prefs = profile.get("preferences", {})
            sys_msgs.append(SystemMessage(content=f"User profile: name={profile.get('name','User')}. Preferences={prefs}"))
        if state.get("summary"):
            sys_msgs.append(SystemMessage(content=f"Conversation summary so far:\n{state['summary']}"))
    response = planner_llm.invoke(sys_msgs + msgs)
    plan = _extract_json(response.content) or {}
    turn = int(state.get("turn_count", 0)) + 1
    return {"messages": [response], "plan": plan, "turn_count": turn}

def executor_node(state: MemState) -> MemState:
    msgs = list(state["messages"])
    last = msgs[-1]
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
    new_state: MemState = {"messages": [response]}
    tool_calls = getattr(response, "tool_calls", None) or (getattr(response, "additional_kwargs", {}) or {}).get("tool_calls")
    if not tool_calls:
        m = re.search(r"EXEC_RESULT:\s*(.+)\s*$", (response.content or "").strip())
        if m:
            new_state["exec_result"] = m.group(1).strip()
    return new_state

def summarizer_node(state: MemState) -> MemState:
    msgs = state["messages"]
    recent = []
    for m in msgs[-20:]:
        if m.type in ("human", "ai", "tool"):
            recent.append(f"[{m.type}] {getattr(m,'content','')}")
    prompt = [
        SystemMessage(content="Summarize the conversation so far in 5 bullet points, retaining user preferences and key facts."),
        HumanMessage(content="\n".join(recent) if recent else "No prior content.")
    ]
    summary = summ_llm.invoke(prompt).content.strip()
    return {"messages": [SystemMessage(content=f"Conversation summary so far:\n{summary}")], "summary": summary}

def validator_node(state: MemState) -> MemState:
    """Guard-rail: If planner claims done but required steps missing for CURRENT CITY, override plan."""
    msgs = state["messages"]
    plan = state.get("plan") or {}
    if not plan:
        return {}

    user_txt = latest_user_text(msgs)
    city = extract_latest_city(user_txt) or ""

    if plan.get("done") is True and city:
        have_weather = has_tool_result_for_city(msgs, "get_weather", city)
        have_acts = has_tool_result_for_city(msgs, "suggest_city_activities", city)
        expr = math_expr_in_text(user_txt)
        have_calc = has_calculator_result(msgs)  # <-- FIX: detect calculator independently of city

        if not have_weather:
            new_plan = {"done": False, "next_step": {"tool": "get_weather", "input": city}, "rationale": "Required step 1: get_weather."}
            return {"plan": new_plan}
        if not have_acts:
            weather = extract_weather_snippet(msgs, city) or ""
            inp = f"city={city}; weather={weather}" if weather else city
            new_plan = {"done": False, "next_step": {"tool": "suggest_city_activities", "input": inp}, "rationale": "Required step 2: suggest activities."}
            return {"plan": new_plan}
        if expr and not have_calc:
            new_plan = {"done": False, "next_step": {"tool": "calculator", "input": expr}, "rationale": "User asked for math: run calculator."}
            return {"plan": new_plan}

        # All required steps satisfied → ensure final answer references current city
        fa = plan.get("final_answer") or ""
        if city and city.lower() not in fa.lower():
            plan["final_answer"] = fa.rstrip() + f" (City: {city.title()})"
            return {"plan": plan}

    return {}

# ---------- Graph wiring ----------

graph = StateGraph(MemState)
graph.add_node("planner", planner_node)
graph.add_node("validator", validator_node)
graph.add_node("summarizer", summarizer_node)
graph.add_node("executor", executor_node)
graph.add_node("tools", ToolNode(TOOLS))

def route_from_planner(state: MemState):
    return "validator"

def route_from_validator(state: MemState):
    plan = state.get("plan") or {}
    if plan.get("done"):
        return END
    N = 3
    turn = int(state.get("turn_count", 0))
    return "summarizer" if (turn % N == 0) else "executor"

def route_from_summarizer(state: MemState):
    return "executor"

def route_from_executor(state: MemState):
    last = state["messages"][-1]
    tc = getattr(last, "tool_calls", None) or (getattr(last, "additional_kwargs", {}) or {}).get("tool_calls")
    return "tools" if tc else "planner"

graph.set_entry_point("planner")
graph.add_conditional_edges("planner", route_from_planner)
graph.add_conditional_edges("validator", route_from_validator)
graph.add_conditional_edges("summarizer", route_from_summarizer)
graph.add_conditional_edges("executor", route_from_executor)
graph.add_edge("tools", "executor")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ---------- LTM profile ----------

def load_user_profile(path: str = "user_profile.json") -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {"name": "Priya", "preferences": {"indoor": True, "budget": "low", "likes_museums": True}}

# ---------- Demo ----------

if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "lab12-strict-thread-v2"}, "recursion_limit": 50}  # bump limit as a safety net

    profile = load_user_profile()
    intro = HumanMessage(content="Plan a short evening in Paris. First, tell me the weather, then suggest one indoor and one outdoor option, then finalize.")
    state_in = {"messages": [intro], "plan": None, "exec_result": None, "summary": None, "turn_count": 0, "user_profile": profile}

    out = app.invoke(state_in, cfg)
    print("\n=== RUN 1 DONE ===")
    for m in out["messages"]:
        print(m.type, "→", getattr(m, "content", ""))

    followup = HumanMessage(content="Now do the same for Chicago and also compute 25 + 18 + 12.5 (only the number).")
    out2 = app.invoke({"messages": [followup]}, cfg)
    print("\n=== RUN 2 DONE ===")
    for m in out2["messages"]:
        print(m.type, "→", getattr(m, "content", ""))
