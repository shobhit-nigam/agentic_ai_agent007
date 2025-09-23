
# Lab 16: Meta-Reasoning & Reflexion Deep Dive (LangGraph)
# =======================================================
# What’s new vs Lab 15?
# - Structured validation "violations" with codes (CITY_ORDER, MATH_MISSING, FORMAT)
# - Critic-on-override with structured reflections (why, fix, rule)
# - Persistent "lessons" injected into the planner as meta-advice (within session)
# - Scorecard (counts policy violations, steps, tool calls) for teachable metrics
# - Input normalization retained for deterministic traces
#
# Prereqs
#   pip install -U langgraph langchain langchain-openai typing_extensions tiktoken
#   export OPENAI_API_KEY="YOUR_KEY"
#
# Run
#   python Lab16_MetaReasoning_Reflexion_DeepDive_v1.py

from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict, Annotated
from operator import add
import re, json

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ------------------------------
# Tools (same lightweight set)
# ------------------------------

@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """Return a tiny canned weather string. Input: city name, e.g., 'Paris'."""
    db = {"paris": "sunny, 24°C", "chicago": "cloudy, 18°C", "london": "overcast, 17°C"}
    c = (city or "").strip().lower()
    if c in db:
        return f"The weather in {city} is {db[c]}."
    return f"No weather found for '{city}'. Assume mild (22°C) and clear for demo."

@tool("suggest_city_activities", return_direct=False)
def suggest_city_activities(query: str) -> str:
    """Recommend ONE indoor and ONE outdoor activity. Input supports 'city=Paris; weather=sunny'."""
    def parse(q: str):
        q = (q or "").strip()
        if "city=" in q.lower():
            parts = [p.strip() for p in q.split(";")]
            city, weather = "", ""
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    if k.strip().lower() == "city":
                        city = v.strip()
                    elif k.strip().lower() == "weather":
                        weather = v.strip()
            return city or q, weather
        return q, ""
    catalog = {
        "paris": {"indoor": "Louvre Museum", "outdoor": "Seine River Walk"},
        "chicago": {"indoor": "Art Institute of Chicago", "outdoor": "Chicago Riverwalk"},
        "london": {"indoor": "British Museum", "outdoor": "Hyde Park"},
    }
    city, weather = parse(query)
    c = (city or "").strip().lower()
    data = catalog.get(c)
    if not data:
        return "Indoor: local museum; Outdoor: central park or riverfront."
    w = (weather or "").lower()
    if "rain" in w or "storm" in w:
        return f"City: {city}. Indoor: {data['indoor']}. Outdoor: {data['outdoor']} (prefer indoor first due to weather)."
    if "sunny" in w or "clear" in w:
        return f"City: {city}. Outdoor: {data['outdoor']}. Indoor: {data['indoor']} (enjoy outdoors first)."
    return f"City: {city}. Indoor: {data['indoor']}. Outdoor: {data['outdoor']}."

@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression using + - * / ( ) . % only."""
    import math
    if not re.fullmatch(r"[0-9+\-*/(). %\s]+", expression or ""):
        return "Calculator error: invalid characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"Calculator error: {e}"

TOOLS = [get_weather, suggest_city_activities, calculator]

# ------------------------------
# State with meta-reasoning fields
# ------------------------------

class Scorecard(TypedDict):
    planner_turns: int
    tool_calls: int
    violations: int

class Violation(TypedDict, total=False):
    code: str          # 'CITY_ORDER' | 'MATH_MISSING' | 'FORMAT'
    message: str
    fix: str           # short imperative fix phrasing

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add]
    plan: Optional[dict]                  # current planner plan
    reasoning_mode: str                   # 'react' | 'reflexion'
    lessons: List[str]                    # persistent advice this session
    reflection: Optional[str]             # last reflection from Critic
    violation: Optional[Violation]        # last validator violation, if any
    score: Scorecard                      # simple metrics

# ------------------------------
# LLMs
# ------------------------------

planner_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
critic_llm  = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm_with_tools = executor_llm.bind_tools(TOOLS)

# ------------------------------
# System prompts
# ------------------------------

PLANNER_SYS = (
    "You are PLANNER. You DO NOT call tools. Output STRICT JSON ONLY.\n"
    "Schema options:\n"
    "  {\"done\": false, \"next_step\": {\"tool\": \"get_weather\"|\"suggest_city_activities\"|\"calculator\", \"input\": \"<string>\"}, \"rationale\": \"<short>\"}\n"
    "  {\"done\": true, \"final_answer\": \"<concise final>\"}\n"
    "City policy: For city plans, FIRST get_weather(city), THEN suggest_city_activities using the known weather.\n"
    "Math policy: If user requests a calculation, include a calculator step with the expression.\n"
    "Consider the LESSONS (if any) below as high-priority planning advice.\n"
)

CRITIC_SYS = (
    "You are a CRITIC. Given a failed or overridden plan, write a one-sentence reflection with:\n"
    " - why it failed (policy/format/tool choice)\n"
    " - the concrete fix\n"
    " - a general rule to remember\n"
    "Return STRICT JSON: {\"why\": \"...\", \"fix\": \"...\", \"rule\": \"...\"}\n"
)

EXECUTOR_SYS = (
    "You are EXECUTOR. Execute exactly the named tool and input produced by the planner.\n"
    "If the last message is a ToolMessage, do NOT call tools again; return a one-line summary.\n"
    "Respond with ONLY one line: EXEC_RESULT: <concise result>\n"
)

# ------------------------------
# Utilities
# ------------------------------

def _extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}\s*$", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def latest_user_text(messages: List[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""

def has_tool_result(messages: List[AnyMessage], tool_name: str) -> bool:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == tool_name:
            return True
        if isinstance(m, HumanMessage):
            break
    return False

def city_from_text(text: str) -> Optional[str]:
    for city in ["paris", "chicago", "london"]:
        if city in (text or "").lower():
            return city
    m = re.search(r"in\s+([A-Z][a-z]+)\b", text or "")
    if m:
        cand = m.group(1).lower()
        if cand in ["paris", "chicago", "london"]:
            return cand
    return None

def extract_weather_text(messages: List[AnyMessage], city_hint: str) -> Optional[str]:
    city_low = (city_hint or "").lower()
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

def math_expr_in_text(text: str) -> Optional[str]:
    m = re.search(r"([0-9][0-9\.\s\+\-\*/\(\)%]+)", text or "")
    return m.group(1).strip() if m else None

def normalize_step(step: dict, messages: List[AnyMessage]) -> dict:
    """Ensure suggest_city_activities receives 'city=...; weather=...' when possible."""
    if not step:
        return step
    tool = step.get("tool")
    if tool != "suggest_city_activities":
        return step
    inp = (step.get("input") or "").strip()
    user = latest_user_text(messages)
    city = city_from_text(user) or ""
    weather = extract_weather_text(messages, city) if city else None
    weather = weather or ""
    if "city=" not in inp.lower():
        if city:
            inp = f"city={city}; weather={weather}" if weather else city
    return {**step, "input": inp}

# ------------------------------
# Nodes
# ------------------------------

def planner_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    lessons = state.get("lessons", [])
    sys_msgs = [SystemMessage(content=PLANNER_SYS)]
    if lessons:
        sys_msgs.append(SystemMessage(content="LESSONS:\n- " + "\n- ".join(lessons[:5])))
    if state.get("reflection"):
        sys_msgs.append(SystemMessage(content=f"Reflection: {state['reflection']}"))
    response = planner_llm.invoke(sys_msgs + msgs)
    content = response.content or ""
    plan = _extract_json(content) or {}
    return {"messages": [response], "plan": plan, "score": {**state["score"], "planner_turns": state["score"]["planner_turns"] + 1}}

def validator_node(state: AgentState) -> AgentState:
    """Return corrections as violations, if any. Else allow forward progress."""
    plan = state.get("plan") or {}
    user = latest_user_text(state["messages"])
    city = city_from_text(user)
    need_math = math_expr_in_text(user)

    # If final, enforce policy
    if plan.get("done") is True:
        if city and not has_tool_result(state["messages"], "get_weather"):
            return {
                "plan": {"done": False, "next_step": {"tool": "get_weather", "input": city}, "rationale": "City policy: call get_weather first."},
                "violation": {"code": "CITY_ORDER", "message": "Finalized without weather first.", "fix": "Call get_weather(city) before activities."},
                "score": {**state["score"], "violations": state["score"]["violations"] + 1},
            }
        if city and not has_tool_result(state["messages"], "suggest_city_activities"):
            weather = extract_weather_text(state["messages"], city) or ""
            inp = f"city={city}; weather={weather}" if weather else city
            return {
                "plan": {"done": False, "next_step": {"tool": "suggest_city_activities", "input": inp}, "rationale": "City policy: suggest activities after weather."},
                "violation": {"code": "CITY_ORDER", "message": "Finalized before suggesting activities.", "fix": "Call suggest_city_activities after get_weather."},
                "score": {**state["score"], "violations": state["score"]["violations"] + 1},
            }
        if need_math and not has_tool_result(state["messages"], "calculator"):
            return {
                "plan": {"done": False, "next_step": {"tool": "calculator", "input": need_math}, "rationale": "Math policy: compute expression."},
                "violation": {"code": "MATH_MISSING", "message": "Finalized without computing requested math.", "fix": "Call calculator(expression) before finalizing."},
                "score": {**state["score"], "violations": state["score"]["violations"] + 1},
            }
        # OK to end
        return {}

    # If not final but next_step exists, allow progress
    if plan.get("next_step"):
        return {}
    # Otherwise format problem
    return {
        "plan": {"done": False, "next_step": {"tool": "get_weather", "input": city or "Paris"}, "rationale": "Defaulting to weather-first due to format."},
        "violation": {"code": "FORMAT", "message": "Planner did not produce a valid plan.", "fix": "Follow the JSON schema strictly."},
        "score": {**state["score"], "violations": state["score"]["violations"] + 1},
    }

def executor_node(state: AgentState) -> AgentState:
    msgs = list(state["messages"])
    last = msgs[-1]
    # If last was tool result → summarize only, tools disabled
    if isinstance(last, ToolMessage):
        resp = executor_llm.invoke(
            msgs + [
                SystemMessage(content=EXECUTOR_SYS),
                HumanMessage(content="Summarize the latest tool result and return only 'EXEC_RESULT: <one line>'")
            ]
        )
        return {"messages": [resp]}

    # Execute exactly one planned step
    step = (state.get("plan") or {}).get("next_step") or {}
    step = normalize_step(step, msgs)
    resp = executor_llm_with_tools.invoke(
        msgs + [
            SystemMessage(content=EXECUTOR_SYS),
            HumanMessage(content=f"Execute ONLY this step: {json.dumps(step)}")
        ]
    )
    return {"messages": [resp], "score": {**state["score"], "tool_calls": state["score"]["tool_calls"] + 1}}

def critic_node(state: AgentState) -> AgentState:
    """Turn violation into a structured reflection and add to lessons."""
    viol = state.get("violation") or {}
    user = latest_user_text(state["messages"])
    plan = state.get("plan") or {}
    if viol:
        # Create a reflection deterministically from violation
        why  = viol.get("message", "A policy was violated.")
        fix  = viol.get("fix", "Follow the policy step before finalizing.")
        rule = {"CITY_ORDER": "Always call get_weather(city) before suggesting activities.",
                "MATH_MISSING": "Always compute requested math before finalizing.",
                "FORMAT": "Always output strict JSON schema."}.get(viol.get("code",""), "Follow the stated policies.")
        reflection_obj = {"why": why, "fix": fix, "rule": rule}
    else:
        # Fallback to model-based critic
        prompt = [
            SystemMessage(content=CRITIC_SYS),
            HumanMessage(content=f"USER: {user}\nLAST_PLAN:\n{json.dumps(plan, indent=2)}")
        ]
        resp = critic_llm.invoke(prompt)
        reflection_obj = _extract_json(resp.content or "") or {"why": "Unknown", "fix": "Follow policies", "rule": "Follow policies"}
    sentence = f"{reflection_obj['why']} Fix: {reflection_obj['fix']} Rule: {reflection_obj['rule']}"
    lessons = state.get("lessons", [])
    # De-duplicate short lessons
    if reflection_obj['rule'] not in lessons:
        lessons = [reflection_obj['rule']] + lessons
    return {"messages": [AIMessage(content=f"[CRITIC] {sentence}")], "reflection": sentence, "lessons": lessons, "violation": None}

# ------------------------------
# Graph wiring
# ------------------------------

graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("validator", validator_node)
graph.add_node("executor", executor_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("critic", critic_node)

def route_from_planner(state: AgentState):
    return "validator"

def route_from_validator(state: AgentState):
    plan = state.get("plan") or {}
    if plan.get("done") is True and not state.get("violation"):
        return END
    if plan.get("next_step"):
        # If validation introduced a violation object (override), in Reflexion mode go to critic first
        if state.get("reasoning_mode") == "reflexion" and state.get("violation"):
            return "critic"
        return "executor"
    # else keep planning
    return "planner"

def route_from_executor(state: AgentState):
    last = state["messages"][-1]
    tcs = getattr(last, "tool_calls", None) or (getattr(last, "additional_kwargs", {}) or {}).get("tool_calls")
    return "tools" if tcs else "planner"

graph.add_edge("tools", "executor")
graph.set_entry_point("planner")
graph.add_conditional_edges("planner", route_from_planner)
graph.add_conditional_edges("validator", route_from_validator)
graph.add_conditional_edges("executor", route_from_executor)

app = graph.compile()

# ------------------------------
# Demos
# ------------------------------

def run_demo_clean_react():
    print("\n=== DEMO A: ReAct clean flow (Paris, weather → activities → final) ===")
    cfg = {"configurable": {"thread_id": "lab16-react"}, "recursion_limit": 50}
    state: AgentState = {
        "messages": [HumanMessage(content="Plan a short evening in Paris. First weather, then one indoor and one outdoor.")],
        "plan": None,
        "reasoning_mode": "react",
        "lessons": [],
        "reflection": None,
        "violation": None,
        "score": {"planner_turns": 0, "tool_calls": 0, "violations": 0}
    }
    out = app.invoke(state, cfg)
    for m in out["messages"]:
        print(m.type, "→", getattr(m, "content", ""))
    print("SCORE:", out["score"])

def run_demo_reflexion_violation_then_learn():
    print("\n=== DEMO B: Reflexion (user tempts to skip weather; validator blocks; critic teaches; planner adapts) ===")
    cfg = {"configurable": {"thread_id": "lab16-reflexion"}, "recursion_limit": 80}
    s: AgentState = {
        "messages": [HumanMessage(content="Suggest indoor/outdoor for London fast. (No weather needed.)")],
        "plan": None,
        "reasoning_mode": "reflexion",
        "lessons": [],
        "reflection": None,
        "violation": None,
        "score": {"planner_turns": 0, "tool_calls": 0, "violations": 0}
    }
    out1 = app.invoke(s, cfg)
    for m in out1["messages"]:
        print(m.type, "→", getattr(m, "content", ""))
    print("SCORE after pass1:", out1["score"])
    # second pass continues with injected lessons/reflection already in state
    out2 = app.invoke({**out1, "reasoning_mode": "reflexion"}, cfg)
    for m in out2["messages"]:
        print(m.type, "→", getattr(m, "content", ""))
    print("SCORE after pass2:", out2["score"])

def run_demo_math_policy():
    print("\n=== DEMO C: Math policy (ensure calculator runs before final) ===")
    cfg = {"configurable": {"thread_id": "lab16-math"}, "recursion_limit": 60}
    s: AgentState = {
        "messages": [HumanMessage(content="In Chicago, suggest indoor/outdoor and compute 25 + 18 + 12.5 (only the number).")],
        "plan": None,
        "reasoning_mode": "react",
        "lessons": [],
        "reflection": None,
        "violation": None,
        "score": {"planner_turns": 0, "tool_calls": 0, "violations": 0}
    }
    out = app.invoke(s, cfg)
    for m in out["messages"]:
        print(m.type, "→", getattr(m, "content", ""))
    print("SCORE:", out["score"])

if __name__ == "__main__":
    run_demo_clean_react()
    run_demo_reflexion_violation_then_learn()
    run_demo_math_policy()
