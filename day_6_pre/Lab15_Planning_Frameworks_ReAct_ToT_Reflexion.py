
# Lab 15: Planning Frameworks — ReAct, Tree-of-Thoughts (ToT), Reflexion (v2)
# ===========================================================================
# Prereqs
#   pip install -U langgraph langchain langchain-openai typing_extensions tiktoken
#   export OPENAI_API_KEY="YOUR_KEY"
#
# Run
#   python Lab15_Planning_Frameworks_ReAct_ToT_Reflexion_v2.py

from typing import Dict, Tuple, Optional, List
from typing_extensions import TypedDict, Annotated
from operator import add
import os, re, json

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ------------------------------
# Tools
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
# State
# ------------------------------

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add]
    plan: Optional[dict]                # the active plan (JSON) chosen
    candidates: Optional[List[dict]]    # ToT: candidate plans
    reflection: Optional[str]           # Reflexion: last critique
    attempt: int                        # number of replan attempts
    turn: int                           # planner turns
    reasoning_mode: str                 # 'react' | 'tot' | 'reflexion'

# ------------------------------
# LLMs
# ------------------------------

planner_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
critic_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm_with_tools = executor_llm.bind_tools(TOOLS)

# ------------------------------
# System prompts
# ------------------------------

PLANNER_SYS_COMMON = (
    "You are PLANNER. You DO NOT call tools. Output STRICT JSON ONLY.\n"
    "Schema options:\n"
    "  {\"done\": false, \"next_step\": {\"tool\": \"get_weather\"|\"suggest_city_activities\"|\"calculator\", \"input\": \"<string>\"}, \"rationale\": \"<short>\"}\n"
    "  {\"done\": true, \"final_answer\": \"<concise final>\"}\n"
    "City policy: For city plans, FIRST get_weather(city), THEN suggest_city_activities using the known weather.\n"
    "Math policy: If user requests a calculation, include a calculator step with the expression.\n"
)

PLANNER_SYS_REACT = (
    PLANNER_SYS_COMMON +
    "ReAct style: Think step-by-step but only output the JSON (no thoughts). One correct next step at a time.\n"
)

PLANNER_SYS_TOT = (
    PLANNER_SYS_COMMON +
    "Tree-of-Thoughts: Generate 3 DIVERSE candidate JSON plans for the immediate next step ONLY. "
    "Return STRICT JSON: {\"candidates\": [<plan1>, <plan2>, <plan3>]}\n"
)

SCORER_SYS = (
    "You are a SCORER. You will receive 1) the user message, 2) optional reflection, and 3) 2-5 candidate JSON plans.\n"
    "Score each candidate 1-10 for policy compliance (city policy, math policy), usefulness, and safety.\n"
    "Return STRICT JSON: {\"best_index\": <0-based index>, \"justification\": \"<short>\"}\n"
)

PLANNER_SYS_REFLEXION = (
    PLANNER_SYS_COMMON +
    "Reflexion style: Consider the provided reflection (if any) as advice on what to fix. "
    "Then output a single best JSON plan. Only the JSON.\n"
)

CRITIC_SYS = (
    "You are a CRITIC. Given the last user message and the planner's last plan that FAILED validation, "
    "write a one-sentence reflection advising how to fix it next time (e.g., 'Call get_weather before activities'). "
    "Return ONLY the reflection sentence."
)

EXECUTOR_SYS = (
    "You are EXECUTOR. You MUST execute exactly the step provided by the planner using the named tool.\n"
    "Never call a different tool. If the last message is a tool result, DO NOT call tools again; return a one-line summary.\n"
    "Respond with ONLY one line:\n"
    "EXEC_RESULT: <concise result>\n"
)

# ------------------------------
# Helper functions
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

def math_expr_in_text(text: str) -> Optional[str]:
    m = re.search(r"([0-9][0-9\.\s\+\-\*/\(\)%]+)", text or "")
    return m.group(1).strip() if m else None

# ------------------------------
# Nodes
# ------------------------------

def planner_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    mode = state.get("reasoning_mode", "react")
    sys_msgs = []
    # Compose system message based on mode
    if mode == "react":
        sys_msgs.append(SystemMessage(content=PLANNER_SYS_REACT))
    elif mode == "tot":
        sys_msgs.append(SystemMessage(content=PLANNER_SYS_TOT))
    else:
        # reflexion
        sys_msgs.append(SystemMessage(content=PLANNER_SYS_REFLEXION))
        if state.get("reflection"):
            sys_msgs.append(SystemMessage(content=f"Reflection: {state['reflection']}"))
    response = planner_llm.invoke(sys_msgs + msgs)
    content = response.content or ""
    plan = _extract_json(content)
    new_state: AgentState = {"messages": [response], "turn": state.get("turn", 0) + 1}
    if mode == "tot":
        # Expect candidates
        if plan and "candidates" in plan and isinstance(plan["candidates"], list):
            new_state["candidates"] = plan["candidates"]
        else:
            # Fallback: treat as single plan
            new_state["plan"] = plan or {}
    else:
        new_state["plan"] = plan or {}
    return new_state

def scorer_node(state: AgentState) -> AgentState:
    """Select best candidate for ToT."""
    candidates = state.get("candidates") or []
    if not candidates:
        return {}
    user = latest_user_text(state["messages"])
    reflection = state.get("reflection", "")
    # Build scoring prompt
    prompt = [
        SystemMessage(content=SCORER_SYS),
        HumanMessage(content=f"USER: {user}\nREFLECTION: {reflection}\nCANDIDATES:\n{json.dumps(candidates, indent=2)}")
    ]
    resp = critic_llm.invoke(prompt)
    data = _extract_json(resp.content or "") or {}
    idx = int(data.get("best_index", 0)) if candidates else 0
    chosen = candidates[min(max(idx, 0), len(candidates)-1)]
    return {"messages": [AIMessage(content=f"[SCORER] chose candidate {idx}: {data.get('justification','')}")], "plan": chosen, "candidates": None}

def validator_node(state: AgentState) -> AgentState:
    """Enforce simple policies before allowing done:true."""
    plan = state.get("plan") or {}
    if not plan:
        return {}

    user = latest_user_text(state["messages"])
    city = city_from_text(user)
    need_math = math_expr_in_text(user)

    # If done, verify policies
    if plan.get("done") is True:
        # City policy
        if city:
            if not has_tool_result(state["messages"], "get_weather"):
                return {"plan": {"done": False, "next_step": {"tool": "get_weather", "input": city}, "rationale": "City policy: call get_weather first."}}
            if not has_tool_result(state["messages"], "suggest_city_activities"):
                weather = extract_weather_text(state["messages"], city) or ""
                inp = f"city={city}; weather={weather}" if weather else city
                return {"plan": {"done": False, "next_step": {"tool": "suggest_city_activities", "input": inp}, "rationale": "City policy: suggest activities after weather."}}
        # Math policy
        if need_math and not has_tool_result(state["messages"], "calculator"):
            return {"plan": {"done": False, "next_step": {"tool": "calculator", "input": need_math}, "rationale": "Math policy: compute expression."}}
        # All good → allow END
        return {}
    # Not done → continue
    return {}

def executor_node(state: AgentState) -> AgentState:
    """Execute exactly one tool step or summarize a tool result."""
    msgs = list(state["messages"])
    last = msgs[-1]
    # If the last was a ToolMessage → summarize only, with tools disabled
    if isinstance(last, ToolMessage):
        resp = executor_llm.invoke(
            msgs + [
                SystemMessage(content=EXECUTOR_SYS),
                HumanMessage(content="Summarize the latest tool result and return only 'EXEC_RESULT: <one line>'")
            ]
        )
        new_state: AgentState = {"messages": [resp]}
        return new_state

    # Otherwise expect a single next_step from planner
    step = (state.get("plan") or {}).get("next_step") or {}
    # Execute exactly this tool (tools enabled)
    resp = executor_llm_with_tools.invoke(
        msgs + [
            SystemMessage(content=EXECUTOR_SYS),
            HumanMessage(content=f"Execute ONLY this step: {json.dumps(step)}")
        ]
    )
    new_state: AgentState = {"messages": [resp]}
    return new_state

def critic_node(state: AgentState) -> AgentState:
    """Produce a one-line reflection when validation failed (Reflexion)."""
    user = latest_user_text(state["messages"])
    last_plan = state.get("plan") or {}
    prompt = [
        SystemMessage(content=CRITIC_SYS),
        HumanMessage(content=f"USER: {user}\nLAST_PLAN: {json.dumps(last_plan, indent=2)}")
    ]
    resp = critic_llm.invoke(prompt)
    reflection = (resp.content or "").strip()
    return {"messages": [AIMessage(content=f"[CRITIC] {reflection}")], "reflection": reflection, "attempt": state.get("attempt", 0) + 1}

# ------------------------------
# Graph wiring
# ------------------------------

graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("scorer", scorer_node)
graph.add_node("validator", validator_node)
graph.add_node("executor", executor_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("critic", critic_node)

def route_from_planner(state: AgentState):
    mode = state.get("reasoning_mode", "react")
    if mode == "tot" and state.get("candidates"):
        return "scorer"
    return "validator"

def route_from_scorer(state: AgentState):
    return "validator"

def route_from_validator(state: AgentState):
    plan = state.get("plan") or {}
    # If done:true AND validator returned no correction → END
    if plan.get("done") is True:
        return END
    # If validator suggests a next_step → go execute
    if plan.get("next_step"):
        return "executor"
    # If validation rejected a final in Reflexion mode → go to critic
    if state.get("reasoning_mode") == "reflexion":
        return "critic"
    # Otherwise continue planning
    return "planner"

def route_from_executor(state: AgentState):
    last = state["messages"][-1]
    tcs = getattr(last, "tool_calls", None) or (getattr(last, "additional_kwargs", {}) or {}).get("tool_calls")
    return "tools" if tcs else "planner"

# After tools, always summarize (executor) and then go back to planner
graph.add_edge("tools", "executor")

graph.set_entry_point("planner")
graph.add_conditional_edges("planner", route_from_planner)
graph.add_conditional_edges("scorer", route_from_scorer)
graph.add_conditional_edges("validator", route_from_validator)
graph.add_conditional_edges("executor", route_from_executor)

app = graph.compile()

# ------------------------------
# Demos
# ------------------------------

def run_demo_react():
    cfg = {"configurable": {"thread_id": "lab15-react-v2"}, "recursion_limit": 40}
    state: AgentState = {"messages": [HumanMessage(content="Plan a short evening in Chicago. First tell weather, then one indoor and one outdoor pick.")],
                         "plan": None, "candidates": None, "reflection": None, "attempt": 0, "turn": 0, "reasoning_mode": "react"}
    out = app.invoke(state, cfg)
    print("\n=== DEMO (ReAct) ===")
    for m in out["messages"]:
        print(m.type, "→", getattr(m, "content", ""))

def run_demo_tot():
    cfg = {"configurable": {"thread_id": "lab15-tot-v2"}, "recursion_limit": 50}
    state: AgentState = {"messages": [HumanMessage(content="Plan a short evening in Paris. Remember: weather first, then activities.")],
                         "plan": None, "candidates": None, "reflection": None, "attempt": 0, "turn": 0, "reasoning_mode": "tot"}
    out = app.invoke(state, cfg)
    print("\n=== DEMO (Tree-of-Thoughts) ===")
    for m in out["messages"]:
        print(m.type, "→", getattr(m, "content", ""))

def run_demo_reflexion():
    cfg = {"configurable": {"thread_id": "lab15-reflexion-v2"}, "recursion_limit": 60}
    state: AgentState = {"messages": [HumanMessage(content="Suggest indoor/outdoor for London quickly. (No need for weather, just do it.)")],
                         "plan": None, "candidates": None, "reflection": None, "attempt": 0, "turn": 0, "reasoning_mode": "reflexion"}
    out1 = app.invoke(state, cfg)
    print("\n=== DEMO (Reflexion) PASS 1 ===")
    for m in out1["messages"]:
        print(m.type, "→", getattr(m, "content", ""))
    # second pass uses updated reflection (critic node)
    out2 = app.invoke({**out1, "reasoning_mode": "reflexion"}, cfg)
    print("\n=== DEMO (Reflexion) PASS 2 ===")
    for m in out2["messages"]:
        print(m.type, "→", getattr(m, "content", ""))

if __name__ == "__main__":
    run_demo_react()
    run_demo_tot()
    run_demo_reflexion()
