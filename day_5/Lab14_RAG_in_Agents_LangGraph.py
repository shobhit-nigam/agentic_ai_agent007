
# Lab 14: RAG as a Tool inside Planner↔Executor (LangGraph) — FIXED
# -----------------------------------------------------------------
# Install:
#   pip install -U langgraph langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu typing_extensions tiktoken
#   export OPENAI_API_KEY="YOUR_KEY"
#
# Run:
#   python Lab14_RAG_in_Agents_LangGraph.py

from typing import Dict, Tuple, Optional, List
import re, json, os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------- Tiny knowledge base for RAG ----------

def load_demo_corpus() -> List[Tuple[str, str]]:
    return [
        ("paris.md", "Paris is the capital of France, known for the Louvre Museum and the Seine River. "
                     "It offers numerous indoor museums and outdoor river walks. "
                     "The Louvre holds famous artworks such as the Mona Lisa."),
        ("chicago.md", "Chicago sits on Lake Michigan. The Chicago Riverwalk is a popular outdoor walkway. "
                       "For indoor activities, the Art Institute of Chicago is one of the oldest and largest art museums in the United States."),
        ("rag.md", "Retrieval-Augmented Generation (RAG) augments an LLM with external knowledge. "
                   "Pipeline: chunk documents, embed them into vectors, index them in a vector store (e.g., FAISS), "
                   "retrieve top-k by similarity, and synthesize an answer that cites sources. "
                   "Benefits: smaller prompts, updatable knowledge, and traceable answers."),
    ]

def build_retriever(k: int = 3):
    docs = load_demo_corpus()
    texts, metas = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for fname, text in docs:
        for chunk in splitter.split_text(text):
            texts.append(chunk)
            metas.append({"source": fname})
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_texts(texts, embedding=emb, metadatas=metas)
    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 8, "lambda_mult": 0.5})
    return retriever

_RETRIEVER = None  # set in __main__

def _citations_from_docs(docs: List[Document], max_chars: int = 260) -> str:
    lines = ["CITATIONS:"]
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("source", "unknown")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        lines.append(f"[{i}] ({src}) {snippet}")
    if len(lines) == 1:
        lines.append("No relevant snippets found.")
    return "\n".join(lines)

# ---------- Tools ----------

@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """Return a minimal weather string for the given city using a tiny offline map.
    Input: city (e.g., 'Paris'). Output: one line weather string.
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
    """Return a one-sentence fact for a known city from a tiny offline knowledge base."""
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
    Input: 'Paris' or 'city=Paris; weather=sunny, 24°C'. Output: one line with picks.
    """
    catalog = {
        "chicago": {"indoor": ["Art Institute of Chicago"], "outdoor": ["Chicago Riverwalk"]},
        "paris": {"indoor": ["Louvre Museum"], "outdoor": ["Seine River Walk"]},
        "london": {"indoor": ["British Museum"], "outdoor": ["Hyde Park"]},
        "tokyo": {"indoor": ["teamLab Planets"], "outdoor": ["Ueno Park"]},
        "mumbai": {"indoor": ["CSMVS Museum"], "outdoor": ["Marine Drive"]},
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
        indoor, outdoor = data["indoor"][0], data["outdoor"][0]
    elif any(k in w for k in ["sunny", "clear"]):
        outdoor, indoor = data["outdoor"][0], data["indoor"][0]
    else:
        indoor, outdoor = data["indoor"][0], data["outdoor"][0]
    return f"City: {city}. Indoor: {indoor}. Outdoor: {outdoor}. (Weather-aware heuristics.)"

@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """Evaluate a simple math expression with + - * / ( ) . % only. Returns the numeric result."""
    import math, re as _re
    if not _re.fullmatch(r"[0-9+\-*/(). %\s]+", expression):
        return "Calculator error: invalid characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"Calculator error: {e}"

@tool("rag_search", return_direct=False)
def rag_search(query: str) -> str:
    """Retrieve top-k knowledge snippets as CITATIONS for a natural-language query."""
    global _RETRIEVER
    if _RETRIEVER is None:
        return "RAG not initialized."
    results: List[Document] = _RETRIEVER.invoke(query)
    return _citations_from_docs(results, max_chars=260)

TOOLS = [get_weather, mini_wiki, suggest_city_activities, calculator, rag_search]

# ---------- State ----------

class MemState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    plan: Optional[dict]
    exec_result: Optional[str]
    summary: Optional[str]
    turn_count: int
    user_profile: Optional[dict]
    last_citations_text: Optional[str]     #Citations block from RAG search

# ---------- LLMs ----------

planner_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
executor_llm_with_tools = executor_llm.bind_tools(TOOLS)
summ_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ---------- Policies ----------

PLANNER_SYS = (
    "You are PLANNER. You DO NOT call tools.\n"
    "Output STRICT JSON ONLY.\n"
    "Rules:\n"
    "1) For city planning requests: first call get_weather(city), then suggest_city_activities using the known weather.\n"
    "2) If the user's latest message asks for \"citations\"/\"sources\" OR general knowledge, plan to call rag_search(query) to fetch CITATIONS.\n"
    "3) If the user's latest message contains a math expression, include a calculator step.\n"
    "Schemas:\n"
    "  {\\\"done\\\": false, \\\"next_step\\\": {\\\"tool\\\": \\\"get_weather\\\"|\\\"suggest_city_activities\\\"|\\\"calculator\\\"|\\\"rag_search\\\", \\\"input\\\": \\\"<string>\\\"}, \\\"rationale\\\": \\\"<short>\\\"}\n"
    "  {\\\"done\\\": true, \\\"final_answer\\\": \\\"<concise final answer>\\\"}\n"
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
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "calculator":
            return True
        if isinstance(m, HumanMessage):
            break
    return False

def rag_used_since_last_user(messages: list[AnyMessage]) -> bool:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "rag_search":
            return True
        if isinstance(m, HumanMessage):
            break
    return False

def citations_required(msg: str, messages: list[AnyMessage], *, last_citations_text: Optional[str]) -> bool:
    need = any(k in msg.lower() for k in ["cite", "citation", "sources", "reference"])
    # If we already have citations captured in state (from any earlier step this turn), treat as required/satisfied context.
    return need or bool(last_citations_text)

def final_has_bracket_citations(final_answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", final_answer or ""))

# ---------- Nodes ----------

def planner_node(state: 'MemState') -> 'MemState':
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

def executor_node(state: 'MemState') -> 'MemState':
    msgs = list(state["messages"])
    last = msgs[-1]
    if isinstance(last, ToolMessage):
        new_state: 'MemState' = {"messages": []}
        if getattr(last, "name", "") == "rag_search":
            new_state["last_citations_text"] = last.content or ""
        exec_msgs = msgs + [
            SystemMessage(content=EXECUTOR_SYS),
            HumanMessage(content="Summarize the latest tool result and return only 'EXEC_RESULT: <one line>'")
        ]
    else:
        step = (state.get("plan") or {}).get("next_step", {})
        new_state = {"messages": []}
        exec_msgs = msgs + [
            SystemMessage(content=EXECUTOR_SYS),
            HumanMessage(content=f"Step to execute: {json.dumps(step)}")
        ]
    response = executor_llm_with_tools.invoke(exec_msgs)
    new_state["messages"] = [response]
    tool_calls = getattr(response, "tool_calls", None) or (getattr(response, "additional_kwargs", {}) or {}).get("tool_calls")
    if not tool_calls:
        m = re.search(r"EXEC_RESULT:\s*(.+)\s*$", (response.content or "").strip())
        if m:
            new_state["exec_result"] = m.group(1).strip()
    return new_state

def summarizer_node(state: 'MemState') -> 'MemState':
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

def validator_node(state: 'MemState') -> 'MemState':
    msgs = state["messages"]
    plan = state.get("plan") or {}
    if not plan:
        return {}

    user_txt = latest_user_text(msgs)
    city = extract_latest_city(user_txt) or ""
    need_cites = citations_required(user_txt, msgs, last_citations_text=state.get("last_citations_text"))

    if plan.get("done") is True:
        if city:
            have_weather = has_tool_result_for_city(msgs, "get_weather", city)
            have_acts = has_tool_result_for_city(msgs, "suggest_city_activities", city)
            if not have_weather:
                return {"plan": {"done": False, "next_step": {"tool": "get_weather", "input": city}, "rationale": "Required step 1: get_weather."}}
            if not have_acts:
                weather = extract_weather_snippet(msgs, city) or ""
                inp = f"city={city}; weather={weather}" if weather else city
                return {"plan": {"done": False, "next_step": {"tool": "suggest_city_activities", "input": inp}, "rationale": "Required step 2: suggest activities."}}
        expr = math_expr_in_text(user_txt)
        if expr and not has_calculator_result(msgs):
            return {"plan": {"done": False, "next_step": {"tool": "calculator", "input": expr}, "rationale": "User asked for math: run calculator."}}
        if need_cites and not final_has_bracket_citations(plan.get("final_answer", "")):
            if rag_used_since_last_user(msgs):
                fa = (plan.get("final_answer") or "").rstrip()
                plan["final_answer"] = fa + " [1]"
                return {"plan": plan}
            else:
                return {"plan": {"done": False, "next_step": {"tool": "rag_search", "input": user_txt}, "rationale": "Citations requested: fetch CITATIONS via rag_search."}}
    return {}

# ---------- Graph wiring ----------

graph = StateGraph(MemState)
graph.add_node("planner", planner_node)
graph.add_node("validator", validator_node)
graph.add_node("summarizer", summarizer_node)
graph.add_node("executor", executor_node)
graph.add_node("tools", ToolNode(TOOLS))

def route_from_planner(state: 'MemState'):
    return "validator"

def route_from_validator(state: 'MemState'):
    plan = state.get("plan") or {}
    if plan.get("done"):
        return END
    N = 3
    turn = int(state.get("turn_count", 0))
    return "summarizer" if (turn % N == 0) else "executor"

def route_from_summarizer(state: 'MemState'):
    return "executor"

def route_from_executor(state: 'MemState'):
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
    _RETRIEVER = build_retriever(k=3)
    cfg = {"configurable": {"thread_id": "lab14-thread-fixed"}, "recursion_limit": 60}
    profile = load_user_profile()

    q1 = HumanMessage(content="Explain Retrieval-Augmented Generation in 3 sentences. Include citations.")
    state1 = {"messages": [q1], "plan": None, "exec_result": None, "summary": None, "turn_count": 0, "user_profile": profile, "last_citations_text": None}
    out1 = app.invoke(state1, cfg)
    print("\n=== DEMO A DONE ===")
    for m in out1["messages"]:
        print(m.type, "→", getattr(m, "content", ""))

    q2 = HumanMessage(content="Plan a short evening in Paris: first weather, then one indoor and one outdoor pick. Include citations. Also compute 25 + 18 + 12.5 (only the number).")
    out2 = app.invoke({"messages": [q2]}, cfg)
    print("\n=== DEMO B DONE ===")
    for m in out2["messages"]:
        print(m.type, "→", getattr(m, "content", ""))
