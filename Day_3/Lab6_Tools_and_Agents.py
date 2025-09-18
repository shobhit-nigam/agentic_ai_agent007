# Lab 6: Tools & Tool-Calling Agents with LangChain
# -------------------------------------------------
# Goal: Build on Lab 5 and teach how to define Tools and let an Agent decide when to use them.
# This lab demonstrates two agent styles:
#   1) OPENAI_FUNCTIONS (tool-calling) agent
#   2) ZERO_SHOT_REACT_DESCRIPTION (ReAct-style) agent
#
# Prereqs:
#   pip install langchain openai
#   export OPENAI_API_KEY="your_key"
#
# Note: Tools below are kept offline-friendly (mock weather, mini wiki, safe calculator).

from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

# -----------------------------
# Step 1 — Define our Tools
# -----------------------------

@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """Return simple weather for a given city from a tiny offline database.
    Input: city name (e.g., 'Paris')."""
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
    """Safely evaluate a simple math expression using Python eval with restrictions.
    Allowed chars: digits, + - * / ( ) . % and spaces."""
    import math, re
    if not re.fullmatch(r"[0-9+\-*/(). %\s]+", expression):
        return "Calculator error: invalid characters."
    try:
        # VERY restricted eval (no builtins)
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"

@tool("mini_wiki", return_direct=False)
def mini_wiki(topic: str) -> str:
    """Return a short summary from a tiny offline encyclopedia.
    Useful for quick references when internet is unavailable."""
    kb: Dict[str, str] = {
        "alan turing": "Alan Turing (1912–1954) was a mathematician and pioneer of computer science who formalized computation and contributed to codebreaking in WWII.",
        "agentic ai": "Agentic AI refers to systems that can plan, choose tools/actions, and adapt using feedback, rather than only producing text responses.",
        "langchain": "LangChain is a framework that provides building blocks for LLM apps: prompts, chains, tools, memory, and agents.",
    }
    return kb.get(topic.strip().lower(), "No entry found in mini_wiki. Try 'Alan Turing', 'Agentic AI', or 'LangChain'.")

# -----------------------------
# Step 2 — Create an LLM
# -----------------------------

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Collect tools into a list for the agent
tools = [get_weather, calculator, mini_wiki]

# -----------------------------
# Step 3 — Build a Tool-Calling Agent
# -----------------------------
# This uses OpenAI's function/tool-calling under the hood.
tool_calling_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,  # prints the reasoning/tool calls
)

# -----------------------------
# Step 4 — Try some queries
# -----------------------------
def demo_tool_calling():
    print("\n=== Tool-Calling Agent Demo ===\n")
    queries = [
        "What's the weather in Paris right now? Suggest one outdoor activity.",
        "Calculate: 23*17 + 3.5, then divide by 2.",
        "Use mini_wiki to summarize 'Alan Turing' in one sentence, then list two contributions."
    ]
    for q in queries:
        print(f"\nUser: {q}")
        # AgentExecutor supports .invoke({"input": ...}) or .run(...)
        result = tool_calling_agent.invoke({"input": q})
        print("Assistant:", result["output"])

# -----------------------------
# Step 5 — (Optional) Build a ReAct Agent
# -----------------------------
# This agent style reasons via 'Thought -> Action -> Observation' steps (printed via verbose).
react_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

def demo_react():
    print("\n=== ReAct Agent Demo ===\n")
    q = "Plan an evening in Chicago based on the weather. Include one indoor and one outdoor option."
    print(f"User: {q}")
    result = react_agent.invoke({"input": q})
    print("Assistant:", result["output"])

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    demo_tool_calling()
    demo_react()
