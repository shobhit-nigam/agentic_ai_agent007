# tiny_langgraph_branch.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class State(TypedDict):
    messages: List[str]
    plan: str
    interim: str
    answer: str

g = StateGraph(State)

def think(state: State) -> State:
    user = state["messages"][-1]
    state["plan"] = "use calculator" if any(ch.isdigit() for ch in user) else "answer directly"
    print("[think] plan:", state["plan"])
    return state

def calculator(state: State) -> State:
    # toy calculator: just count digits as a silly "sum"
    digits = [int(c) for c in state["messages"][-1] if c.isdigit()]
    state["interim"] = f"calc_result={sum(digits)}"
    print("[calculator] interim:", state["interim"])
    return state

def finalize(state: State) -> State:
    plan = state["plan"]
    if plan == "use calculator":
        state["answer"] = f"Used tool → {state['interim']}"
    else:
        state["answer"] = f"Direct answer to: {state['messages'][-1]}"
    print("[finalize] answer ready.")
    return state

g.add_node("think", think)
g.add_node("calculator", calculator)
g.add_node("finalize", finalize)

g.set_entry_point("think")

# --- conditional routing ---
def route_from_think(state: State) -> str:
    return "calculator" if state["plan"] == "use calculator" else "finalize"

g.add_conditional_edges(
    "think",
    route_from_think,            # returns the name of the next node
    {"calculator": "calculator", "finalize": "finalize"}
)

g.add_edge("calculator", "finalize")
g.add_edge("finalize", END)

app = g.compile()

if __name__ == "__main__":
    print("\nCase 1: No numbers")
    out = app.invoke({"messages": ["Explain photosynthesis"], "plan": "", "interim": "", "answer": ""})
    print("->", out["answer"])

    print("\nCase 2: Has numbers")
    out = app.invoke({"messages": ["What is 123 + 45?"], "plan": "", "interim": "", "answer": ""})
    print("->", out["answer"])
