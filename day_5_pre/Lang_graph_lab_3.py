# tiny_langgraph_loop.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class State(TypedDict):
    messages: List[str]
    steps: int
    draft: str
    answer: str

g = StateGraph(State)

def planner(state: State) -> State:
    state["steps"] += 1
    # fake improvement by appending "✔"
    if not state["draft"]:
        state["draft"] = f"Draft: {state['messages'][-1]}"
    else:
        state["draft"] += " ✔"
    print(f"[planner] step={state['steps']} draft='{state['draft']}'")
    return state

def guard(state: State) -> State:
    # could be: score >= 0.8, tests pass, etc.
    print("[guard] check stop condition")
    return state

def finalize(state: State) -> State:
    state["answer"] = f"Final: {state['draft']}"
    print("[finalize] done.")
    return state

g.add_node("planner", planner)
g.add_node("guard", guard)
g.add_node("finalize", finalize)

g.set_entry_point("planner")
g.add_edge("planner", "guard")

def should_continue(state: State) -> str:
    return "planner" if state["steps"] < 2 else "finalize"

g.add_conditional_edges("guard", should_continue, {"planner": "planner", "finalize": "finalize"})
g.add_edge("finalize", END)

app = g.compile()

if __name__ == "__main__":
    out = app.invoke({"messages": ["Summarize LangGraph simply."], "steps": 0, "draft": "", "answer": ""})
    print("\n--- RESULT ---")
    print(out["answer"])
