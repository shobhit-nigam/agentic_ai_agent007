# tiny_langgraph_linear.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# 1) define the shared backpack (state)
class State(TypedDict):
    messages: List[str]         # conversation log
    plan: str                   # what we'll do next
    answer: str                 # final output

# 2) make the graph object
g = StateGraph(State)

# 3) define nodes (plain Python functions)
def ingest(state: State) -> State:
    print("[ingest] got:", state["messages"][-1])
    # (normally: clean text, do safety checks, etc.)
    return state

def think(state: State) -> State:
    # (normally: call an LLM. here we'll fake it)
    user_text = state["messages"][-1]
    state["plan"] = f"Plan: answer the question: '{user_text}' in 1-2 lines."
    print("[think] plan:", state["plan"])
    return state

def finalize(state: State) -> State:
    # (normally: call an LLM. here we'll fake it)
    state["answer"] = f"Short answer to: {state['messages'][-1]}"
    print("[finalize] answer ready.")
    return state

# 4) register nodes
g.add_node("ingest", ingest)
g.add_node("think", think)
g.add_node("finalize", finalize)

# 5) wire arrows (edges)
g.set_entry_point("ingest")            # where the run starts
g.add_edge("ingest", "think")          # after ingest -> think
g.add_edge("think", "finalize")        # after think  -> finalize
g.add_edge("finalize", END)            # finalize ends the run

# 6) compile the graph into an app (you can also add a checkpointer here)
app = g.compile()

if __name__ == "__main__":
    # 7) run it
    out = app.invoke({"messages": ["What is LangGraph in simple terms?"], "plan": "", "answer": ""})
    print("\n--- RESULT ---")
    print(out["answer"])
