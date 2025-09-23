# ReAct (Reason + Act) — toy demo
# --------------------------------
# A tiny agent alternates between "thinking" and "acting" by calling simple tools:
# - calculator(expression): safe arithmetic
# - wiki_lookup(query): look up fixed facts from a tiny in-memory "wiki"
#
# This shows the ReAct pattern:
# Thought -> Action -> Observation -> Thought -> Action -> ... -> Answer

import re

# --- Tools ------------------------------------------------------------
def calculator(expression: str) -> float:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("Unsafe characters in expression.")
    # Evaluate safely (for our limited arithmetic use-case)
    return eval(expression, {"__builtins__": {}}, {})

WIKI = {
    "Pride and Prejudice author": "Jane Austen",
    "capital of France": "Paris",
    "speed of light (m/s)": "299792458",
}

def wiki_lookup(query: str) -> str:
    # Very small toy "knowledge base" lookup
    return WIKI.get(query.strip(), "I don't know")

# --- ReAct loop -------------------------------------------------------
def react_solve(task: str):
    log = []
    def think(text): log.append(("Thought", text))
    def act(tool, arg):
        log.append(("Action", f"{tool}({arg!r})"))
        if tool == "calculator":
            obs = str(calculator(arg))
        elif tool == "wiki_lookup":
            obs = wiki_lookup(arg)
        else:
            obs = "Unknown tool"
        log.append(("Observation", obs))
        return obs

    # Very naive pattern-based "reasoning" to keep this pure-Python and deterministic
    if "calculate" in task or re.search(r"[0-9].*[+\-*/()].*[0-9]", task):
        expr = re.findall(r"[\d+\-*/(). ]+", task)
        expr = max(expr, key=len).strip() if expr else ""
        think(f"I should evaluate the arithmetic expression: {expr}")
        val = act("calculator", expr)
        think("Now I can respond with the final numeric result.")
        log.append(("Answer", val))
        return log

    if "author" in task and "Pride and Prejudice" in task:
        think("This is a fact lookup; I should use wiki_lookup.")
        obs = act("wiki_lookup", "Pride and Prejudice author")
        think("Use the looked-up value as the answer.")
        log.append(("Answer", obs))
        return log

    if "capital of France" in task:
        think("This is a fact lookup; I should use wiki_lookup.")
        obs = act("wiki_lookup", "capital of France")
        think("Use the looked-up value as the answer.")
        log.append(("Answer", obs))
        return log

    # Fallback
    think("I'm not sure which tool to use. I will answer directly if possible.")
    log.append(("Answer", "I don't know"))
    return log

def pretty_print(log):
    for kind, text in log:
        print(f"{kind}: {text}")

if __name__ == "__main__":
    print("=== ReAct Demo ===")
    tasks = [
        "What is (2 + 3) * 4? Please calculate it.",
        "Who is the author of Pride and Prejudice?",
        "What is the capital of France?",
    ]
    for t in tasks:
        print(f"\nTask: {t}")
        pretty_print(react_solve(t))