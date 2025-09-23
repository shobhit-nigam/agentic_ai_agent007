# Tree-of-Thoughts (ToT) — toy demo
# ----------------------------------
# We expand multiple "thought branches" (states) and keep the best using a small beam search.
# Problem: start at 1, reach target=17 using operations [+3, *2] with max depth.
# Score: negative absolute distance to the target (higher is better).

from collections import deque

TARGET = 17
OPS = [
    ("+3", lambda x: x + 3),
    ("*2", lambda x: x * 2),
]

def score(x):
    # Higher is better
    return -abs(TARGET - x)

def expand(state):
    x, path = state
    for name, fn in OPS:
        nx = fn(x)
        yield (nx, path + [(name, nx)])

def beam_search(start=1, beam_width=3, max_depth=6):
    # Each item: (value, path) where path is list of (op_name, new_value)
    frontier = [(start, [])]
    best = (start, [])

    for depth in range(max_depth):
        # Generate children for all in frontier
        children = []
        for state in frontier:
            for child in expand(state):
                children.append(child)

        # Keep top-k by score
        children.sort(key=lambda s: score(s[0]), reverse=True)
        frontier = children[:beam_width]

        # Track best so far
        if frontier and score(frontier[0][0]) > score(best[0]):
            best = frontier[0]

        # Early exit if exact target found
        if frontier and frontier[0][0] == TARGET:
            return frontier[0], depth + 1, True

    return best, max_depth, False

def describe_path(path):
    lines = ["Start at 1"]
    cur = 1
    for op_name, new_val in path:
        lines.append(f"Think: If I apply {op_name} to {cur}, I get {new_val}.")
        cur = new_val
    lines.append(f"Result: {cur}")
    return "\n".join(lines)

if __name__ == "__main__":
    print("=== Tree-of-Thoughts (ToT) Demo ===")
    best_state, steps, found = beam_search(start=1, beam_width=3, max_depth=6)
    value, path = best_state
    print(f"Found exact target: {found}")
    print(f"Best value: {value} after exploring depth={steps}")
    print("\nReasoning trace:")
    print(describe_path(path))