import sys
from agent.core import run_once

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(run_once(query))
        return

    print("Lab 1 demo. Try things like:")
    print(" - what's the weather in Paris today?")
    print(" - wikipedia summary about Agentic AI")
    print(" - recommend dinner in New Delhi")
    print("Type 'quit' to exit.\n")

    while True:
        q = input("you> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        print(run_once(q))

if __name__ == "__main__":
    main()
