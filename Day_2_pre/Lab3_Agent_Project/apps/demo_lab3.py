import sys
from agent.core import run_once

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(run_once(query))
        return

    print("Lab 3 demo. Examples:")
    print(" - remember favorite city is Chicago")
    print(" - what's the weather?")
    print(" - contact me at 555-123-4567 or a@b.com")
    print(" - wikipedia summary about Agentic AI")
    print("Type 'quit' to exit.\n")

    while True:
        q = input("you> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        print(run_once(q))

if __name__ == "__main__":
    main()
