import sys
from agent.core import run_goal

def main():
    if len(sys.argv) > 1:
        print(run_goal(" ".join(sys.argv[1:])))
        return

    print("Lab 4 demo. Examples:")
    print(" - remember favorite city is Chicago")
    print(" - plan my weekend trip")
    print(" - plan my day in Paris")
    print(" - research Agentic AI")
    print("Type 'quit' to exit.\n")

    while True:
        q = input("you> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        print(run_goal(q))

if __name__ == "__main__":
    main()
