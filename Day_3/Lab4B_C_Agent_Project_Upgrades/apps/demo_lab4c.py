import sys
from agent.core import run_goal

def main():
    if len(sys.argv) > 1:
        print(run_goal(" ".join(sys.argv[1:]), live=True, use_openai=True))
        return
    print("Lab 4C demo (OpenAI + live). Set OPENAI_API_KEY first. Type 'quit' to exit.\n")
    while True:
        q = input("you> ").strip()
        if q.lower() in {"quit","exit"}: break
        print(run_goal(q, live=True, use_openai=True))

if __name__ == "__main__":
    main()
