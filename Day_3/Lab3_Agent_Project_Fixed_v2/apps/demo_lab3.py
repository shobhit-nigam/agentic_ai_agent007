import sys
from agent.core import run_once

def main():
    if len(sys.argv) > 1:
        print(run_once(" ".join(sys.argv[1:])))
        return
    print("Lab 3 demo (PII redaction + memory). Examples:")
    print(" - remember favorite city is Chicago")
    print(" - what's the weather?   (uses memory default city)")
    print(" - contact me at 555-123-4567 or mr.nigam@gmail.com  (will redact)")
    print(" - wikipedia summary about Agentic AI")
    print("Type 'quit' to exit.\n")
    while True:
        q = input("you> ").strip()
        if q.lower() in {'quit','exit'}: break
        print(run_once(q))

if __name__ == "__main__":
    main()
