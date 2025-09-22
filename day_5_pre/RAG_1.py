# RAG impact 
# - No external libraries
# - A tiny bag-of-words retriever
# - Two questions showing how RAG fixes wrong guesses

from collections import Counter
import math

# --- Mini knowledge base (what the agent can search) ---
DOCS = [
    ("returns",  "Our store accepts returns within 30 days with a receipt."),
    ("apollo11", "Apollo 11 landed on the Moon on July 20, 1969."),
]

# --- Utilities: tokenize, vectorize, cosine similarity ---
def tokenize(s: str):
    s = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s)
    return [w for w in s.split() if w]

def vec(tokens):
    return Counter(tokens)

def cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[t] * b[t] for t in set(a) & set(b))
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

# Build a tiny index once
INDEX = [(doc_id, text, vec(tokenize(text))) for doc_id, text in DOCS]

def retrieve(query: str, k: int = 1):
    qv = vec(tokenize(query))
    scored = sorted(
        ((cosine(qv, v), doc_id, text) for doc_id, text, v in INDEX),
        key=lambda x: x[0],
        reverse=True,
    )
    return scored[:k]

# --- BEFORE: "No RAG" (just guess from memory; can be wrong) ---
def no_rag_answer(q: str) -> str:
    ql = q.lower()
    if "return" in ql:
        return "Returns are allowed for 45 days."  # WRONG on purpose
    if "apollo" in ql or "moon" in ql:
        return "Apollo 11 landed in 1970."         # WRONG on purpose
    return "It depends."

# --- AFTER: With RAG (retrieve then generate from retrieved facts) ---
def with_rag_answer(q: str, threshold: float = 0.1) -> str:
    hits = retrieve(q, k=1)
    if not hits or hits[0][0] < threshold:
        return "Not enough info found in my sources."
    _, doc_id, text = hits[0]
    # "Generation" here is just quoting the best snippet to keep it simple.
    return f"(from: {doc_id}) {text}"

# --- Demo ---
def demo(question: str):
    print("QUESTION:", question)
    print("BEFORE (No RAG): ", no_rag_answer(question))
    print("AFTER  (With RAG):", with_rag_answer(question))
    print("-" * 60)

if __name__ == "__main__":
    demo("What is the store return window?")
    demo("When did Apollo 11 land on the Moon?")
