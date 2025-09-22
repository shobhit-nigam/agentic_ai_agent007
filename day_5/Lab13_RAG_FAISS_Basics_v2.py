
# Lab 13: Retrieval-Augmented Generation (RAG) with FAISS — Foundations
# --------------------------------------------------------------------
# What you’ll learn (theory -> code):
# - Why RAG exists: context window limits vs external knowledge
# - RAG pipeline: CHUNK -> EMBED -> INDEX -> RETRIEVE -> SYNTHESIZE
# - Build a tiny local corpus, index it with FAISS, and query it
# - Return compact CITATIONS and synthesize an answer that cites sources
#
# Prereqs:
#   pip install -U langchain langchain-openai faiss-cpu tiktoken
#   export OPENAI_API_KEY="your_key"
#
# Run:
#   python Lab13_RAG_FAISS_Basics.py
#
# Notes:
# - We keep everything local (FAISS). No external vector DB needed.
# - This lab focuses on *RAG basics*. In Lab 14 we’ll wire this up as a tool
#   inside the Planner↔Executor MAS and enforce citations there.

from typing import List, Tuple
from pathlib import Path
import os, re, json

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------- 1) Build a tiny corpus (for teaching) ----------

def load_demo_corpus() -> List[Tuple[str, str]]:
    """Return (filename, text) pairs. In real projects you'd read files from disk."""
    docs = [
        ("paris.md", """Paris is the capital of France, known for the Louvre Museum and the Seine River.
It offers numerous indoor museums and outdoor river walks. The Louvre holds famous artworks such as the Mona Lisa."""),
        ("chicago.md", """Chicago sits on Lake Michigan. The Chicago Riverwalk is a popular outdoor walkway.
For indoor activities, the Art Institute of Chicago is one of the oldest and largest art museums in the United States."""),
        ("rag.md", """Retrieval-Augmented Generation (RAG) augments an LLM with external knowledge.
Pipeline: chunk documents, embed them into vectors, index them in a vector store (e.g., FAISS), retrieve top-k by similarity,
and synthesize an answer that cites sources. Benefits: smaller prompts, updatable knowledge, and traceable answers."""),
    ]
    return docs

# ---------- 2) Chunk & embed ----------

def build_faiss_retriever(chunk_size: int = 500, chunk_overlap: int = 50, k: int = 3):
    docs = load_demo_corpus()
    texts, metadatas = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for fname, text in docs:
        for chunk in splitter.split_text(text):
            texts.append(chunk)
            metadatas.append({"source": fname})

    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_texts(texts, embedding=emb, metadatas=metadatas)
    retriever = store.as_retriever(
        search_type = "mmr", 
        search_kwargs={"k": k, "fetch_k": 8, "lambda_mult": 0.5}
    )
    return retriever

# ---------- 3) A compact RAG "tool": return enumerated CITATIONS ----------

def rag_search(retriever, query: str, max_chars: int = 260) -> str:
    """Retrieve top snippets relevant to `query` and return numbered citations.
    Output format:
      CITATIONS:
      [1] (source.md) snippet...
      [2] (source.md) snippet...
    """
    results: List[Document] = retriever.get_relevant_documents(query)
    results: List[Document] = retriever.invoke(query)
    lines = ["CITATIONS:"]
    for i, d in enumerate(results, 1):
        src = d.metadata.get("source", "unknown")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        lines.append(f"[{i}] ({src}) {snippet}")
    if len(lines) == 1:
        lines.append("No relevant snippets found.")
    return "\n".join(lines)

# ---------- 4) Synthesize an answer that uses the citations ----------

SYS_RAG = (
    "You are an expert assistant.\n"
    "You will receive CITATIONS listing numbered snippets from the knowledge base.\n"
    "Write a concise answer ONLY the information from those snippets.\n"
    "Rules:\n"
    "1) Add bracketed citations [n] immediately after corresponding claim.\n"
    "2) Ignore irrelevant snippets: only cite numbers you actually used.\n"
    "3) Do NOT add disclaimers like 'not mentioned' if at least one snippet contains the needed information\n"
    "4) If information is truly missing from ALL snippetrs, say so briefly.\n"
)

def answer_with_citations(query: str, citations_text: str, llm=None) -> str:
    """Given a query and the CITATIONS text, ask the LLM to write a short cited answer."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = f"{SYS_RAG}\n\nUSER QUERY:\n{query}\n\n{citations_text}\n\nANSWER:"
    return llm.invoke(prompt).content

# ---------- 5) Demo ----------

def demo():
    retriever = build_faiss_retriever(k=3)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    demos = [
        ("Explain RAG in one short paragraph with 2 citations.",
         "Explain RAG in one short paragraph with 2 citations."),
        ("Plan an evening in Paris (one indoor + one outdoor) using the knowledge base. Cite sources.",
         "Plan an evening in Paris (one indoor + one outdoor) using the knowledge base. Cite sources."),
        ("What is the Chicago Riverwalk? Suggest one indoor alternative; include citations.",
         "What is the Chicago Riverwalk? Suggest one indoor alternative; include citations."),
    ]

    for title, q in demos:
        print(f"\n=== {title} ===")
        cits = rag_search(retriever, q)
        print(cits)
        ans = answer_with_citations(q, cits, llm=llm)
        print("\nAnswer:\n", ans)

if __name__ == "__main__":
    demo()
