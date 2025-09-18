# Lab 4B/C (Upstream-safe LLM)

This build fixes the `unexpected keyword argument 'upstream'` error by registering `llm_openai`
with a `lambda **kw: ...` wrapper that merges the upstream output into the prompt before calling OpenAI.

Quick start:
    pip install requests

    # Part B (live fetch + mock LLM)
    python -m apps.demo_lab4b "plan my weekend trip in Paris"

    # Part C (live fetch + OpenAI)
    export OPENAI_API_KEY=sk-...   # mac/linux
    python -m apps.demo_lab4c "plan my weekend trip in Paris"
