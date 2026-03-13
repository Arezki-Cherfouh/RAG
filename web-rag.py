import json
import ollama
from langchain_community.tools.tavily_search import TavilySearchResults

# ── Config ──────────────────────────────────────────────
import os
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key-here"  # Replace with your key

CHAT_MODEL = "llama3.2:3b"
TOP_K      = 3

# ── Init ─────────────────────────────────────────────────
tavily = TavilySearchResults(
    max_results=TOP_K,
    include_answer=True,        # get Tavily's own summary
    include_raw_content=False,
)

# ── Retrieve ──────────────────────────────────────────────
def retrieve(query: str) -> tuple[list[dict], str]:
    """
    Returns:
        sources  – list of {title, url} dicts
        context  – combined snippet text for the LLM
    """
    results = tavily.invoke(query)          # list of dicts from Tavily
    sources = []
    chunks  = []

    for r in results:
        title   = r.get("title", "No title")
        url     = r.get("url", "")
        content = r.get("content", "")

        sources.append({"title": title, "url": url})
        chunks.append(f"[{title}]\n{content}")

    context = "\n\n".join(chunks)
    return sources, context

# ── RAG Chat ──────────────────────────────────────────────
def rag_chat(user_query: str):
    sources, context = retrieve(user_query)

    # Build a numbered source index for the LLM to reference
    source_index = "\n".join(
        f"[{i+1}] title: {s['title']} | url: {s['url']}"
        for i, s in enumerate(sources)
    )

    system_prompt = f"""You are a helpful research assistant.
Answer the user's question using ONLY the web sources provided below.
If the information is not in the sources, say "I couldn't find relevant information online."
Do NOT use outside knowledge.

IMPORTANT FORMATTING RULES:
- Write your answer as a series of short statements (1-2 sentences each).
- After EVERY statement, on its own line, print a JSON object for the source(s) used for that statement.
- The JSON must use this exact format (one object per line, no array wrapper):
  {{"title": "...", "url": "..."}}
- Only cite sources that actually support that specific statement.
- Do not group all citations at the end.

Source index:
{source_index}

Web Content:
{context}"""

    # ── Get full response (no stream, so we can print cleanly) ──
    print("\n📝 Answer:\n")
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_query},
        ],
        stream=False,
    )
    answer = response["message"]["content"]

    # ── Pretty-print: colorize JSON citation lines ──
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                print(f"  \033[90m{json.dumps(parsed)}\033[0m")   # dim grey
            except json.JSONDecodeError:
                print(line)
        else:
            print(line)

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌐 Web RAG Chat (Ctrl+C to quit)\n")
    while True:
        try:
            query = input(">>> ")
            if query.strip():
                rag_chat(query)
        except KeyboardInterrupt:
            print("\nBye!")
            break
