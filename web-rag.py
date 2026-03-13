import json
import ollama
from langchain_community.tools.tavily_search import TavilySearchResults

# ── Config ──────────────────────────────────────────────
import os
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"  # Replace with your key

CHAT_MODEL = "llama3.2:3b"
TOP_K      = 3

# ── Init ─────────────────────────────────────────────────
tavily = TavilySearchResults(
    max_results=TOP_K,
    include_answer=True,
    include_raw_content=False,
)

# ── Retrieve ──────────────────────────────────────────────
def retrieve(query: str) -> tuple[list[dict], str]:
    results = tavily.invoke(query)
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

    source_index = "\n".join(
        f"[{i+1}] title: {s['title']} | url: {s['url']}"
        for i, s in enumerate(sources)
    )

    system_prompt = f"""You are a helpful research assistant.

Answer the user's question naturally. Use the web sources below as your primary knowledge.

RULES:
- Answer conversationally — no forced structure or bullet points unless it genuinely helps clarity.
- If your answer is based on the web sources, append the relevant source(s) as JSON at the very end, one per line:
  {{"title": "...", "url": "..."}}
- If the web sources are not relevant to the question, answer from your own knowledge but begin your response with "I'm not entirely sure, but..." and do NOT append any JSON.
- Only include sources that genuinely support your answer.

Source index:
{source_index}

Web Content:
{context}"""

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

    # ── Print answer, highlight JSON citation lines in grey ──
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                print(f"  \033[90m{json.dumps(parsed)}\033[0m")  # dim grey
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
