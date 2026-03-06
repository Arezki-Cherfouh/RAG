import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
from sentence_transformers import SentenceTransformer
import uuid

# ── Config ──────────────────────────────────────────────
QDRANT_URL    = "http://localhost:6333"
COLLECTION    = "documents"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHAT_MODEL    = "llama3.2:3b"
VECTOR_SIZE   = 384   # matches all-MiniLM-L6-v2 output
TOP_K         = 3

# ── Init ─────────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)
client   = QdrantClient(url=QDRANT_URL)

def setup_db():
    """Create Qdrant collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print("✅ Collection created")
    else:
        print("✅ Collection already exists")

# ── Ingest ────────────────────────────────────────────────
def ingest_documents(docs: list[str]):
    """Embed and store a list of text chunks."""
    embeddings = embedder.encode(docs, show_progress_bar=True)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"content": doc}
        )
        for doc, emb in zip(docs, embeddings)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"✅ Ingested {len(docs)} documents")

# ── Retrieve ──────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    query_emb = embedder.encode([query])[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION,
        query=query_emb,
        limit=top_k,
        with_payload=True
    ).points
    return [r.payload["content"] for r in results]

# ── RAG Chat ──────────────────────────────────────────────
def rag_chat(user_query: str):
    context_chunks = retrieve(user_query)
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))

    system_prompt = f"""You are a legal assistant. Answer ONLY from the context below.
If the case is not in the context, say "This case is not in my database."
Do NOT use outside knowledge.

Context:
{context}"""

    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_query},
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()

# ── Ingest from CSV (uncomment to run once) ───────────────
# if __name__ == "__main__":
#     setup_db()
#     import pandas as pd
#     df = pd.read_csv("justice.csv")
#     df["combined"] = df.fillna("").apply(
#         lambda r: (
#             f"Case: {r['name']} | "
#             f"Parties: {r['first_party']} vs {r['second_party']} | "
#             f"Facts: {r['facts']} | "
#             f"Decision: {r['decision_type']} | "
#             f"Disposition: {r['disposition']} | "
#             f"Issue: {r['issue_area']}"
#         ),
#         axis=1
#     )
#     docs = df["combined"].tolist()
#     print(f"Loaded {len(docs)} cases")
#     ingest_documents(docs)

if __name__ == "__main__":
    setup_db()
    print("\n🦙 RAG Chat (Ctrl+C to quit)\n")
    while True:
        try:
            query = input(">>> ")
            if query.strip():
                rag_chat(query)
        except KeyboardInterrupt:
            print("\nBye!")
            break