import ollama
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Config ──────────────────────────────────────────────
DB_CONFIG = {
    "dbname": "ragdb",
    "user": "postgres",
    "password": "Z4m$Xp!tLq9W@7%K&r3JpF5T8",
    "host": "localhost",
    "port": 5432,
}
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, local
CHAT_MODEL  = "llama3.2:3b"
TOP_K       = 3

# ── Init ─────────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)

def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def setup_db():
    """Create table if not exists."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(384)
            );
        """)
    conn.commit()
    conn.close()
    print("✅ DB ready")

# ── Ingest ────────────────────────────────────────────────
def ingest_documents(docs: list[str]):
    """Embed and store a list of text chunks."""
    conn = get_conn()
    embeddings = embedder.encode(docs, show_progress_bar=True)
    with conn.cursor() as cur:
        for doc, emb in zip(docs, embeddings):
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (doc, emb.tolist())
            )
    conn.commit()
    conn.close()
    print(f"✅ Ingested {len(docs)} documents")

# ── Retrieve ──────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    query_emb = embedder.encode([query])[0].tolist()
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_emb, top_k))
        rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

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

# if __name__ == "__main__":
#     setup_db()

#     # ── Load justice.csv ──
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

    # chat only
    print("\n🦙 RAG Chat (Ctrl+C to quit)\n")
    while True:
        try:
            query = input(">>> ")
            if query.strip():
                rag_chat(query)
        except KeyboardInterrupt:
            print("\nBye!")
            break