# Legal RAG — pgvector vs Qdrant

A local Retrieval-Augmented Generation system for querying US Supreme Court cases, built with two interchangeable vector store backends.

## Versions

| File                  | Vector Store          | Notes                              |
| --------------------- | --------------------- | ---------------------------------- |
| `pgvector-version.py` | PostgreSQL + pgvector | Requires running Postgres instance |
| `qdrant-version.py`   | Qdrant                | Requires running Qdrant container  |

Both use `all-MiniLM-L6-v2` for embeddings and `llama3.2:3b` via Ollama for generation.

## Setup

### Prerequisites

```bash
pip install qdrant-client sentence-transformers ollama psycopg2 pgvector pandas
```

### Qdrant (recommended)

```bash
docker run -d --name qdrant -p 6333:6333 \
  -v ~/.qdrant_storage:/qdrant/storage qdrant/qdrant
```

### pgvector

```bash
sudo apt install postgresql
sudo -u postgres psql -c "CREATE DATABASE ragdb;"
# then install pgvector extension
```

## Ingest Data

Uncomment the CSV ingestion block at the bottom of either file, then run once:

```bash
python qdrant-version.py   # or pgvector-version.py
```

## Chat

```bash
python qdrant-version.py
>>> Find cases related to civil rights
```

## Delete / Reset the Database

### Qdrant — delete the collection

```python
from qdrant_client import QdrantClient
client = QdrantClient("http://localhost:6333")
client.delete_collection("documents")
```

Or wipe everything by stopping the container and removing the storage volume:

```bash
docker stop qdrant && docker rm qdrant
rm -rf ~/.qdrant_storage
```

### PostgreSQL — drop the table or database

```bash
# drop just the documents table
sudo -u postgres psql -d ragdb -c "DROP TABLE documents;"

# or drop the entire database
sudo -u postgres psql -c "DROP DATABASE ragdb;"
```

## Stack

- **Embeddings**: `sentence-transformers` (local, no API key needed)
- **Vector stores**: Qdrant · pgvector
- **LLM**: Ollama (`llama3.2:3b`, runs fully offline)
- **Data**: [SCDB Supreme Court Database](http://scdb.wustl.edu/)
