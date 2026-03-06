"""
Excel → Embedding → Qdrant pipeline.

Processes all .xlsx files in a folder:
1. Chunks with excel_chunker_v3 (handles messy layouts, merges, footnotes)
2. Embeds with qwen3-embedding
3. Upserts to Qdrant in batches

Env vars required:
    QDRANT_URL          - Qdrant server URL
    QDRANT_API_KEY      - Qdrant API key
    EMBEDDING_URL       - Embedding service endpoint
    EMBEDDING_API_KEY   - Embedding service API key (separate from Qdrant)
"""

import os
import httpx
from uuid import uuid5, NAMESPACE_URL
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from excel_chunker_v3 import process_excel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

COLLECTION_NAME = "denizbot_excel"
VECTOR_SIZE = 4096          # qwen3-embedding output dim
BATCH_SIZE = 100
FOLDER_PATH = "documents"

client = QdrantClient(url=QDRANT_URL, port=6333, api_key=QDRANT_API_KEY)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    response = httpx.post(
        url=EMBEDDING_URL,
        json={"model": "qwen3-embedding", "input": [text]},
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed if your endpoint supports multiple inputs (qwen3 does)."""
    response = httpx.post(
        url=EMBEDDING_URL,
        json={"model": "qwen3-embedding", "input": texts},
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()["data"]
    # Sort by index to guarantee order
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


# ---------------------------------------------------------------------------
# Qdrant collection setup
# ---------------------------------------------------------------------------

def ensure_collection():
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")
        else:
            raise


# ---------------------------------------------------------------------------
# Point ID — deterministic so re-runs overwrite, not duplicate
# ---------------------------------------------------------------------------

def make_point_id(doc_path: str, chunk_idx: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"{doc_path}:{chunk_idx}"))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_and_upload(folder_path: str = FOLDER_PATH):
    ensure_collection()

    xlsx_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
    print(f"Found {len(xlsx_files)} .xlsx files in '{folder_path}'.\n")

    total_uploaded = 0

    for filename in xlsx_files:
        doc_path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}")

        # --- Chunk ---
        try:
            chunks = process_excel(doc_path, inline_footnotes=True)
        except Exception as e:
            print(f"  [ERROR] Chunking failed: {e}")
            continue

        if not chunks:
            print(f"  [SKIP] No chunks produced.")
            continue

        # --- Embed & upload in batches ---
        points = []
        texts = [c.text for c in chunks]

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
            batch_chunks = chunks[batch_start : batch_start + BATCH_SIZE]

            try:
                embeddings = get_embeddings_batch(batch_texts)
            except Exception as e:
                print(f"  [ERROR] Embedding failed at batch {batch_start}: {e}")
                continue

            for i, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                idx = batch_start + i
                points.append(PointStruct(
                    id=make_point_id(doc_path, idx),
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "doc_name": filename,
                        "chunk_idx": idx,
                        "sheet_name": chunk.metadata.get("sheet_name", ""),
                        "content_type": chunk.metadata.get("content_type", ""),
                        "hierarchy": chunk.metadata.get("hierarchy", []),
                        "row": chunk.metadata.get("row"),
                    },
                ))

            # Upsert this batch
            if points:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                total_uploaded += len(points)
                points = []

        print(f"  Uploaded {len(chunks)} chunks.")

    print(f"\nDone. Total points uploaded: {total_uploaded}")


if __name__ == "__main__":
    process_and_upload()
