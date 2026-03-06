"""
Excel → Embedding → Qdrant pipeline (v2 — hybrid search ready).

Stores BOTH dense (qwen3-embedding) and sparse (BM25) vectors per chunk.
Enables hybrid retrieval with RRF fusion at query time.

Env vars:
    QDRANT_URL          - Qdrant server URL
    QDRANT_API_KEY      - Qdrant API key
    EMBEDDING_URL       - Dense embedding endpoint
    EMBEDDING_API_KEY   - Dense embedding API key

Dependencies: qdrant-client, httpx, fastembed
"""

import os
import httpx
from uuid import uuid5, NAMESPACE_URL
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, SparseVector,
    SparseVectorParams, Modifier,
)

from excel_chunker_v3 import process_excel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

COLLECTION_NAME = "denizbot_excel"
DENSE_VECTOR_SIZE = 4096   # qwen3-embedding output dim
BATCH_SIZE = 100
FOLDER_PATH = "documents"

client = QdrantClient(url=QDRANT_URL, port=6333, api_key=QDRANT_API_KEY)

# BM25 sparse model — loaded once, runs locally, no API call needed
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")


# ---------------------------------------------------------------------------
# Dense embedding (your existing qwen3 endpoint)
# ---------------------------------------------------------------------------

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        url=EMBEDDING_URL,
        json={"model": "qwen3-embedding", "input": texts},
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


# ---------------------------------------------------------------------------
# Sparse embedding (BM25 via fastembed — local, fast)
# ---------------------------------------------------------------------------

def get_bm25_sparse_batch(texts: list[str]) -> list[SparseVector]:
    embeddings = list(bm25_model.embed(texts))
    return [
        SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
        for emb in embeddings
    ]


# ---------------------------------------------------------------------------
# Collection setup — dense + sparse
# ---------------------------------------------------------------------------

def ensure_collection(recreate: bool = False):
    """
    Create collection with both dense and sparse vector configs.
    Set recreate=True to wipe and rebuild (needed first time when switching
    from dense-only to hybrid schema).
    """
    config_kwargs = dict(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                modifier=Modifier.IDF,  # Qdrant computes IDF server-side
            ),
        },
    )

    if recreate:
        client.recreate_collection(**config_kwargs)
        print(f"Collection '{COLLECTION_NAME}' recreated with hybrid schema.")
        return

    try:
        client.create_collection(**config_kwargs)
        print(f"Collection '{COLLECTION_NAME}' created with hybrid schema.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Collection '{COLLECTION_NAME}' already exists.")
        else:
            raise


# ---------------------------------------------------------------------------
# Deterministic point IDs
# ---------------------------------------------------------------------------

def make_point_id(doc_path: str, chunk_idx: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"{doc_path}:{chunk_idx}"))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_and_upload(
    folder_path: str = FOLDER_PATH,
    recreate: bool = False,
):
    ensure_collection(recreate=recreate)

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

        texts = [c.text for c in chunks]

        # --- Embed & upload in batches ---
        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
            batch_chunks = chunks[batch_start : batch_start + BATCH_SIZE]

            # Dense embeddings (API call)
            try:
                dense_embeddings = get_embeddings_batch(batch_texts)
            except Exception as e:
                print(f"  [ERROR] Dense embedding failed at batch {batch_start}: {e}")
                continue

            # Sparse embeddings (local, fast)
            try:
                sparse_embeddings = get_bm25_sparse_batch(batch_texts)
            except Exception as e:
                print(f"  [ERROR] Sparse embedding failed at batch {batch_start}: {e}")
                continue

            # Build points with BOTH vector types
            points = []
            for i, (chunk, dense_vec, sparse_vec) in enumerate(
                zip(batch_chunks, dense_embeddings, sparse_embeddings)
            ):
                idx = batch_start + i
                points.append(PointStruct(
                    id=make_point_id(doc_path, idx),
                    vector={
                        "dense": dense_vec,
                        "bm25": sparse_vec,
                    },
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

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_uploaded += len(points)

        print(f"  Uploaded {len(chunks)} chunks (dense + sparse).")

    print(f"\nDone. Total points uploaded: {total_uploaded}")


if __name__ == "__main__":
    # First run: recreate=True to create hybrid schema
    # Subsequent runs: recreate=False to preserve data
    process_and_upload(recreate=True)
