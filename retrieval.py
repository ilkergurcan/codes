"""
Hybrid retrieval: dense (qwen3) + sparse (BM25) with RRF fusion.

Qdrant's Query API runs both searches server-side and fuses results
using Reciprocal Rank Fusion — no client-side score normalization needed.

Env vars: same as upload_to_qdrant.py
Dependencies: qdrant-client, httpx, fastembed
"""

import os
import re
import httpx
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

COLLECTION_NAME = "denizbot_excel"

client = QdrantClient(url=QDRANT_URL, port=6333, api_key=QDRANT_API_KEY)
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_dense_embedding(text: str) -> list[float]:
    response = httpx.post(
        url=EMBEDDING_URL,
        json={"model": "qwen3-embedding", "input": [text]},
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def get_sparse_embedding(text: str) -> models.SparseVector:
    emb = list(bm25_model.query_embed(text))[0]
    return models.SparseVector(
        indices=emb.indices.tolist(),
        values=emb.values.tolist(),
    )


# ---------------------------------------------------------------------------
# Clean query text
# ---------------------------------------------------------------------------

def clean_query(query: str) -> str:
    query = query.replace("\u00A0", " ").replace("\u200B", "")
    return re.sub(r"\s+", " ", query).strip()


# ---------------------------------------------------------------------------
# Hybrid retrieval with RRF
# ---------------------------------------------------------------------------

def retrieval(
    query: str,
    top_k: int = 20,
    prefetch_limit: int = 40,
    fusion: str = "rrf",
) -> tuple[str, list[str]]:
    """
    Hybrid search: dense + BM25 sparse, fused with RRF.

    Args:
        query:           User question
        top_k:           Final number of results after fusion
        prefetch_limit:  How many candidates each sub-search returns
                         (should be > top_k for better fusion quality)
        fusion:          "rrf" (Reciprocal Rank Fusion) or "dbsf"
                         (Distribution-Based Score Fusion)

    Returns:
        (combined_text, list_of_doc_names) — same interface as your original
    """
    query = clean_query(query)

    # Generate both embeddings for the query
    dense_vec = get_dense_embedding(query)
    sparse_vec = get_sparse_embedding(query)

    # Pick fusion strategy
    fusion_fn = (
        models.Fusion.RRF if fusion == "rrf"
        else models.Fusion.DBSF
    )

    # Qdrant runs both searches server-side and fuses results
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=sparse_vec,
                using="bm25",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=fusion_fn),
        limit=top_k,
        with_payload=True,
    ).points

    # Format results
    results = []
    for hit in search_results:
        results.append({
            "score": float(hit.score),
            "text": hit.payload.get("text"),
            "doc_name": hit.payload.get("doc_name"),
            "chunk_idx": hit.payload.get("chunk_idx"),
            "sheet_name": hit.payload.get("sheet_name"),
            "content_type": hit.payload.get("content_type"),
            "hierarchy": hit.payload.get("hierarchy"),
        })

    combined_text = "\n".join(res["text"] for res in results)
    doc_names = [res["doc_name"] for res in results]

    return combined_text, doc_names


# ---------------------------------------------------------------------------
# Convenience: dense-only or sparse-only (for comparison / debugging)
# ---------------------------------------------------------------------------

def retrieval_dense_only(query: str, top_k: int = 20) -> tuple[str, list[str]]:
    """Dense (semantic) search only — same as your original approach."""
    query = clean_query(query)
    dense_vec = get_dense_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vec,
        using="dense",
        limit=top_k,
        with_payload=True,
    ).points

    combined = "\n".join(hit.payload.get("text", "") for hit in results)
    docs = [hit.payload.get("doc_name", "") for hit in results]
    return combined, docs


def retrieval_sparse_only(query: str, top_k: int = 20) -> tuple[str, list[str]]:
    """BM25 (keyword) search only — useful for exact term matching."""
    query = clean_query(query)
    sparse_vec = get_sparse_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vec,
        using="bm25",
        limit=top_k,
        with_payload=True,
    ).points

    combined = "\n".join(hit.payload.get("text", "") for hit in results)
    docs = [hit.payload.get("doc_name", "") for hit in results]
    return combined, docs


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_query = "yıllık izin kaç gün"

    print("=== HYBRID (RRF) ===")
    text, docs = retrieval(test_query, top_k=5)
    print(text[:500])
    print(f"Sources: {docs}\n")

    print("=== DENSE ONLY ===")
    text, docs = retrieval_dense_only(test_query, top_k=5)
    print(text[:500])
    print(f"Sources: {docs}\n")

    print("=== SPARSE ONLY (BM25) ===")
    text, docs = retrieval_sparse_only(test_query, top_k=5)
    print(text[:500])
    print(f"Sources: {docs}")
