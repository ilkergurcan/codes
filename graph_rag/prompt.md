# prompt.md — Graph RAG Implementation Guide

## Objective

Build a **Graph RAG layer** on top of an existing chunking + vector RAG pipeline. The graph captures entities, relationships, and chunk-to-chunk connections extracted from document chunks, enabling multi-hop reasoning and relationship-aware retrieval.

---

## Prerequisites (Already Available)

You have:
- **Document chunkers** for PDF, DOCX, XLSX that produce chunks with `chunk_id`, `text`, `metadata`.
- **Qwen LLM** via vLLM (OpenAI-compatible API).
- **Embedding model** (OpenAI-compatible `/v1/embeddings`).
- **Qdrant** vector database with indexed chunks.

You need to install: `networkx`, `python-louvain` (imports as `community`), `httpx`, `pydantic`, `tqdm` (all pip-installable, no GPU required).

---

## Step 1: Define Data Models

Create Pydantic models that represent the graph schema. These are the building blocks.

```python
# models.py
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    PRODUCT = "PRODUCT"
    REGULATION = "REGULATION"
    PROCESS = "PROCESS"
    SYSTEM = "SYSTEM"
    LOCATION = "LOCATION"
    DATE = "DATE"
    METRIC = "METRIC"
    CONCEPT = "CONCEPT"
    OTHER = "OTHER"

class Entity(BaseModel):
    name: str = Field(..., description="Normalized entity name (lowercase, trimmed)")
    entity_type: EntityType
    description: Optional[str] = Field(None, description="Brief description if available")

class Relationship(BaseModel):
    source: str = Field(..., description="Source entity name (must match an Entity.name)")
    target: str = Field(..., description="Target entity name (must match an Entity.name)")
    relation: str = Field(..., description="Relationship label, e.g. 'manages', 'contains', 'regulates'")
    description: Optional[str] = None

class ExtractionResult(BaseModel):
    entities: list[Entity] = []
    relationships: list[Relationship] = []
```

---

## Step 2: LLM-Based Entity & Relationship Extraction

Send each chunk to Qwen with a structured extraction prompt. This is the core of Graph RAG.

### Extraction Prompt Template

```
You are an expert entity and relationship extractor for banking/financial documents.

Given the following text chunk, extract:
1. All named entities (people, organizations, products, regulations, systems, processes, metrics, concepts).
2. All relationships between extracted entities.

Rules:
- Normalize entity names to lowercase.
- Merge near-duplicates (e.g., "denizbank" and "deniz bank" → "denizbank").
- Only extract relationships where BOTH source and target are in your entity list.
- Use concise, verb-based relationship labels (e.g., "manages", "reports_to", "contains", "regulates", "depends_on").
- If the chunk has no meaningful entities, return empty lists.

Respond ONLY with valid JSON matching this schema (no markdown, no preamble):
{
  "entities": [{"name": "...", "entity_type": "...", "description": "..."}],
  "relationships": [{"source": "...", "target": "...", "relation": "...", "description": "..."}]
}

Valid entity_type values: PERSON, ORGANIZATION, PRODUCT, REGULATION, PROCESS, SYSTEM, LOCATION, DATE, METRIC, CONCEPT, OTHER

--- TEXT CHUNK ---
{chunk_text}
```

### Extraction Function

```python
import httpx
import json
from models import ExtractionResult
from config import QWEN_BASE_URL, QWEN_MODEL_NAME, EXTRACTION_MAX_TOKENS, EXTRACTION_TEMPERATURE

def extract_entities_and_relations(chunk_text: str, chunk_id: str) -> ExtractionResult | None:
    """Extract entities and relationships from a single chunk via Qwen."""
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": EXTRACTION_MAX_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
    }

    try:
        resp = httpx.post(f"{QWEN_BASE_URL}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Truncation guard
        finish_reason = data["choices"][0].get("finish_reason", "")
        if finish_reason == "length":
            print(f"[WARN] chunk_id={chunk_id} — LLM output truncated. Skipping.")
            return None

        raw = data["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = ExtractionResult.model_validate_json(raw)
        return result

    except (httpx.HTTPError, json.JSONDecodeError, Exception) as e:
        print(f"[ERROR] chunk_id={chunk_id} — {type(e).__name__}: {e}")
        return None
```

### Batch Extraction

```python
from tqdm import tqdm

def extract_all(chunks: list[dict]) -> dict[str, ExtractionResult]:
    """Run extraction on all chunks. Returns {chunk_id: ExtractionResult}."""
    results = {}
    for chunk in tqdm(chunks, desc="Extracting entities"):
        cid = chunk["chunk_id"]
        res = extract_entities_and_relations(chunk["text"], cid)
        if res and (res.entities or res.relationships):
            results[cid] = res
    return results
```

---

## Step 3: Build the Knowledge Graph

Construct a NetworkX MultiDiGraph from extraction results.

### Node Types

| Node Type   | Attributes                                          |
|-------------|-----------------------------------------------------|
| `chunk`     | `chunk_id`, `source_file`, `page`, `section`, `text_preview` (first 200 chars) |
| `entity`    | `name`, `entity_type`, `description`, `chunk_ids` (list of chunks mentioning it) |

### Edge Types

| Edge Type               | From       | To        | Attributes          |
|--------------------------|------------|-----------|---------------------|
| `MENTIONS`              | chunk      | entity    | —                   |
| `RELATION`              | entity     | entity    | `relation`, `description`, `source_chunk_id` |
| `CO_OCCURS_IN`          | chunk      | chunk     | `shared_entities` (list of entity names appearing in both) |
| `SAME_DOCUMENT`         | chunk      | chunk     | `source_file`       |

### Graph Builder

```python
import networkx as nx
from collections import defaultdict

def build_graph(
    chunks: list[dict],
    extractions: dict[str, "ExtractionResult"],
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    entity_to_chunks: dict[str, list[str]] = defaultdict(list)

    # --- Add chunk nodes ---
    for chunk in chunks:
        cid = chunk["chunk_id"]
        G.add_node(cid, node_type="chunk", **{
            "source_file": chunk["metadata"].get("source_file", ""),
            "page": chunk["metadata"].get("page", ""),
            "section": chunk["metadata"].get("section", ""),
            "text_preview": chunk["text"][:200],
        })

    # --- Add entity nodes & MENTIONS edges ---
    for cid, result in extractions.items():
        for ent in result.entities:
            ent_key = f"entity::{ent.name}"
            if not G.has_node(ent_key):
                G.add_node(ent_key, node_type="entity",
                           entity_type=ent.entity_type.value,
                           description=ent.description or "")
            G.add_edge(cid, ent_key, edge_type="MENTIONS")
            entity_to_chunks[ent.name].append(cid)

        # --- Add RELATION edges ---
        for rel in result.relationships:
            src_key = f"entity::{rel.source}"
            tgt_key = f"entity::{rel.target}"
            if G.has_node(src_key) and G.has_node(tgt_key):
                G.add_edge(src_key, tgt_key, edge_type="RELATION",
                           relation=rel.relation,
                           description=rel.description or "",
                           source_chunk_id=cid)

    # --- Add CO_OCCURS_IN edges (chunks sharing entities) ---
    for ent_name, cids in entity_to_chunks.items():
        if len(cids) < 2:
            continue
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                if not G.has_edge(cids[i], cids[j], key=f"co_occur_{ent_name}"):
                    G.add_edge(cids[i], cids[j],
                               key=f"co_occur_{ent_name}",
                               edge_type="CO_OCCURS_IN",
                               shared_entity=ent_name)

    # --- Add SAME_DOCUMENT edges ---
    doc_groups = defaultdict(list)
    for chunk in chunks:
        doc_groups[chunk["metadata"].get("source_file", "unknown")].append(chunk["chunk_id"])
    for doc, cids in doc_groups.items():
        for i in range(len(cids) - 1):
            G.add_edge(cids[i], cids[i + 1], edge_type="SAME_DOCUMENT", source_file=doc)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
```

---

## Step 4: Graph Persistence & Export

```python
import pickle
import json

def save_graph(G: nx.MultiDiGraph, path: str = "knowledge_graph.pkl"):
    with open(path, "wb") as f:
        pickle.dump(G, f)

def load_graph(path: str = "knowledge_graph.pkl") -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)

def export_graph_json(G: nx.MultiDiGraph, path: str = "knowledge_graph.json"):
    """Export graph to JSON for visualization. Produces {nodes: [...], edges: [...]}."""
    nodes = []
    for nid, attrs in G.nodes(data=True):
        nodes.append({"id": nid, **attrs})

    edges = []
    for src, tgt, key, attrs in G.edges(data=True, keys=True):
        edges.append({"source": src, "target": tgt, "key": str(key), **attrs})

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(nodes)} nodes, {len(edges)} edges to {path}")
```

---

## Step 5: Hybrid Retriever

This is the retrieval logic that combines vector search, graph traversal, and community context.

```python
import httpx
import json
import networkx as nx
from config import (
    QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_BASE_URL,
    EMBEDDING_MODEL_NAME, RETRIEVAL_TOP_K, GRAPH_TRAVERSAL_DEPTH,
    QWEN_BASE_URL, QWEN_MODEL_NAME,
)

def embed_query(query: str) -> list[float]:
    resp = httpx.post(f"{EMBEDDING_BASE_URL}/embeddings", json={
        "model": EMBEDDING_MODEL_NAME,
        "input": query,
    }, timeout=30)
    return resp.json()["data"][0]["embedding"]

def vector_search(query_vector: list[float], top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    resp = httpx.post(f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search", json={
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
    }, timeout=30)
    return resp.json().get("result", [])

def graph_search(
    G: nx.MultiDiGraph,
    query_entities: list[str],
    depth: int = GRAPH_TRAVERSAL_DEPTH,
) -> set[str]:
    """BFS from query entities, collecting connected chunk_ids."""
    chunk_ids = set()
    for ent_name in query_entities:
        ent_key = f"entity::{ent_name}"
        if ent_key not in G:
            continue
        visited = set()
        queue = [(ent_key, 0)]
        while queue:
            node, d = queue.pop(0)
            if node in visited or d > depth:
                continue
            visited.add(node)
            node_data = G.nodes[node]
            if node_data.get("node_type") == "chunk":
                chunk_ids.add(node)
            if d < depth:
                for neighbor in G.neighbors(node):
                    queue.append((neighbor, d + 1))
                for predecessor in G.predecessors(node):
                    queue.append((predecessor, d + 1))
    return chunk_ids

def community_search(
    G: nx.MultiDiGraph,
    query_entities: list[str],
    community_summaries: dict[int, str],
) -> list[str]:
    """Find community summaries relevant to the query entities."""
    relevant_community_ids = set()
    for ent_name in query_entities:
        ent_key = f"entity::{ent_name}"
        if ent_key in G and "community_id" in G.nodes[ent_key]:
            relevant_community_ids.add(G.nodes[ent_key]["community_id"])

    summaries = []
    for cid in relevant_community_ids:
        if cid in community_summaries:
            summaries.append(f"[Community {cid} Summary]\n{community_summaries[cid]}")
    return summaries

def extract_query_entities(query: str) -> list[str]:
    """Use LLM to extract entity names from the user query."""
    prompt = f"""Extract entity names from this query. Return ONLY a JSON list of lowercase strings.
If no entities found, return [].

Query: {query}"""

    resp = httpx.post(f"{QWEN_BASE_URL}/chat/completions", json={
        "model": QWEN_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    }, timeout=30)

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)

def hybrid_retrieve(
    query: str,
    G: nx.MultiDiGraph,
    chunks_lookup: dict[str, dict],
    community_summaries: dict[int, str] | None = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> tuple[list[dict], list[str]]:
    """Combine vector search + graph traversal + community context.
    
    Returns:
        (context_chunks, community_context_strings)
    """
    # 1. Vector path
    qvec = embed_query(query)
    vector_results = vector_search(qvec, top_k=top_k)
    vector_chunk_ids = {r["payload"]["chunk_id"] for r in vector_results if "payload" in r}
    vector_scores = {r["payload"]["chunk_id"]: r["score"] for r in vector_results if "payload" in r}

    # 2. Graph path
    query_entities = extract_query_entities(query)
    graph_chunk_ids = graph_search(G, query_entities)

    # 3. Community path
    comm_context = []
    if community_summaries:
        comm_context = community_search(G, query_entities, community_summaries)

    # 4. Merge & rank
    all_ids = vector_chunk_ids | graph_chunk_ids
    ranked = []
    for cid in all_ids:
        score = vector_scores.get(cid, 0.0)
        if cid in graph_chunk_ids:
            score += 0.15
        if cid in vector_chunk_ids and cid in graph_chunk_ids:
            score += 0.10
        # Community membership boost
        if cid in G and "community_id" in G.nodes.get(cid, {}):
            comm_id = G.nodes[cid]["community_id"]
            if comm_id in (community_summaries or {}):
                score += 0.05
        ranked.append({"chunk_id": cid, "score": score})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = ranked[:top_k]

    # 5. Assemble context
    context_chunks = []
    for item in top_chunks:
        chunk_data = chunks_lookup.get(item["chunk_id"])
        if chunk_data:
            context_chunks.append({
                **item,
                "text": chunk_data["text"],
                "metadata": chunk_data["metadata"],
            })
    return context_chunks, comm_context
```

---

## Step 6: Answer Generation

```python
def generate_answer(
    query: str,
    context_chunks: list[dict],
    community_context: list[str] | None = None,
) -> str:
    """Send retrieved context + community summaries + query to Qwen."""
    context_block = "\n\n".join(
        f"[Source {i+1}: {c['metadata'].get('source_file', '?')}, "
        f"Page {c['metadata'].get('page', '?')}]\n{c['text']}"
        for i, c in enumerate(context_chunks)
    )

    # Append community summaries as additional context
    if community_context:
        context_block += "\n\n--- TOPIC SUMMARIES ---\n"
        context_block += "\n\n".join(community_context)

    system_prompt = """You are a precise document assistant for DenizBank internal documents.
Answer the user's question using ONLY the provided source documents and topic summaries below.
If the information is not in the sources, say so explicitly.
Always cite which source(s) you used (e.g., [Source 1], [Source 3]).
When topic summaries provide useful context, reference them as [Community N Summary].
Respond in the same language as the user's question."""

    user_prompt = f"""--- SOURCE DOCUMENTS ---
{context_block}

--- QUESTION ---
{query}"""

    resp = httpx.post(f"{QWEN_BASE_URL}/chat/completions", json={
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
    }, timeout=120)

    return resp.json()["choices"][0]["message"]["content"]
```

---

## Step 7: Visualization Export

To see which chunks connect to which chunks (and why), export a filtered subgraph:

```python
def export_chunk_connections(G: nx.MultiDiGraph, path: str = "chunk_connections.json"):
    """Export only chunk-to-chunk connections with their linking entities."""
    connections = []
    for src, tgt, key, attrs in G.edges(data=True, keys=True):
        src_type = G.nodes[src].get("node_type")
        tgt_type = G.nodes[tgt].get("node_type")

        if src_type == "chunk" and tgt_type == "chunk":
            connections.append({
                "source_chunk": src,
                "target_chunk": tgt,
                "edge_type": attrs.get("edge_type", ""),
                "shared_entity": attrs.get("shared_entity", ""),
                "source_file": attrs.get("source_file", ""),
            })

    chunk_nodes = [
        {"id": nid, **{k: v for k, v in attrs.items() if k != "text_preview"}}
        for nid, attrs in G.nodes(data=True) if attrs.get("node_type") == "chunk"
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunk_nodes, "connections": connections}, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(chunk_nodes)} chunks, {len(connections)} connections")
```

---

## Step 8: Community Detection & Summarization

Louvain clustering identifies topical communities in the graph. LLM-generated summaries of each community enable "global search" — answering broad thematic queries without retrieving individual chunks.

### Community Detection

```python
# community.py
import networkx as nx
import community as community_louvain  # pip install python-louvain
import httpx
from collections import defaultdict
from config import (
    COMMUNITY_RESOLUTION, COMMUNITY_SUMMARY_MAX_TOKENS,
    COMMUNITY_MIN_SIZE, QWEN_BASE_URL, QWEN_MODEL_NAME,
)

def detect_communities(G: nx.MultiDiGraph, resolution: float = COMMUNITY_RESOLUTION) -> dict[str, int]:
    """Run Louvain on the undirected projection. Returns {node_id: community_id}."""
    # Louvain needs undirected, no multi-edges
    G_undirected = nx.Graph(G.to_undirected())
    
    partition = community_louvain.best_partition(G_undirected, resolution=resolution, random_state=42)
    
    # Write community_id back to the original directed graph
    for node_id, comm_id in partition.items():
        G.nodes[node_id]["community_id"] = comm_id
    
    # Stats
    comm_sizes = defaultdict(int)
    for comm_id in partition.values():
        comm_sizes[comm_id] += 1
    
    print(f"Detected {len(comm_sizes)} communities")
    print(f"  Largest: {max(comm_sizes.values())} nodes")
    print(f"  Smallest: {min(comm_sizes.values())} nodes")
    print(f"  Median: {sorted(comm_sizes.values())[len(comm_sizes)//2]} nodes")
    
    return partition


def get_community_members(G: nx.MultiDiGraph, partition: dict[str, int]) -> dict[int, dict]:
    """Group nodes by community, separating chunks and entities."""
    communities = defaultdict(lambda: {"chunks": [], "entities": []})
    
    for node_id, comm_id in partition.items():
        node_data = G.nodes.get(node_id, {})
        if node_data.get("node_type") == "chunk":
            communities[comm_id]["chunks"].append({
                "id": node_id,
                "source_file": node_data.get("source_file", ""),
                "section": node_data.get("section", ""),
                "text_preview": node_data.get("text_preview", ""),
            })
        elif node_data.get("node_type") == "entity":
            communities[comm_id]["entities"].append({
                "id": node_id,
                "entity_type": node_data.get("entity_type", ""),
                "description": node_data.get("description", ""),
            })
    
    return dict(communities)
```

### Community Summarization via LLM

```python
def summarize_community(community_data: dict, community_id: int) -> str | None:
    """Generate an LLM summary for a single community."""
    chunks = community_data["chunks"]
    entities = community_data["entities"]
    
    if len(chunks) + len(entities) < COMMUNITY_MIN_SIZE:
        return None
    
    # Build context for summarization
    entity_list = ", ".join(
        f"{e['id'].replace('entity::', '')} ({e['entity_type']})"
        for e in entities[:30]  # cap to avoid prompt overflow
    )
    
    chunk_texts = "\n---\n".join(
        f"[{c['source_file']} / {c['section']}]: {c['text_preview']}"
        for c in chunks[:20]  # cap
    )
    
    prompt = f"""You are summarizing a thematic cluster of related documents from a banking knowledge base.

This cluster contains {len(chunks)} document chunks and {len(entities)} entities.

Key entities in this cluster: {entity_list}

Representative text excerpts:
{chunk_texts}

Write a concise summary (3-5 sentences) that captures:
1. The main topic/theme of this cluster
2. Key entities and their roles
3. How the documents in this cluster relate to each other

Respond with the summary only, no preamble."""

    try:
        resp = httpx.post(f"{QWEN_BASE_URL}/chat/completions", json={
            "model": QWEN_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": COMMUNITY_SUMMARY_MAX_TOKENS,
            "temperature": 0.2,
        }, timeout=60)
        
        finish_reason = resp.json()["choices"][0].get("finish_reason", "")
        if finish_reason == "length":
            print(f"[WARN] Community {community_id} summary truncated")
        
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] Community {community_id} summarization failed: {e}")
        return None


def summarize_all_communities(
    G: nx.MultiDiGraph,
    partition: dict[str, int],
) -> dict[int, str]:
    """Generate summaries for all communities above minimum size."""
    from tqdm import tqdm
    
    members = get_community_members(G, partition)
    summaries = {}
    
    for comm_id, data in tqdm(members.items(), desc="Summarizing communities"):
        summary = summarize_community(data, comm_id)
        if summary:
            summaries[comm_id] = summary
    
    print(f"Generated {len(summaries)} community summaries "
          f"(skipped {len(members) - len(summaries)} below min_size={COMMUNITY_MIN_SIZE})")
    return summaries
```

---

## Step 9: Incremental Graph Updates

When new documents arrive, only process the new chunks instead of rebuilding everything.

```python
# incremental.py
import json
import os
import networkx as nx
from pathlib import Path
from extraction import extract_all
from graph_builder import build_graph
from graph_export import save_graph
from community import detect_communities, summarize_all_communities, get_community_members, summarize_community
from config import GRAPH_PICKLE_PATH, PROCESSED_CHUNKS_LOG


def load_processed_ids(log_path: str = PROCESSED_CHUNKS_LOG) -> set[str]:
    """Load the set of already-processed chunk_ids."""
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r") as f:
        return set(json.load(f))


def save_processed_ids(ids: set[str], log_path: str = PROCESSED_CHUNKS_LOG):
    """Persist the set of processed chunk_ids."""
    with open(log_path, "w") as f:
        json.dump(sorted(ids), f)


def get_new_chunks(all_chunks: list[dict], processed_ids: set[str]) -> list[dict]:
    """Filter to only unprocessed chunks."""
    new = [c for c in all_chunks if c["chunk_id"] not in processed_ids]
    print(f"Total chunks: {len(all_chunks)}, Already processed: {len(processed_ids)}, New: {len(new)}")
    return new


def merge_into_graph(
    existing_G: nx.MultiDiGraph,
    new_chunks: list[dict],
    new_extractions: dict,
) -> nx.MultiDiGraph:
    """Merge new nodes and edges into the existing graph.
    
    Rules:
    - New chunk nodes are added (should not exist already if diff is correct)
    - New entity nodes are deduplicated: if entity::name already exists, 
      just add the MENTIONS edge from the new chunk
    - New RELATION edges are added even if both entities existed
    - CO_OCCURS_IN edges are recomputed for entities that gained new chunks
    - SAME_DOCUMENT edges are added for new chunks in existing documents
    """
    from collections import defaultdict
    
    G = existing_G  # mutate in place
    
    entity_to_chunks: dict[str, list[str]] = defaultdict(list)
    
    # Collect existing entity->chunk mappings
    for src, tgt, data in G.edges(data=True):
        if data.get("edge_type") == "MENTIONS":
            src_data = G.nodes.get(src, {})
            tgt_data = G.nodes.get(tgt, {})
            if src_data.get("node_type") == "chunk" and tgt_data.get("node_type") == "entity":
                ent_name = tgt.replace("entity::", "")
                entity_to_chunks[ent_name].append(src)
    
    # Add new chunk nodes
    for chunk in new_chunks:
        cid = chunk["chunk_id"]
        if G.has_node(cid):
            continue
        G.add_node(cid, node_type="chunk", **{
            "source_file": chunk["metadata"].get("source_file", ""),
            "page": chunk["metadata"].get("page", ""),
            "section": chunk["metadata"].get("section", ""),
            "text_preview": chunk["text"][:200],
        })
    
    # Add entities & edges from new extractions
    for cid, result in new_extractions.items():
        for ent in result.entities:
            ent_key = f"entity::{ent.name}"
            if not G.has_node(ent_key):
                G.add_node(ent_key, node_type="entity",
                           entity_type=ent.entity_type.value,
                           description=ent.description or "")
            G.add_edge(cid, ent_key, edge_type="MENTIONS")
            entity_to_chunks[ent.name].append(cid)
        
        for rel in result.relationships:
            src_key = f"entity::{rel.source}"
            tgt_key = f"entity::{rel.target}"
            if G.has_node(src_key) and G.has_node(tgt_key):
                G.add_edge(src_key, tgt_key, edge_type="RELATION",
                           relation=rel.relation,
                           description=rel.description or "",
                           source_chunk_id=cid)
    
    # Recompute CO_OCCURS_IN edges for entities that have new chunks
    affected_entities = set()
    for cid in new_extractions:
        for ent in new_extractions[cid].entities:
            affected_entities.add(ent.name)
    
    for ent_name in affected_entities:
        cids = entity_to_chunks.get(ent_name, [])
        if len(cids) < 2:
            continue
        # Only add edges involving at least one NEW chunk
        new_cids = set(new_extractions.keys())
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                if cids[i] in new_cids or cids[j] in new_cids:
                    key = f"co_occur_{ent_name}"
                    if not G.has_edge(cids[i], cids[j], key=key):
                        G.add_edge(cids[i], cids[j], key=key,
                                   edge_type="CO_OCCURS_IN", shared_entity=ent_name)
    
    # Add SAME_DOCUMENT edges for new chunks
    doc_groups = defaultdict(list)
    for chunk in new_chunks:
        doc_groups[chunk["metadata"].get("source_file", "unknown")].append(chunk["chunk_id"])
    
    # Also include existing chunks from the same documents
    for doc in doc_groups:
        existing_in_doc = [
            n for n, d in G.nodes(data=True)
            if d.get("node_type") == "chunk" and d.get("source_file") == doc
            and n not in set(doc_groups[doc])
        ]
        all_in_doc = existing_in_doc + doc_groups[doc]
        # Connect new chunks to their neighbors in the document
        for i in range(len(all_in_doc) - 1):
            if all_in_doc[i] in set(doc_groups[doc]) or all_in_doc[i + 1] in set(doc_groups[doc]):
                if not G.has_edge(all_in_doc[i], all_in_doc[i + 1]):
                    G.add_edge(all_in_doc[i], all_in_doc[i + 1],
                               edge_type="SAME_DOCUMENT", source_file=doc)
    
    print(f"After merge: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def incremental_update(
    all_chunks: list[dict],
    existing_graph_path: str = GRAPH_PICKLE_PATH,
    rerun_communities: bool = True,
) -> tuple[nx.MultiDiGraph, dict[int, str] | None]:
    """Full incremental update pipeline.
    
    Returns:
        (updated_graph, community_summaries_or_None)
    """
    from graph_export import load_graph
    
    processed_ids = load_processed_ids()
    new_chunks = get_new_chunks(all_chunks, processed_ids)
    
    if not new_chunks:
        print("No new chunks to process.")
        G = load_graph(existing_graph_path)
        return G, None
    
    # Load or create graph
    if os.path.exists(existing_graph_path):
        G = load_graph(existing_graph_path)
        print(f"Loaded existing graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        G = nx.MultiDiGraph()
        print("Starting with empty graph")
    
    # Extract entities from new chunks only
    new_extractions = extract_all(new_chunks)
    
    # Merge into existing graph
    G = merge_into_graph(G, new_chunks, new_extractions)
    
    # Update processed IDs
    new_ids = {c["chunk_id"] for c in new_chunks}
    save_processed_ids(processed_ids | new_ids)
    
    # Re-run community detection on the full graph
    community_summaries = None
    if rerun_communities:
        partition = detect_communities(G)
        
        # Only regenerate summaries for communities with new members
        old_partition_path = existing_graph_path.replace(".pkl", "_partition.json")
        if os.path.exists(old_partition_path):
            with open(old_partition_path, "r") as f:
                old_partition = json.load(f)
            changed_communities = set()
            for node_id in new_ids:
                if node_id in partition:
                    changed_communities.add(partition[node_id])
            print(f"Re-summarizing {len(changed_communities)} affected communities")
            
            # Load old summaries, regenerate only changed ones
            summ_path = existing_graph_path.replace(".pkl", "_summaries.json")
            old_summaries = {}
            if os.path.exists(summ_path):
                with open(summ_path, "r") as f:
                    old_summaries = {int(k): v for k, v in json.load(f).items()}
            
            members = get_community_members(G, partition)
            for comm_id in changed_communities:
                if comm_id in members:
                    s = summarize_community(members[comm_id], comm_id)
                    if s:
                        old_summaries[comm_id] = s
            community_summaries = old_summaries
        else:
            community_summaries = summarize_all_communities(G, partition)
        
        # Persist partition
        with open(old_partition_path, "w") as f:
            json.dump({str(k): v for k, v in partition.items()}, f)
        
        # Persist summaries
        summ_path = existing_graph_path.replace(".pkl", "_summaries.json")
        with open(summ_path, "w") as f:
            json.dump({str(k): v for k, v in community_summaries.items()}, f, ensure_ascii=False, indent=2)
    
    # Save updated graph
    save_graph(G, existing_graph_path)
    
    return G, community_summaries


def remove_document(
    G: nx.MultiDiGraph,
    source_file: str,
) -> nx.MultiDiGraph:
    """Remove all chunks (and orphaned entities) from a specific document."""
    chunks_to_remove = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "chunk" and d.get("source_file") == source_file
    ]
    
    G.remove_nodes_from(chunks_to_remove)
    
    # Remove orphaned entity nodes (no remaining MENTIONS edges)
    orphaned = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "entity" and G.degree(n) == 0
    ]
    G.remove_nodes_from(orphaned)
    
    print(f"Removed {len(chunks_to_remove)} chunks + {len(orphaned)} orphaned entities for '{source_file}'")
    return G
```

---

## Step 10: Graph Visualization (Self-Contained HTML)

Generate a standalone HTML file that renders the graph as a D3.js force-directed visualization. Open it in any browser — no server needed.

```python
# visualize_graph.py
import json
import networkx as nx
from pathlib import Path

def generate_graph_html(
    G: nx.MultiDiGraph,
    output_path: str = "graph_viewer.html",
    max_nodes: int = 500,
    title: str = "Graph RAG — Knowledge Graph Viewer",
):
    """Generate a self-contained HTML file with D3.js graph visualization.
    
    Args:
        G: The knowledge graph
        output_path: Where to write the HTML file
        max_nodes: If graph exceeds this, sample the most-connected subgraph
        title: Page title
    """
    # Sample if too large
    if G.number_of_nodes() > max_nodes:
        print(f"Graph has {G.number_of_nodes()} nodes, sampling top {max_nodes} by degree")
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
    
    # Build JSON data
    nodes = []
    for nid, attrs in G.nodes(data=True):
        node = {"id": nid, "node_type": attrs.get("node_type", "unknown")}
        if attrs.get("node_type") == "chunk":
            node["source_file"] = attrs.get("source_file", "")
            node["page"] = str(attrs.get("page", ""))
            node["section"] = attrs.get("section", "")
            node["text_preview"] = attrs.get("text_preview", "")
        elif attrs.get("node_type") == "entity":
            node["entity_type"] = attrs.get("entity_type", "")
            node["description"] = attrs.get("description", "")
        if "community_id" in attrs:
            node["community_id"] = attrs["community_id"]
        nodes.append(node)
    
    edges = []
    for src, tgt, key, attrs in G.edges(data=True, keys=True):
        edge = {
            "source": src,
            "target": tgt,
            "edge_type": attrs.get("edge_type", ""),
        }
        if attrs.get("shared_entity"):
            edge["shared_entity"] = attrs["shared_entity"]
        if attrs.get("relation"):
            edge["relation"] = attrs["relation"]
        if attrs.get("source_file"):
            edge["source_file"] = attrs["source_file"]
        edges.append(edge)
    
    graph_data = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    
    html = _GRAPH_HTML_TEMPLATE.replace("__GRAPH_DATA__", graph_data).replace("__TITLE__", title)
    
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Visualization saved to {output_path} ({len(nodes)} nodes, {len(edges)} edges)")


_GRAPH_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: system-ui,-apple-system,sans-serif; background:#0f1117; color:#e0e0e0; }
  #controls { padding:12px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap;
              background:#181b23; border-bottom:1px solid #2a2d38; }
  #controls label { font-size:12px; color:#999; }
  #controls select { font-size:13px; padding:4px 8px; background:#1e2130; color:#ccc;
                     border:1px solid #2a2d38; border-radius:4px; }
  #graph-container { width:100%; height:calc(100vh - 140px); }
  #detail { padding:10px 16px; background:#181b23; border-top:1px solid #2a2d38;
            font-size:12px; min-height:48px; max-height:140px; overflow-y:auto; }
  #detail strong { color:#60a5fa; }
  .legend { display:flex; gap:14px; align-items:center; }
  .legend-item { display:flex; align-items:center; gap:4px; font-size:11px; color:#888; }
  .legend-dot { width:10px; height:10px; border-radius:50%; }
  #tooltip { position:absolute; display:none; background:#1e2130; border:1px solid #2a2d38;
             border-radius:6px; padding:8px 12px; font-size:11px; color:#ccc;
             pointer-events:none; max-width:260px; z-index:100; }
  #search-box { font-size:13px; padding:4px 10px; background:#1e2130; color:#ccc;
                border:1px solid #2a2d38; border-radius:4px; width:180px; }
</style>
</head>
<body>
<div id="controls">
  <label>Edge type</label>
  <select id="edge-filter">
    <option value="all">All</option>
    <option value="CO_OCCURS_IN">Co-occurs</option>
    <option value="MENTIONS">Mentions</option>
    <option value="SAME_DOCUMENT">Same doc</option>
    <option value="RELATION">Relations</option>
  </select>
  <label>Node type</label>
  <select id="node-filter">
    <option value="all">All</option>
    <option value="chunk">Chunks only</option>
    <option value="entity">Entities only</option>
  </select>
  <label>Color by</label>
  <select id="color-mode">
    <option value="type">Node type</option>
    <option value="community">Community</option>
    <option value="document">Document</option>
  </select>
  <input type="text" id="search-box" placeholder="Search nodes...">
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#60a5fa"></div>Chunk</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f87171"></div>Entity</div>
    <div class="legend-item"><div class="legend-dot" style="background:#5eead4;width:20px;height:2px;border-radius:1px"></div>Mentions</div>
    <div class="legend-item"><div class="legend-dot" style="background:#fbbf24;width:20px;height:2px;border-radius:1px"></div>Co-occurs</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f9a8d4;width:20px;height:2px;border-radius:1px;border:none;border-top:2px dashed #f9a8d4"></div>Same doc</div>
    <div class="legend-item"><div class="legend-dot" style="background:#c4b5fd;width:20px;height:2px;border-radius:1px"></div>Relation</div>
  </div>
</div>
<div id="graph-container"></div>
<div id="detail">Click any node to inspect its connections and metadata.</div>
<div id="tooltip"></div>
<script>
const DATA = __GRAPH_DATA__;
const edgeColors = {MENTIONS:"#5eead4",CO_OCCURS_IN:"#fbbf24",SAME_DOCUMENT:"#f9a8d4",RELATION:"#c4b5fd"};
const entityColors = {ORGANIZATION:"#f87171",PRODUCT:"#fb923c",REGULATION:"#a78bfa",METRIC:"#34d399",
                      CONCEPT:"#60a5fa",SYSTEM:"#fbbf24",PERSON:"#f472b6",LOCATION:"#818cf8",
                      PROCESS:"#22d3ee",DATE:"#a3a3a3",OTHER:"#6b7280"};
const communityPalette = ["#60a5fa","#f87171","#34d399","#fbbf24","#a78bfa","#f472b6",
                          "#22d3ee","#fb923c","#818cf8","#a3e635","#e879f9","#38bdf8"];
const container = document.getElementById('graph-container');
const w = container.clientWidth, h = container.clientHeight || 600;
const svg = d3.select(container).append('svg').attr('width',w).attr('height',h);
const g = svg.append('g');
svg.call(d3.zoom().scaleExtent([0.1,6]).on('zoom',e=>g.attr('transform',e.transform)));

let nodes = DATA.nodes.map(d=>({...d}));
let edges = DATA.edges.map(d=>({...d}));
const nodeMap = Object.fromEntries(nodes.map(n=>[n.id,n]));

// Assign doc colors
const docSet = [...new Set(nodes.filter(n=>n.node_type==='chunk').map(n=>n.source_file))];
const docPalette = ["#60a5fa","#38bdf8","#818cf8","#22d3ee","#34d399","#a78bfa","#f472b6","#fbbf24"];
const docColors = Object.fromEntries(docSet.map((d,i)=>[d,docPalette[i%docPalette.length]]));

const sim = d3.forceSimulation(nodes)
  .force('link',d3.forceLink(edges).id(d=>d.id).distance(d=>d.edge_type==='SAME_DOCUMENT'?40:d.edge_type==='CO_OCCURS_IN'?90:70))
  .force('charge',d3.forceManyBody().strength(d=>d.node_type==='chunk'?-250:-120))
  .force('center',d3.forceCenter(w/2,h/2))
  .force('collision',d3.forceCollide().radius(d=>d.node_type==='chunk'?22:14));

let linkG=g.append('g'), nodeG=g.append('g');

function getNodeColor(d,mode){
  if(mode==='community'&&d.community_id!=null) return communityPalette[d.community_id%communityPalette.length];
  if(mode==='document'&&d.node_type==='chunk') return docColors[d.source_file]||'#60a5fa';
  if(d.node_type==='entity') return entityColors[d.entity_type]||'#f87171';
  return '#60a5fa';
}

function render(){
  const ef=document.getElementById('edge-filter').value;
  const nf=document.getElementById('node-filter').value;
  const cm=document.getElementById('color-mode').value;
  const sq=document.getElementById('search-box').value.toLowerCase();

  let filtEdges=edges;
  if(ef!=='all') filtEdges=edges.filter(e=>e.edge_type===ef);

  let visIds=new Set(nodes.map(n=>n.id));
  if(nf!=='all') visIds=new Set(nodes.filter(n=>n.node_type===nf).map(n=>n.id));
  filtEdges=filtEdges.filter(e=>{
    const s=typeof e.source==='object'?e.source.id:e.source;
    const t=typeof e.target==='object'?e.target.id:e.target;
    return visIds.has(s)&&visIds.has(t);
  });

  linkG.selectAll('*').remove(); nodeG.selectAll('*').remove();

  linkG.selectAll('line').data(filtEdges).join('line')
    .attr('stroke',d=>edgeColors[d.edge_type]||'#444')
    .attr('stroke-width',d=>d.edge_type==='CO_OCCURS_IN'?1.8:1)
    .attr('stroke-dasharray',d=>d.edge_type==='SAME_DOCUMENT'?'4 3':'none')
    .attr('stroke-opacity',0.5);

  const nd=nodes.filter(n=>visIds.has(n.id));
  const gs=nodeG.selectAll('g').data(nd,d=>d.id).join('g').attr('cursor','pointer')
    .call(d3.drag().on('start',(e,d)=>{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y})
      .on('drag',(e,d)=>{d.fx=e.x;d.fy=e.y})
      .on('end',(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}));

  gs.append('circle')
    .attr('r',d=>d.node_type==='chunk'?14:8)
    .attr('fill',d=>getNodeColor(d,cm))
    .attr('stroke','rgba(255,255,255,0.08)').attr('stroke-width',0.5)
    .attr('opacity',d=>{
      if(!sq) return 1;
      return d.id.toLowerCase().includes(sq)||(d.section||'').toLowerCase().includes(sq)
        ||(d.source_file||'').toLowerCase().includes(sq)
        ||(d.entity_type||'').toLowerCase().includes(sq)?1:0.1;
    });

  gs.append('text').text(d=>{
    const n=d.node_type==='chunk'?(d.section||d.id):d.id.replace('entity::','');
    return n.length>16?n.slice(0,14)+'..':n;
  }).attr('text-anchor','middle').attr('dy',d=>d.node_type==='chunk'?26:18)
    .attr('font-size','10px').attr('fill','#777').attr('pointer-events','none');

  gs.on('click',(e,d)=>showDetail(d))
    .on('mouseenter',(e,d)=>{
      const tt=document.getElementById('tooltip');
      tt.style.display='block';tt.style.left=(e.pageX+10)+'px';tt.style.top=(e.pageY-16)+'px';
      tt.innerHTML=d.node_type==='chunk'
        ?`<strong>${d.id}</strong><br>${d.source_file}<br>p.${d.page} — ${d.section}${d.community_id!=null?'<br>Community '+d.community_id:''}`
        :`<strong>${d.id.replace('entity::','')}</strong><br>${d.entity_type}<br>${d.description||''}${d.community_id!=null?'<br>Community '+d.community_id:''}`;
    })
    .on('mouseleave',()=>{document.getElementById('tooltip').style.display='none'});
}

function showDetail(d){
  const p=document.getElementById('detail');
  const conn=edges.filter(e=>{
    const s=typeof e.source==='object'?e.source.id:e.source;
    const t=typeof e.target==='object'?e.target.id:e.target;
    return s===d.id||t===d.id;
  });
  const grouped={};
  conn.forEach(e=>{
    const tp=e.edge_type;if(!grouped[tp])grouped[tp]=[];
    const s=typeof e.source==='object'?e.source.id:e.source;
    const t=typeof e.target==='object'?e.target.id:e.target;
    const o=(s===d.id?t:s).replace('entity::','');
    const x=e.shared_entity?` (${e.shared_entity})`:e.relation?` [${e.relation}]`:'';
    grouped[tp].push(o+x);
  });
  let h=`<strong>${d.node_type==='chunk'?d.id:d.id.replace('entity::','')}</strong>`;
  if(d.node_type==='chunk') h+=` | ${d.source_file} p.${d.page} — ${d.section}`;
  else h+=` | ${d.entity_type} — ${d.description||''}`;
  if(d.community_id!=null) h+=` | Community ${d.community_id}`;
  h+='<br>';
  for(const[tp,targets]of Object.entries(grouped)){
    h+=`<span style="color:${edgeColors[tp]||'#888'}">${tp}</span>: ${targets.join(', ')} &nbsp;`;
  }
  if(d.node_type==='chunk'&&d.text_preview) h+=`<br><span style="color:#666">${d.text_preview.slice(0,200)}</span>`;
  p.innerHTML=h;
  
  // Highlight
  nodeG.selectAll('circle').attr('opacity',n=>{
    if(n.id===d.id)return 1;
    return conn.some(e=>{
      const s=typeof e.source==='object'?e.source.id:e.source;
      const t=typeof e.target==='object'?e.target.id:e.target;
      return s===n.id||t===n.id;
    })?0.9:0.08;
  });
  linkG.selectAll('line').attr('stroke-opacity',e=>{
    const s=typeof e.source==='object'?e.source.id:e.source;
    const t=typeof e.target==='object'?e.target.id:e.target;
    return(s===d.id||t===d.id)?0.8:0.03;
  });
}

render();
sim.on('tick',()=>{
  linkG.selectAll('line').attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  nodeG.selectAll('g').attr('transform',d=>`translate(${d.x},${d.y})`);
});

['edge-filter','node-filter','color-mode'].forEach(id=>
  document.getElementById(id).addEventListener('change',()=>{render();sim.alpha(0.3).restart();}));
document.getElementById('search-box').addEventListener('input',()=>render());
svg.on('click',e=>{
  if(e.target===svg.node()){
    nodeG.selectAll('circle').attr('opacity',1);
    linkG.selectAll('line').attr('stroke-opacity',0.5);
    document.getElementById('detail').innerHTML='Click any node to inspect its connections and metadata.';
  }
});
</script>
</body>
</html>"""
```

### Usage

```python
# %% Generate visualization
from visualize_graph import generate_graph_html

generate_graph_html(G, "graph_viewer.html", max_nodes=500)
# Open graph_viewer.html in browser — full interactive graph with:
#   - Filter by edge type (co-occurs, mentions, same-doc, relations)
#   - Filter by node type (chunks, entities)
#   - Color by: node type / community / document
#   - Search box to highlight nodes
#   - Click any node for full metadata + connections
#   - Drag, zoom, pan
```

---

## Step 11: Full Pipeline (Notebook-Style)

```python
# %% [markdown]
# # Graph RAG Pipeline — End to End
# Supports both full build and incremental update modes.

# %%
import json
import os
from config import *

# %%
# Load your existing chunks (adapt to your chunker output)
with open("chunks.json", "r") as f:
    chunks = json.load(f)  # List[{"chunk_id": ..., "text": ..., "metadata": {...}}]
print(f"Loaded {len(chunks)} chunks")

# %% [markdown]
# ## Mode A: Full Build (first run or after extraction prompt changes)

# %%
from extraction import extract_all
from graph_builder import build_graph
from graph_export import export_graph_json, save_graph
from community import detect_communities, summarize_all_communities
from visualize_graph import generate_graph_html

# Step 1: Entity extraction (LLM call per chunk — the slow part)
extractions = extract_all(chunks)
print(f"Extracted entities from {len(extractions)}/{len(chunks)} chunks")

# %%
# Step 2: Build graph
G = build_graph(chunks, extractions)

# %%
# Step 3: Community detection
partition = detect_communities(G, resolution=COMMUNITY_RESOLUTION)
community_summaries = summarize_all_communities(G, partition)

# Persist summaries
with open("community_summaries.json", "w") as f:
    json.dump({str(k): v for k, v in community_summaries.items()}, f, ensure_ascii=False, indent=2)

# %%
# Step 4: Persist graph
save_graph(G, GRAPH_PICKLE_PATH)
export_graph_json(G, "knowledge_graph.json")

# Save processed chunk IDs for incremental mode
from incremental import save_processed_ids
save_processed_ids({c["chunk_id"] for c in chunks})

# %%
# Step 5: Generate visualization
generate_graph_html(G, VIZ_OUTPUT_PATH, max_nodes=500)
print(f"Open {VIZ_OUTPUT_PATH} in your browser to explore the graph")

# %% [markdown]
# ## Mode B: Incremental Update (new documents added)

# %%
from incremental import incremental_update

# Load ALL chunks (old + new) — the function diffs internally
G, community_summaries = incremental_update(chunks, rerun_communities=True)

# Regenerate visualization after update
generate_graph_html(G, VIZ_OUTPUT_PATH, max_nodes=500)

# %% [markdown]
# ## Query

# %%
from hybrid_retriever import hybrid_retrieve
from answer_gen import generate_answer

# Load community summaries
with open("community_summaries.json", "r") as f:
    community_summaries = {int(k): v for k, v in json.load(f).items()}

chunks_lookup = {c["chunk_id"]: c for c in chunks}

# --- Local search (specific query) ---
query = "Hangi ürünler KMH kapsamında değerlendiriliyor?"
context, comm_ctx = hybrid_retrieve(query, G, chunks_lookup, community_summaries)
answer = generate_answer(query, context, comm_ctx)
print(answer)

# %%
# --- Global search (broad/thematic query using community summaries) ---
query_global = "DenizBank'ın risk yönetimi süreçleri nelerdir?"
context, comm_ctx = hybrid_retrieve(query_global, G, chunks_lookup, community_summaries)
answer = generate_answer(query_global, context, comm_ctx)
print(answer)

# %% [markdown]
# ## Graph Statistics

# %%
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

entity_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'entity']
chunk_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'chunk']
print(f"Entity nodes: {len(entity_nodes)}")
print(f"Chunk nodes: {len(chunk_nodes)}")

# Community stats
from collections import Counter
comm_ids = [G.nodes[n].get('community_id') for n in G.nodes() if 'community_id' in G.nodes[n]]
comm_counts = Counter(comm_ids)
print(f"Communities: {len(comm_counts)}")
for cid, count in comm_counts.most_common(10):
    preview = community_summaries.get(cid, "No summary")[:80]
    print(f"  Community {cid} ({count} nodes): {preview}...")

# %%
# Top entities by degree
degrees = sorted(
    [(n, G.degree(n)) for n in entity_nodes],
    key=lambda x: x[1], reverse=True
)[:20]
for name, deg in degrees:
    print(f"  {name}: {deg} connections")
```

---

## Graph Schema Summary

```
┌──────────┐  MENTIONS   ┌──────────┐  RELATION   ┌──────────┐
│  CHUNK   │────────────►│  ENTITY  │────────────►│  ENTITY  │
│          │             │          │             │          │
│ chunk_id │             │ name     │             │ name     │
│ source   │             │ type     │             │ type     │
│ page     │             │ desc     │             │ desc     │
│ comm_id  │             │ comm_id  │             │ comm_id  │
└────┬─────┘             └──────────┘             └──────────┘
     │
     │ CO_OCCURS_IN (shared entities)
     │ SAME_DOCUMENT (sequential chunks)
     ▼
┌──────────┐
│  CHUNK   │
└──────────┘

Communities (Louvain):  comm_id stored on each node
                        Community summaries stored in community_summaries.json
```

---

## Tuning Parameters

| Parameter               | Default | Notes                                              |
|--------------------------|---------|-----------------------------------------------------|
| `EXTRACTION_MAX_TOKENS` | 2048    | Increase if chunks are long (>1000 tokens)          |
| `EXTRACTION_TEMPERATURE`| 0.1     | Low = deterministic extraction                       |
| `RETRIEVAL_TOP_K`       | 10      | Number of chunks to retrieve                         |
| `GRAPH_TRAVERSAL_DEPTH` | 2       | BFS depth from entity nodes. 3+ gets noisy.         |
| Graph boost score       | 0.15    | Weight for graph-found chunks vs vector-only         |
| Both-path bonus         | 0.10    | Extra boost when chunk found by both vector & graph  |
| Community membership    | 0.05    | Boost for chunks in a summarized community           |
| `COMMUNITY_RESOLUTION`  | 1.0     | Louvain resolution. Higher = more smaller communities|
| `COMMUNITY_MIN_SIZE`    | 3       | Skip communities with fewer nodes than this          |
| `COMMUNITY_SUMMARY_MAX_TOKENS` | 512 | LLM output cap for each community summary       |
| `max_nodes` (viz)       | 500     | Sample threshold for HTML visualization              |
