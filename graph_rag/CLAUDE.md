# CLAUDE.md — Graph RAG Project

## Project Overview

This is an **on-premise Graph RAG system** for DenizBank internal documents (PDF, DOCX, XLSX).
It augments a standard vector-based RAG pipeline with a **knowledge graph layer** that captures
entities, relationships, and cross-chunk connections extracted from document chunks.

### Existing Infrastructure (DO NOT rebuild)

| Component         | Detail                                                        |
|-------------------|---------------------------------------------------------------|
| **Chunkers**      | Already implemented for PDF, DOCX, XLSX. Output: `List[Chunk]` where each `Chunk` has `chunk_id`, `text`, `metadata` (source_file, page, section, doc_type). |
| **LLM**           | Qwen (hosted via vLLM on OpenShift). OpenAI-compatible endpoint. |
| **Embedding**     | Qwen3-Embedding (or equivalent) on OpenShift. OpenAI-compatible `/v1/embeddings`. |
| **Vector DB**     | Qdrant (running, accessible via REST/gRPC).                   |

### What This Project Adds

1. **Entity & Relationship Extraction** — LLM-based extraction from each chunk.
2. **In-Memory Knowledge Graph** — `networkx.MultiDiGraph` with persistence via `pickle` / JSON export.
3. **Community Detection** — Louvain clustering on the graph to identify topical communities, with LLM-generated community summaries for global search.
4. **Incremental Graph Updates** — Diff-based merge so new documents are added without rebuilding the full graph.
5. **Hybrid Retriever** — Combines Qdrant vector search + graph traversal + community context for assembly.
6. **Graph Visualization** — Self-contained HTML file with D3.js force-directed graph, filterable by edge/node type, with click-to-inspect detail panels. Generated from Python via `visualize_graph.py`.

---

## Architecture

```
Chunks (existing)
  │
  ├──► Embedding ──► Qdrant (existing)
  │
  └──► LLM Entity Extraction ──► NetworkX Graph ──► pickle / JSON
                                       │
                                       ├──► Louvain Community Detection ──► Community Summaries
                                       │
                                       ├──► Incremental Merge (new docs only)
                                       │
                                       ├──► Graph Visualization (HTML export)
                                       │
                                       ▼
                          Hybrid Retriever (vector + graph + community)
                                       │
                                       ▼
                              LLM Answer Generation
```

## Directory Structure

```
graph_rag/
├── CLAUDE.md                  # This file
├── prompt.md                  # Detailed implementation prompt
├── config.py                  # All endpoints, model names, collection names
├── models.py                  # Pydantic models: Entity, Relationship, ChunkNode, GraphEdge
├── extraction.py              # LLM-based entity/relationship extraction
├── graph_builder.py           # NetworkX graph construction & persistence
├── community.py               # Louvain community detection & LLM-based community summarization
├── incremental.py             # Incremental graph update logic (diff-based merge)
├── hybrid_retriever.py        # Vector + Graph + Community retrieval logic
├── graph_export.py            # Export graph to JSON for visualization
├── visualize_graph.py         # Generate self-contained HTML graph viewer
├── pipeline.py                # End-to-end orchestration
└── notebooks/
    └── graph_rag_pipeline.py  # Notebook-style (.py with # %% cells) runnable pipeline
```

## Code Style & Conventions

- **Notebook-style**: All pipeline code uses `# %%` and `# %% [markdown]` cell markers.
- **Minimal abstractions**: No unnecessary wrapper classes. Functions over classes when possible.
- **Type hints everywhere**: Use `Pydantic BaseModel` for data structures, `typing` for functions.
- **Direct HTTP calls**: Use `httpx` (async) or `requests` for LLM/embedding/Qdrant endpoints. No LangChain, no LlamaIndex — we keep the stack transparent.
- **Error handling**: Every LLM call must handle `finish_reason == "length"` (truncation guard). Every Qdrant call must handle connection errors gracefully.
- **Bilingual**: Code and comments in English. User-facing strings can be Turkish where needed.

## Key Technical Decisions

### Graph Storage: NetworkX (not Neo4j)
- **Why**: No additional infrastructure to deploy. The graph is metadata-scale (thousands to low millions of nodes), not web-scale. NetworkX handles this easily in-memory.
- **Persistence**: `pickle` for fast reload, `JSON` export for visualization and portability.
- **Migration path**: If graph outgrows memory, migrate to Neo4j/ArangoDB later. The `models.py` Pydantic schemas make this straightforward.

### Entity Extraction: LLM-based (not spaCy NER)
- **Why**: Domain-specific banking entities (product codes like KKSA, ETKRD, KMH; internal project names; regulatory references) won't be caught by generic NER.
- **How**: Structured output via Qwen with a JSON schema prompt. Each chunk produces `List[Entity]` and `List[Relationship]`.
- **Fallback**: If structured output fails (truncation, malformed JSON), log the chunk_id and skip — don't crash the pipeline.

### Hybrid Retrieval Strategy
1. **Vector path**: Query → embed → Qdrant top-k → candidate chunks.
2. **Graph path**: Extract entities from query → find matching nodes in graph → BFS/DFS to depth=2 → collect connected chunks.
3. **Community path**: Identify which communities the query entities belong to → include community summary as extra context.
4. **Merge**: Union of chunk_ids from all paths, deduplicated, ranked by a combined score (vector similarity + graph centrality + community membership).
5. **Context assembly**: Retrieve full chunk texts + relevant community summaries, format as numbered context blocks for LLM.

### Community Detection: Louvain
- **Why Louvain**: Fast, deterministic, works well on undirected projections of the graph. The `python-louvain` package (`community` import) runs in seconds on graphs with <100K nodes.
- **How**: Project the `MultiDiGraph` to an undirected `Graph` (only chunk-chunk and entity-entity edges), run Louvain, store `community_id` as a node attribute.
- **Community Summaries**: For each community, collect the `text_preview` of all member chunks, send to LLM with a summarization prompt. Store summaries as a `dict[int, str]` and persist alongside the graph.
- **Global Search**: When the query is broad/thematic (e.g., "What are all risk management processes?"), retrieve relevant community summaries instead of individual chunks. This is the Microsoft GraphRAG "global search" pattern.

### Incremental Graph Updates
- **Why**: Full graph rebuild on every new document is wasteful when you add 5 files to a 10K-chunk corpus.
- **How**: Maintain a `processed_chunk_ids.json` log. On each pipeline run, diff incoming chunk_ids against the log. Only extract entities for new chunks, then merge new nodes/edges into the existing graph.
- **Merge rules**: New entity nodes are deduplicated against existing ones (lowercase match). New edges are added. Existing edges are NOT removed — deletion requires explicit "remove document" operation.
- **Community re-detection**: After merging new nodes, re-run Louvain on the full graph and regenerate summaries for affected communities only.
- **Caveat**: If entity extraction prompts change, you should do a full rebuild to ensure consistency.

### Graph Visualization
- **What**: A self-contained HTML file with D3.js force-directed graph. No server needed — just open in browser.
- **Generated by**: `visualize_graph.py` reads the graph (pickle or JSON export) and injects node/edge data into an HTML template with embedded D3.js.
- **Features**: Filter by edge type, filter by node type, click-to-inspect with full metadata, community coloring, drag-to-reposition, zoom/pan.
- **When to generate**: After every graph build or incremental update. The HTML file is a snapshot — it reflects the graph state at generation time.

## Environment

- **Runtime**: Python 3.10+ on OpenShift Jupyter or VSCode
- **Key packages**: `networkx`, `networkx[community]` (or `python-louvain`/`community`), `httpx`, `pydantic`, `requests`, `tqdm`
- **No GPU needed** for graph operations — GPU is consumed by the Qwen vLLM endpoint.

## Config Pattern

All endpoints and parameters live in `config.py`:

```python
# config.py
QWEN_BASE_URL = "http://<vllm-host>:<port>/v1"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"  # or whatever is deployed
EMBEDDING_BASE_URL = "http://<embedding-host>:<port>/v1"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding"
QDRANT_URL = "http://<qdrant-host>:6333"
QDRANT_COLLECTION = "documents"

EXTRACTION_MAX_TOKENS = 2048
EXTRACTION_TEMPERATURE = 0.1
RETRIEVAL_TOP_K = 10
GRAPH_TRAVERSAL_DEPTH = 2

# Community detection
COMMUNITY_RESOLUTION = 1.0          # Louvain resolution param (higher = more smaller communities)
COMMUNITY_SUMMARY_MAX_TOKENS = 512  # Max tokens for each community summary
COMMUNITY_MIN_SIZE = 3              # Skip communities with fewer than N nodes

# Incremental updates
GRAPH_PICKLE_PATH = "knowledge_graph.pkl"
PROCESSED_CHUNKS_LOG = "processed_chunk_ids.json"  # Track already-processed chunk_ids

# Visualization
VIZ_OUTPUT_PATH = "graph_viewer.html"
```

## Testing

- Test extraction on 5-10 representative chunks first before running full pipeline.
- Validate graph connectivity: `nx.is_connected(G.to_undirected())` — expect multiple components (one per document cluster).
- Test hybrid retrieval with known queries where the answer spans multiple documents.

## Common Pitfalls

1. **JSON truncation from LLM**: Always check `finish_reason`. If `"length"`, increase `max_tokens` or split the chunk.
2. **Entity deduplication**: Same entity appears with different surface forms ("DenizBank", "Denizbank", "DENIZBANK"). Normalize to lowercase before graph insertion.
3. **Graph bloat**: Filter out low-confidence entities. Set a minimum entity mention threshold.
4. **Qdrant collection schema mismatch**: Ensure the vector dimension matches your embedding model output.
5. **Louvain on directed graph**: Louvain requires undirected graph. Always call `G.to_undirected()` before running community detection. Do NOT modify the original `MultiDiGraph`.
6. **Community summary staleness**: After incremental update, only regenerate summaries for communities whose membership changed. Track `community_id → set(chunk_ids)` to detect changes.
7. **Incremental merge race conditions**: If two pipeline runs overlap, the pickle file can corrupt. Use file locking (`fcntl.flock` or `filelock` package) on writes.
8. **Visualization file size**: For graphs >5K nodes, the HTML file gets heavy. Add a `max_nodes` param to `visualize_graph.py` that samples the most-connected subgraph.
