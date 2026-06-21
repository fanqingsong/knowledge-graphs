# Upload (Document Ingestion) Pipeline

> This document explains what happens after a file is uploaded on the Streamlit **Upload** page — how it is
> transformed, stage by stage, into a knowledge graph and community reports. Entry point: `streamlit_pages/upload.py`.
> *(Chinese version: [`upload-pipeline.md`](upload-pipeline.md))*

## 1. Overview

The Upload page separates file reception from the ingestion pipeline:

- **UI layer**: receives uploaded files and saves them to the local `source_docs/` folder.
- **Pipeline**: reads from `source_docs/`, runs 8 stages over `List[ProcessedDocument]`, enriching the data
  at each step (text → chunks → embeddings → graph → store → communities → reports).

Every stage takes and returns `List[ProcessedDocument]`:

```
source_docs/*.pdf,docx,txt,html
   │  ① load    (MIME→loader)         → doc.source
   │  ② clean   (regex normalization)  → source
   │  ③ chunk   (RecursiveSplitter)    → doc.chunks[].text
   │  ④ embed   (embedding model)      → chunk.embedding
   │  ⑤ mine    (LLM extracts graph)   → chunk.nodes / chunk.relationships
   │  ⑥ store   → Neo4j (Chunk/Entity/Relationship + vectors)
   │  ⑦ graph   → DiGraph → Leiden/Louvain/centralities → write back node attrs
   └─ ⑧ report  → per-community LLM summary + vector → CommunityReport vector store
```

## 2. UI layer: file reception (`streamlit_pages/upload.py`)

1. **Load config**: first try `Configuration.from_file(configuration.json)`, fall back to environment variables (`get_configuration_from_env()`).
2. **Upload files**: `st.file_uploader` accepts `.pdf / .docx / .txt / .html` and **writes each to the local `source_docs/` folder**.
   > ⚠ The pipeline reads **from disk** (`source_docs/`), not from the in-memory upload objects.
3. Clicking **"Ingest into Knowledge Graph"** triggers the pipeline; progress is shown per stage via `st.status`.

## 3. The eight pipeline stages

### ① Loading (`LocalIngestor` → `Ingestor`)
`graphrag/ingestion/ingestor.py`, `graphrag/ingestion/local_ingestor.py`

- `list_files()` walks `source_docs/` via `os.walk` to collect every file path.
- `file_preparation()` takes the parent folder name as `folder` metadata.
- `load_file()` uses **python-magic** to sniff the MIME type and pick a loader; unsupported types are skipped with a warning:

  | MIME | Loader |
  | --- | --- |
  | `application/pdf` | `PDFPlumberLoader` |
  | `text/plain` | `TextLoader` |
  | `text/html` | `BSHTMLLoader` (UTF-8) |
  | `.docx` | `Docx2txtLoader` |

- Page contents are joined with `\n\n` and wrapped in `ProcessedDocument(filename, source, metadata)`.

### ② Cleaning (`Cleaner`)
`graphrag/ingestion/cleaner.py` — `_clean_text()` is a set of regexes that normalize text:

- strip asterisks; fix bullet chars misread as `l`;
- normalize dashes/hyphens and curly apostrophes to `'`;
- remove the BOM (`﻿`) and control characters;
- insert a space between a lowercase/digit and a following uppercase (split glued camelCase);
- collapse extra blank lines/spaces; drop the Italian page footer `"Pagina X di Y"`.

Goal: improve chunking and extraction quality downstream.

### ③ Chunking (`Chunker`)
`graphrag/ingestion/chunker.py`

- Uses `RecursiveCharacterTextSplitter` with params from `CHUNKER_*` (defaults `chunk_size=1000`, `chunk_overlap=100`).
- Assigns an incrementing `chunk_id` to each chunk and wraps it in `Chunk(text, chunk_id, chunk_size, chunk_overlap)` on `doc.chunks`.

### ④ Embedding (`ChunkEmbedder`)
`graphrag/ingestion/embedder.py`

- Embeds each `chunk.text` into `chunk.embedding` and records `chunk.embeddings_model`.
- The model is chosen by `EMBEDDINGS_TYPE` (default: SiliconFlow `bge-m3`, 1024-dim).

### ⑤ Graph extraction (`GraphMiner` → `GraphExtractor`)
`graphrag/ingestion/graph_miner.py`, `graphrag/agents/graph_extractor.py`

- **Per chunk**, an LLM extracts entities and relations into `_Graph(nodes, relationships)`.
- `map_to_lc_graph()` (`graphrag/graph/graph_converters.py`) converts it to a langchain `GraphDocument`, attaching `nodes` / `relationships` to the chunk.
- An optional `Ontology` (`conf.database.ontology`) constrains allowed labels/relations.
- **This is the most expensive step** — one LLM call per chunk; the main bottleneck for large corpora.

### ⑥ Uploading to the graph (`KnowledgeGraph.add_documents`)
Facade `graphrag/graph/knowledge_graph.py`; implementation in `graphrag/graph/ingestion.py` (`IngestionMixin`); transaction callbacks in `graphrag/graph/cypher.py`

- Iterates docs → `store_chunks_for_doc()`: writes Chunks (with vectors) and their extracted nodes/relationships into Neo4j, creating `Chunk -[:MENTIONS]-> Entity` edges.
- Entities, relations, and vectors are now persisted.

### ⑦ Centralities & communities (`update_centralities_and_communities`)
Orchestration in `graphrag/graph/analysis.py` (`AnalysisMixin`); detection algorithms in `graphrag/graph/graph_data_structure.py`

- `get_digraph()` reads the whole Neo4j graph into a `networkx.DiGraph`.
- Runs **Louvain** and **Leiden** community detection, each producing a modularity score.
- `compute_centralities()` computes PageRank / betweenness / closeness.
- `update_properties()` writes `community_leiden`, `community_louvain`, the three centralities, and both modularities back onto node properties.
- Each sub-step is wrapped in try/except; a failure only logs a warning and does not abort the whole stage.
- For the meaning of "community" itself, see [`community.en.md`](community.en.md).

### ⑧ Community reports (`CommunitiesSummarizer`)
`graphrag/agents/community_summarizer.py`

Run **once for each of `leiden` and `louvain`**:

- `get_communities(comm_type)` fetches each community's entities/relations/chunks.
- `get_reports()` per community: joins chunk texts as `context` → LLM produces a summary → embeds the summary → wraps it in a `CommunityReport`.
- `store_community_reports()` writes the summary text + vector + metadata into the `CommunityReport` vector store (`cr_store`).
- This stage prepares the retrieval source for the **Communities / Subgraph** answering modes.

## 4. Finish & cleanup

- After `status.update(state="complete")`, a success message and a **"Cleanup Folder"** button appear.
- Clicking it deletes every file in `source_docs/` (**source files only — the graph in the database is untouched**).

## 5. Notes & caveats

- **Sequential, no concurrency**: the pipeline processes documents and chunks one at a time; stage ⑤ (LLM extraction) is the dominant cost.
- **Vector-index creation is commented out**: the `index_exists()/create_index()` block in `upload.py` is a `# TODO`; ingestion relies on an existing index (`INDEX_NAME`) provisioned at deploy time.
- **Re-ingestion creates duplicate nodes**: no dedup/entity resolution exists today; to start fresh, stop the services, `./bin/stop.sh -v` to delete the data volume, and restart.
