# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A GraphRAG toolkit + Streamlit demo: documents are ingested into **Neo4j** (used simultaneously as graph DB and vector store), entities/relationships are mined by an LLM, communities are detected (Leiden/Louvain) and summarized, and a `GraphAgentResponder` answers queries by combining vector search with Cypher traversal. The `graphrag` Python package is the toolkit; `app.py` + `streamlit_pages/` is the demo UI.

## Commands

```bash
# --- Local dev (requires Neo4j reachable + .env) ---
pip install -r requirements.txt
streamlit run app.py

# --- Docker (recommended; spins up Neo4j + app with hot reload) ---
./bin/start.sh        # builds + starts stack; seeds .env from config_example.env on first run
./bin/stop.sh         # stop, keep data volumes
./bin/stop.sh -v      # stop AND wipe neo4j_data volume
docker compose logs -f app

# Endpoints when running in Docker: app http://localhost:8080, Neo4j Browser http://localhost:7474 (neo4j/password123), Bolt :7687

```

There is **no test suite, lint config, or build step** in this repo. Verify changes by running the app and exercising the relevant Streamlit page.

## Configuration model (read this before touching config)

Config is layered and resolved in `streamlit_pages/upload.py` at startup:

1. If `./configuration.json` exists → `Configuration.from_file(...)` (JSON, full schema).
2. Otherwise → `get_configuration_from_env()` builds a `Configuration` from `config_example.env` via `dotenv`.

> Note: README mentions `config.env`; the **actual** files are `config_example.env` (committed template, also baked into the Docker image) and `.env` (gitignored, overrides in Docker via `env_file`). `KnowledgeGraphConfig.from_file` only reads JSON.

All config objects are pydantic models in `graphrag/core/config.py` (`Configuration`, `KnowledgeGraphConfig`, `LLMConf`, `EmbedderConf`, `ChunkerConf`, `Source`). The `ModelType` enum (`graphrag/core/config.py`) is the canonical list of supported providers: `azure-openai, google, groq, openai, ollama, trf, zhipu, siliconflow`. Provider SDKs are imported **lazily** inside the factories in `graphrag/core/` (`llm.py`, `embeddings.py`) — only the requested provider is loaded at runtime. Import them via `from graphrag.core.llm import fetch_llm` / `from graphrag.core.embeddings import get_embeddings`.

**Active defaults (config_example.env)** use Zhipu GLM (`glm-4.7`) for LLMs and SiliconFlow (`BAAI/bge-m3`, 1024-dim) for embeddings — both OpenAI-compatible endpoints. Dockerfile uses Huawei Cloud mirrors for apt/pip (CN-network optimization).

## Architecture

### Module layout: open interface vs sealed implementation
All data models (`Chunk`, `ProcessedDocument`, `Ontology`, `Community`, `CommunityReport`, `_Node`/`_Relationship`/`_Graph`) live in `graphrag/core/models.py` — a **pure-model** module with no backend/config/SDK deps (this is what broke the old `config <-> graph` import cycle).

Every subpackage follows the **same directory contract**, so the module boundary is visible in the tree:
- `<pkg>/__init__.py` — the **open interface**: re-exports the public API. External code imports only from here.
- `<pkg>/_impl/` — the **sealed implementation**: private internals. The `_` prefix is the contract — never import from `_impl/` outside its own package.

Currently `graph/`, `ingestion/`, `ontologies/` use this pattern. `core/` is the foundation layer — its modules (`config`, `models`, `logger`, `llm`, `embeddings`) are flat "module-as-interface" files (no `_impl` — each module IS the interface). `agents/` (each agent co-locates its own prompt templates inline — there is no separate `prompts/` package) also stays flat. Always reach for the public name:
```
from graphrag.graph import KnowledgeGraph            # NOT from graphrag.graph._impl...
from graphrag.ingestion import IngestionPipeline, ChunkEmbedder
from graphrag.core.llm import fetch_llm
from graphrag.core.logger import get_logger
from graphrag.ontologies import beiyin_ontology
```
The top-level `graphrag/__init__.py` itself stays import-light (docstring only); only an explicit `import graphrag.<pkg>` pulls in that package's chain.

### `KnowledgeGraph` backend
Import the facade from the package: `from graphrag.graph import KnowledgeGraph`. It is defined in `graphrag/graph/_impl/knowledge_graph.py:38` as `class KnowledgeGraph(MetadataMixin, IngestionMixin, AnalysisMixin, QueryMixin, Neo4jGraph)` — a composition where `Neo4jGraph` (from `langchain_neo4j`) provides the connection. Behaviour is split across **mixins** (`graphrag/graph/_impl/mixins/`):
- `metadata.py` (`MetadataMixin`) — graph schema (labels, relationships)
- `ingestion.py` (`IngestionMixin`) — `add_documents`, vector index writes
- `analysis.py` (`AnalysisMixin`) — `update_centralities_and_communities`, `get_communities`
- `queries.py` (`QueryMixin`) — vector + graph queries used by the QA agent

The mixins build on `graphrag/graph/_impl/base/`:
- `cypher.py` — all Cypher statements & transaction callbacks (single Cypher entry point)
- `algorithms.py` — pure networkx algorithms (community detection, centralities)

Add new graph capabilities by extending the relevant mixin, not by bloating `knowledge_graph.py`. Everything under `_impl/` is private — reach it via the `KnowledgeGraph` facade.

### Ingestion pipeline
Orchestrated end-to-end by `IngestionPipeline` (`graphrag/ingestion/_impl/pipeline.py`), exposed as `from graphrag.ingestion import IngestionPipeline`. The Streamlit Upload page just calls `pipeline.run(on_progress=st.write)`; the stages themselves (`Cleaner`/`Chunker`/`ChunkEmbedder`/`GraphMiner`, each implementing the unified `Stage` protocol `process(docs) -> docs`) plus the graph-write and community-report steps are private to `_impl/` and not re-exported.
```
LocalIngestor  →  Cleaner  →  Chunker  →  ChunkEmbedder  →  GraphMiner  →  KnowledgeGraph.add_documents  →  update_centralities_and_communities  →  CommunitiesSummarizer
```
- `GraphMiner` wraps the `GraphExtractor` agent (`graphrag/agents/graph_extractor.py`), which produces structured `_Node`/`_Relationship`/`_Graph` output (pydantic) from each chunk.
- After community detection, **CommunityReports are generated for both `leiden` and `louvain`** and stored — two community answering modes (`answer_with_community_reports`, `answer_with_community_subgraph`) retrieve from these, so ingestion is incomplete without them.
- `COMMUNITY_SOURCE=full` (whole graph) vs `entities` (entity-only subgraph, community ids propagated to Chunks via MENTIONS — better for large corpora). See `KnowledgeGraphConfig.community_source`.

### Agents (`graphrag/agents/`)
- `GraphExtractor` — mines a per-chunk subgraph; optionally constrained by an `Ontology`.
- `CommunitiesSummarizer` — produces `CommunityReport`s from `Community` sets.
- `GraphAgentResponder` — the QA entrypoint with multiple retrieval strategies (`answer_with_cypher`, `answer_with_context`, `answer_with_community_reports`, `answer_with_community_subgraph`, `answer`). Backed by up to three LLMs (qa / cypher / optional rephrase) via `LLMConf`. The full comparison table of strategies is in README.md.

### Ontologies (`graphrag/ontologies/`)
An `Ontology` (defined in `graphrag/core/models.py`) pins allowed node labels / relationship types. Built-in ontologies live under `ontologies/_impl/` (e.g. `beiyin.py`) and are exported as constants by `ontologies/__init__.py` (e.g. `beiyin_ontology`). `KnowledgeGraphConfig.ontology` takes an `Ontology` instance directly — pass it in code (the name-based `.env` selection and the `get_ontology()` / `ONTOLOGIES` registry have been removed). The ontology constrains extraction in two places: the `GraphExtractor` prompt (soft) and `KnowledgeGraph.add_documents` (hard — entities/relationships outside the allowed schema are dropped before writing to Neo4j). To add a domain ontology: define it under `ontologies/_impl/`, export it from `ontologies/__init__.py`.

### Prompts
Prompt templates are co-located with the agent that uses them: each `graphrag/agents/*.py` defines its own `get_*_prompt()` factory (langchain `PromptTemplate`). Edit a prompt in the same file as its agent — there is no separate `prompts/` package.

## Docs
In-repo concept/theory docs explain non-obvious areas in depth — consult before changing them:
- `docs/upload-pipeline.en.md` — stage-by-stage ingestion walkthrough
- `docs/community.en.md` / `docs/community-theory.en.md` — community detection concepts + Leiden/Louvain math
- `docs/community-implementation.md` — code-level walkthrough of the community layer
- `DEPLOYMENT_CN.md` — Docker deployment guide (Chinese; Zhipu GLM + SiliconFlow stack)

`.md` files without `.en`/`.cn` suffixes are Chinese translations of the same content.
