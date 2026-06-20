# Communities in a Knowledge Graph

> This document explains the concept of a "community", its role in this project's
> (GraphRAG-style) pipeline, and where each piece is implemented in the codebase.
> *(Chinese version: [`community.md`](community.md))*
>
> 🔬 For the underlying math & algorithms (modularity, Louvain vs Leiden, why community summaries solve global Q&A), see **[community-theory.en.md](community-theory.en.md)**.

## 1. What is a community

A **community** is a group of tightly-connected nodes (and their edges) that form a
"cluster" within a knowledge graph:

- **Inside** a community: nodes are highly interconnected.
- **Between** communities: connections are relatively sparse.

Think of it as a "friend group" or "interest group" in a social network — members
within the group know each other well, but interact little with other groups. At its
core, a community is a **clustering / grouping structure** that organizes scattered
entities into meaningful, theme-based clusters.

## 2. Why communities matter

Classic RAG recalls text chunks via vector nearest-neighbors. It handles concrete,
**local** questions well, but struggles with questions that require a "bird's-eye
view", such as:

- What are the main themes in this corpus?
- What are the key entities and how do they relate to each other?

Communities address this by pre-organizing the graph into thematic clusters and
generating **community summaries (Community Reports)**. Retrieval can then "see the
trees" (specific entities/relationships) *and* "see the forest" (community-level
summaries), enabling macro-level, summarizing answers. This is the core idea behind
Microsoft's **GraphRAG**.

## 3. Common community-detection algorithms

This project uses **modularity** as the quality metric for a partition and provides
two detectors:

| Algorithm | Characteristics | Node attribute |
| --- | --- | --- |
| **Leiden** | Improved Louvain; guarantees connected communities and higher quality. Preferred in this project. | `community_leiden` |
| **Louvain** | Greedy modularity-based algorithm; fast and the most widely used. | `community_louvain` |

Higher modularity means a "dense inside, sparse outside" — i.e. higher-quality — partition.

Other common (not implemented here) algorithms: Girvan-Newman (iteratively removing
"bridge" edges), Label Propagation (label spreading, good for large graphs), and
hierarchical clustering (communities nested within communities, multi-level).

### 3.1 What graph community detection runs over (`community_source`)

Community detection runs **on a graph** — but "the whole graph or only the entity
subgraph?" changes quality. This project switches between them via the
`community_source` config (`COMMUNITY_SOURCE` in `config.env`):

| Value | Detection object | How a Chunk gets a community id | Good for |
| --- | --- | --- | --- |
| `full` (default) | **the whole graph** (entities + Chunks + Documents + all edges) | assigned directly by the algorithm (often clustering along `NEXT` chains) | preserves historical behaviour |
| `entities` | **entity subgraph only** (`__Entity__` nodes + entity-entity edges) | **propagated**: each Chunk takes the majority community of the entities it `MENTIONS` | more theme-coherent communities; **recommended for large corpora** |

> Why does `full` mode produce diffuse communities? Chunks usually outnumber
> entities and are linked by many `NEXT` edges, which dilutes the real thematic
> structure between entities — many Chunks just cluster "in document order".
> `entities` mode clusters only on entities and then propagates the community id
> down to Chunks along `MENTIONS`, so communities track themes better. Note: a
> Chunk that mentions no entity is left unassigned (`null`) in `entities` mode and
> therefore takes no part in community retrieval — this is intended (it carries
> no thematic signal).

Implementation: `analysis.py` — `get_entity_digraph()` (entity subgraph) and
`_propagate_communities_to_chunks()` (propagation); the propagation Cypher is
`propagate_communities_to_chunks` in `cypher.py`.

## 4. The community pipeline in this project

The flow has four stages:

```
documents → entity/relation extraction → build knowledge graph → community detection → community summaries → retrieval & QA
```

1. **Build the graph**: extract entities and relations from documents and write them into the Neo4j knowledge graph.
2. **Detect communities**: run Leiden / Louvain to assign each node a community id.
3. **Summarize communities**: for each community, an LLM summarizes the chunks it contains into a summary, and embeds it for retrieval.
4. **Answer**: feed community summaries to the LLM as macro-level context to answer global questions.

## 5. Where it lives in the code

> `KnowledgeGraph` is a facade whose methods are split into focused mixins; the
> table points at the file where each concern is actually implemented.

| Concern | Location |
| --- | --- |
| Facade class (construction, connection, `vector_store`/`cr_store`) | `graphrag/graph/knowledge_graph.py` |
| Leiden / Louvain detection + modularity | `graphrag/graph/graph_data_structure.py` — `detect_leiden_communities()`, `detect_louvain_communities()`, `update_modularity()` |
| Data models for communities & reports | `graphrag/graph/graph_model.py` — `Community`, `CommunityReport` |
| Summary generation (LLM summary + embedding) | `graphrag/agents/community_summarizer.py` — `CommunitiesSummarizer.get_reports()` / `get_community_report()` |
| Community read/write (`get_communities`, `store_community_reports`, `update_centralities_and_communities`, `get_digraph`) | `graphrag/graph/analysis.py` (`AnalysisMixin`) |
| Stat properties (`number_of_*_communities`, `*_modularity`, …) | `graphrag/graph/metadata.py` (`MetadataMixin`) |
| Community-related Cypher & transaction callbacks | `graphrag/graph/cypher.py` (`fetch_communities`, `fetch_chunk`) |

### Key data structures

**`Community`** (`graphrag/graph/graph_model.py`) — describes one community in the graph:

- `community_type`: the partition type, e.g. `leiden` or `louvain`.
- `community_id`: the id stored as a node attribute.
- `community_size`: number of nodes in the community.
- `entity_ids` / `entity_names`: entities belonging to the community.
- `relationship_ids` / `relationship_types`: relationships belonging to the community.
- `chunks`: the text chunks that fall within the community; the summary is built from these.

**`CommunityReport`** (same file) — the summary report produced for a community:

- `community_type` / `community_id`: the owning community.
- `summary`: the LLM-generated summary text.
- `rank`: sorting weight; higher is better.
- `summary_embeddings`: vector embedding of the summary, used for retrieval.

### How a summary is produced (`community_summarizer.py`)

`get_community_report()`:

1. If the community has no chunks, log a warning and skip it.
2. Concatenate all chunk texts in the community as `context`.
3. Call the LLM with the summary prompt to produce `summary`.
4. Embed `summary` into `summary_embeddings`.
5. Wrap the result in a `CommunityReport` and return it.

## 6. References

- GraphRAG (Microsoft): answering global questions via community summaries.
- Traag, V. A. et al. *From Louvain to Leiden: guaranteeing well-connected communities.* (Leiden)
- Blondel, V. et al. *Fast unfolding of communities in large networks.* (Louvain)
