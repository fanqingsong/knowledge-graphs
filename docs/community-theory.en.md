# Communities & Community Reports: a Principles-Level Explanation

> This document explains community detection, community summaries, and the principles behind the retrieval
> strategies from a **first-principles** (math + algorithm) standpoint, with little code.
> See also: concepts intro [`community.en.md`](community.en.md), ingestion [`upload-pipeline.en.md`](upload-pipeline.en.md).
> *(Chinese version: [`community-theory.md`](community-theory.md))*

## 0. The one-line chain of principles

**Modularity** defines "what is a good partition" → **Louvain/Leiden** approximate-maximize it via greedy moves +
aggregation (Leiden additionally guarantees connectivity) → locally-related chunks are **grouped into communities** →
**LLM local summarization** reduces "global Q&A" to "retrieval in summary-space" (a map-reduce idea) → at retrieval
time **summaries locate the theme, Chunks/subgraph supply detail or structure**, so RAG can both "see the forest"
(community reports) and "see the trees" (entities/subgraphs).

---

## 1. The essence of community detection: cutting a graph into "dense-inside, sparse-outside"

### 1.1 Problem statement
Given a graph $G=(V,E)$, find a partition $\mathcal{C}=\{C_1,\dots,C_k\}$ such that edges are **dense within** communities
and **sparse between** them. This is **unsupervised clustering** — no labels, purely structural.

### 1.2 The core metric: modularity
Modularity measures "how much better this partition is than random." Intuition: if two nodes share a community merely
because the graph is dense anyway (random wiring would do the same), that's not skill — subtract that baseline.

Definition:
$$
Q = \frac{1}{2m}\sum_{i,j}\left[A_{ij} - \frac{k_i k_j}{2m}\right]\delta(c_i,c_j)
$$

- $A_{ij}$: adjacency matrix (1 if there is an edge).
- $k_i$: degree of node $i$; $m$ total edges, $2m$ the sum of degrees.
- $\frac{k_i k_j}{2m}$: the **expected** number of edges between $i,j$ under a degree-preserving random graph (configuration model).
- $\delta(c_i,c_j)$: 1 if same community, else 0.

**Meaning**: sum only over same-community pairs and see how many more edges exist than the random expectation.
$Q\in[-1/2,1]$; in practice $Q\in[0.3,0.7]$ signals clear community structure. The project's stored
`leiden_modularity` / `louvain_modularity` are exactly this — **higher = better cut**.

> Key insight: modularity turns "clustering quality" into a **scalar optimizable objective**. Community detection thus
> becomes "maximize $Q$", a combinatorial optimization problem.

### 1.3 Why it's hard: NP-hard
Exactly maximizing $Q$ is NP-hard (partition space explodes exponentially with nodes), so every practical algorithm is a
**heuristic approximation** — Louvain and Leiden included.

---

## 2. Louvain vs Leiden: the principle difference

Both are "**greedy local optimization + hierarchical aggregation**", but Leiden fixes a theoretical flaw of Louvain.

### 2.1 Louvain (two-phase iteration)

**Phase 1 (local move)**: each node evaluates the $\Delta Q$ of moving into a neighbor's community and moves to the one
that maximizes the gain, until no node wants to move. The gain of node $i$ joining community $C$ is approximated by:

$$
\Delta Q \approx \left[\frac{\Sigma_{in}+2k_{i,in}}{2m}-\left(\frac{\Sigma_{tot}+k_i}{2m}\right)^2\right]-\left[\frac{\Sigma_{in}}{2m}-\left(\frac{\Sigma_{tot}}{2m}\right)^2-\left(\frac{k_i}{2m}\right)^2\right]
$$

$\Sigma_{in}$, $\Sigma_{tot}$, $k_{i,in}$ are all local community/node quantities, so $\Delta Q$ **needs only local
information and is $O(1)$** — this is the fundamental reason Louvain is fast.

**Phase 2 (aggregation)**: contract each community into a **supernode**, inter-community edges become weighted edges,
producing a smaller graph, and return to Phase 1. Alternate until $Q$ stops growing; the result is a **hierarchical
community tree**.

- **Pros**: very fast, near-linear $O(n)$.
- **Flaw**: produces **disconnected communities** — a community may contain two mutually unreachable groups lumped
  together (greedy checks only $\Delta Q$, not connectivity), and it gets stuck in local optima.

### 2.2 Leiden: adding "connectivity + refinement"

Leiden adds a **refinement phase** to Louvain's scheme, making it three phases:

1. **Local move** (like Louvain but more conservative; moves nodes out of a single community).
2. **Refinement** ⭐: for each community, check whether it can be split into purer sub-communities — **this step
   guarantees every community is connected**, eliminating Louvain's disconnected-community problem.
3. **Aggregation**: aggregate on the refined partition, then loop.

**Leiden's theoretical guarantees (absent in Louvain)**:
- Every community is **connected from within**.
- At convergence, **no community can raise $Q$ by being split wholesale** ($\gamma$-separation).
- At convergence, **no single node can raise $Q$ by moving** (asymptotically optimal).

Cost: slightly slower than Louvain, but higher quality and more stable — which is why the project makes **leiden the
default `community_type="leiden"`**.

### 2.3 A frequently-overlooked point
Both optimize $Q$ on an **undirected** graph (the project explicitly `to_undirected()` for Louvain; Leiden uses igraph
but modularity is symmetric). So **directional information is discarded during partitioning** — communities are formed
by "who is connected to whom", not by edge direction. Direction survives only in centralities and later subgraph
retrieval.

---

## 3. Why community summaries solve "global Q&A"

This is the core principle distinguishing GraphRAG from plain RAG.

### 3.1 The principled shortcoming of plain RAG
Vector-retrieval RAG is essentially: **approximate "answering a question" as "recalling the most similar local chunk"**.
This approximation rests on two premises:
1. The answer's information is **self-contained in some local chunk** (locality);
2. The question can be aligned to that chunk by "semantic similarity" (query-chunk alignment).

For global questions — "what is the core theme of these documents", "how are X and Y related overall" — **both premises
collapse**:
- the answer is spread across hundreds of chunks, no single one contains it;
- the question is "summary-level" and not very similar to any single chunk.

The brute-force fix — "stuff the whole corpus into the LLM" — blows up cost / context window.

### 3.2 GraphRAG's principle: map-reduce hierarchical summarization
Community summaries reduce a **global aggregation problem** to a **hierarchical summarization problem**:

1. **Aggregate into communities first (space for time)**: cluster $N$ chunks into $k \ll N$ communities by entity
   co-occurrence; one community = one thematic cluster.
2. **Local summarization (map)**: chunks within a community are already highly related, so one LLM call yields a report
   that **keeps key points, drops redundancy**. $N$ chunks → $k$ reports: compressed + denoised.
3. **Reduce on demand at retrieval**: for a global question, instead of recalling raw chunks, retrieve the **most
   relevant community reports** (high information density, short context) and let the LLM synthesize.

**Why communities are a good aggregation granularity?** Because community detection has already physically grouped
semantically-related, entity-sharing content — it is a **data-driven, graph-structure-guaranteed** grouping, more
on-theme than fixed chunking or naive clustering. Each summary then covers one cohesive theme and deduplicates most
thoroughly.

### 3.3 Why summaries are stored as vectors
A community report is just text. Embedding it into a separate `cr_store` is equivalent to **retrieving in
"summary-space"** rather than "raw-chunk-space". Summary-space advantages: higher signal-to-noise (redundancy removed),
purer themes, more accurate similarity matching, and recall of **condensed macro-semantics**. This is also why community
summaries live in an **independent vector index** (`CommunityReport`), separate from the raw `Chunk` store — two indices,
two retrieval granularities.

### 3.4 A principled trade-off
Summarization compresses information, so **detail is inevitably lost**. Community summaries excel at
"macro / summary / theme-level" answers and **struggle with precise facts / numbers / verbatim text**. This is exactly
what the two retrieval strategies divide labor over.

---

## 4. The principled trade-off of the two retrieval strategies

| Dimension | Communities (reports) | Subgraph |
|---|---|---|
| Carrier | summary text + same-community Chunks | summary + **community subgraph topology** + Chunks + entities |
| Retrieval principle | recall top-k communities in **summary-vector-space**, then fill detail with Chunks | recall 1 community, then feed **structured relations** to the LLM |
| Problem solved | global/thematic summary ("what is it about") | relational/topological ("how are things connected") |
| Context cost | medium | high (graph structure eats tokens) |

### 4.1 Communities mode: summary-first + detail backfill
- Use summary retrieval to **locate the theme** (coarse filter; `score_threshold=0.8` keeps only strongly-related communities);
- once locked, **locally vector-search** within that community to fill Chunk detail (fine filter).

Essentially a **two-stage funnel**: coarse-filter communities in low-dimensional summary-space ($k$=3), then fine-filter
Chunks in raw-space, balancing "macro location" with "detail support". `use_adjacent_chunks` further uses `NEXT/PREV`
edges to restore context continuity severed by chunking — i.e., using graph structure to **mend the fragmentation
chunking causes**.

### 4.2 Subgraph mode: let structure participate in reasoning
It treats the community report as an "entry point", but what really drives the LLM's reasoning is the **community
subgraph structure** (`filter_graph_by_communities`).

Principle: **some answers live in "structure", not "text"**. For example "through what path does A influence B" — the
answer lies in the edge path between entities, invisible in any single text span. Feeding the subgraph (nodes + edges)
as structured context lets the LLM perform **multi-hop relational reasoning**; layering in raw Chunks and `MENTIONS`
entities gives the reasoning evidentiary footing.

Cost: graph structure is token-heavy and demands strong structural understanding from the model (small models get
confused) — hence the **most expensive, most model-picky** strategy, used only when relational reasoning is needed.

### 4.3 The principled role of centralities
Communities answer "grouping"; centralities (PageRank/betweenness/closeness) answer "**who matters within / across
groups**":
- **PageRank** = random-walk stationary distribution → **global influence**;
- **Betweenness** = share of shortest paths → **bridging / hub** role;
- **Closeness** = average distance to the graph → **information diffusion speed**.

They are orthogonal to communities: communities answer "which clique", centralities answer "how central within/across
cliques". Current retrieval strategies mainly use communities; centralities are mostly graph-analysis fuel and a reserve
for future re-ranking.

---

## 5. References
- Newman, M. E. J. *Modularity and community structure in networks.* (modularity)
- Blondel, V. et al. *Fast unfolding of communities in large networks.* (Louvain)
- Traag, V. A. et al. *From Louvain to Leiden: guaranteeing well-connected communities.* (Leiden)
- Edge, D. et al. (Microsoft) *From local to global: a graph RAG approach to query-focused summarization.* (GraphRAG / community summaries)
