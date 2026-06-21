"""
analysis.py — AnalysisMixin
===========================

The graph-analysis side of `KnowledgeGraph`: turn the raw Neo4j graph into
*communities* + *centralities*, and persist LLM-generated community reports.

High-level pipeline (driven by `update_centralities_and_communities`)::

    ┌──────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐
    │  snapshot    │─>│   Louvain     │─>│   Leiden      │─>│ centralities │
    │  as DiGraph  │  │ communities   │  │ communities   │  │ PR/betw/close│
    │ (full/entity)│  │ + modularity  │  │ + modularity  │  └──────┬───────┘
    └──────────────┘  └───────────────┘  └───────────────┘         │
                                                                    ▼
                                              ┌──────────────────────────────────┐
                                              │ update_properties()              │
                                              │   write community/centrality     │
                                              │   back onto Neo4j nodes          │
                                              │   (+ propagate to Chunks when    │
                                              │       community_source="entities")│
                                              └──────────────────────────────────┘

Public methods:

    · get_digraph                         whole graph as networkx (Entities + Chunks)
    · get_entity_digraph                  entity-only subgraph (no Chunk nodes)
    · update_centralities_and_communities orchestrator: detect + compute + write
    · update_properties                   low-level node-property writer
    · get_communities                     read communities (with their chunks) back out
    · store_community_reports             persist LLM-generated community summaries

The pure algorithms (Leiden/Louvain/centralities) live in
`algorithms.py`; the Cypher callbacks live in `cypher.py`.
"""

from typing import List, Optional

import networkx as nx

from graphrag.graph._impl.base import cypher
from graphrag.graph._impl.base.algorithms import (
    compute_centralities,
    detect_leiden_communities,
    detect_louvain_communities,
)
from graphrag.core.models import Community, CommunityReport
from graphrag.core.models import Chunk
from graphrag.core.logger import get_logger


logger = get_logger(__name__)


class AnalysisMixin:
    """Provides `KnowledgeGraph` with community / centrality analysis methods."""

    def get_digraph(self) -> nx.DiGraph:
        """Snapshot the *whole* Neo4j graph into a `networkx.DiGraph`.

        Pulls every node (Entities **and** Chunks) and every relationship,
        including the dense Chunk chaining edges::

            Neo4j                                  networkx.DiGraph
            ─────                                  ────────────────
            (:Person)──MENTIONS─>(:Chunk)──NEXT─>(:Chunk)
                │                     │              Chunk nodes + NEXT/PART_OF edges
                │KNOWS                │              make this graph large & dense;
                ▼                     │              use `get_entity_digraph` for
            (:Person)                 └─>            community detection over
                                                   entities only.

        Node data carries ``labels`` + all Neo4j properties; edge data carries
        ``type`` + properties.
        """
        G = nx.DiGraph()

        with self._driver.session() as session:
            nodes = session.run(cypher.DIGRAPH_NODES_QUERY)
            for record in nodes:
                G.add_node(record["node_id"], labels=record["labels"], **record["properties"])

            relationships = session.run(cypher.DIGRAPH_RELS_QUERY)
            for record in relationships:
                G.add_edge(record["source"], record["target"], type=record["rel_type"], **record["properties"])

        logger.info(f"DiGraph with {len(G.nodes)} nodes and {len(G.edges)} relationships")
        return G

    def get_entity_digraph(self) -> nx.DiGraph:
        """Entity-only subgraph as a `networkx.DiGraph`: `__Entity__` nodes and
        the relationships directly between them (no Chunk nodes, no NEXT/PART_OF).

        Contrast with the full snapshot::

            full digraph (`get_digraph`)      entity digraph (this method)
            ────────────────────────────      ──────────────────────────
            (:Person)─MENTIONS─>(:Chunk)      (:Person)──KNOWS──>(:Person)
                │                   │              Chunk nodes and their
                │KNOWS              │              MENTIONS / NEXT / PART_OF
                ▼                   ▼              edges are dropped
            (:Person)           (:Chunk)

        Used in `community_source="entities"` mode so community detection
        reflects the *thematic* structure of entities rather than being diluted
        by the (much more numerous) Chunk nodes and their chaining edges.
        """
        G = nx.DiGraph()

        with self._driver.session() as session:
            nodes = session.run(cypher.ENTITY_DIGRAPH_NODES_QUERY)
            for record in nodes:
                G.add_node(record["node_id"], labels=record["labels"], **record["properties"])

            relationships = session.run(cypher.ENTITY_DIGRAPH_RELS_QUERY)
            for record in relationships:
                G.add_edge(record["source"], record["target"], type=record["rel_type"], **record["properties"])

        logger.info(f"Entity DiGraph with {len(G.nodes)} nodes and {len(G.edges)} relationships")
        return G

    def update_properties(
        self,
        G: Optional[nx.DiGraph] = None,
        centralities: bool = False,
        leiden_communities: bool = False,
        louvain_communities: bool = False,
        leiden_modularity: Optional[float] = None,
        louvain_modularity: Optional[float] = None,
    ):
        """Write community ids + centrality scores from `G` back onto Neo4j nodes.

        For each networkx node, `cypher.build_update_query` produces a parameterised
        ``SET`` statement::

            G node data                       Neo4j node (matched by elementId)
            ┌────────────────────┐            ┌─────────────────────────────┐
            │ "community_leiden" │   ─SET─>   │ n.community_leiden = 3      │
            │ "community_louvain"│            │ n.community_louvain = 5     │
            │ "pagerank"         │            │ n.pagerank = 0.42          │
            │ "betweenness"      │            │ n.betweenness = 0.17       │
            │ "closeness"        │            │ n.closeness = 0.55         │
            └────────────────────┘            └─────────────────────────────┘

        Missing values default to ``-1`` (community) / ``0.0`` (centrality).
        Graph-level modularity scores are written separately as standalone
        `GraphMetric` rows via `cypher.update_modularity`.
        """
        with self._driver.session() as session:

            if any([centralities, leiden_communities, louvain_communities]):

                for node, data in G.nodes(data=True):
                    query, params = cypher.build_update_query(
                        node_id=node,
                        centralities=centralities,
                        leiden_communities=leiden_communities,
                        louvain_communities=louvain_communities,
                        community_leiden=int(data.get("community_leiden", -1)),
                        community_louvain=int(data.get("community_louvain", -1)),
                        pagerank=float(data.get("pagerank", 0.0)),
                        betweenness=float(data.get("betweenness", 0.0)),
                        closeness=float(data.get("closeness", 0.0)),
                    )
                    try:
                        session.run(query, params)
                    except Exception:
                        logger.warning(f"Update Query failed for node_id: {node}")

                logger.info("Updated nodes properties in Graph")

            if leiden_modularity is not None:
                cypher.update_modularity(session, leiden_modularity, "leiden")
                logger.info("Updated Leiden Modularity property in Graph")

            if louvain_modularity is not None:
                cypher.update_modularity(session, louvain_modularity, "louvain")
                logger.info("Updated Louvain Modularity property in Graph")

    def update_centralities_and_communities(self):
        """Run the full analysis pipeline on the Knowledge Graph.

        Steps (each wrapped so a failure in one doesn't abort the rest)::

            ┌──────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐
            │ snapshot as  │─>│   Louvain     │─>│   Leiden      │─>│ centralities │
            │  DiGraph     │  │ communities   │  │ communities   │  │ PR/betw/close│
            │ (full/ent.)  │  │ + modularity  │  │ + modularity  │  └──────┬───────┘
            └──────────────┘  └───────────────┘  └───────────────┘         │
                                                                         ▼
                                              ┌────────────────────────────────────┐
                                              │ update_properties()  -> Neo4j nodes │
                                              │ + propagate community ids to Chunks │
                                              │   (only in entity mode)             │
                                              └────────────────────────────────────┘

        The graph source follows `community_source`: ``"entities"`` runs detection
        over the entity subgraph only and then propagates the resulting ids onto
        Chunks via `_propagate_communities_to_chunks`; anything else uses the
        whole graph.
        """
        source = getattr(self, "community_source", "full")
        if source == "entities":
            G = self.get_entity_digraph()
        else:
            G = self.get_digraph()

        lv = False
        louvain_mod = None
        ld = False
        leiden_mod = None
        centralities = False

        try:
            G, louvain_mod = detect_louvain_communities(G, return_modularity=True)
            lv = True
        except Exception as e:
            logger.warning(f"Something went wrong detecting Louvain Communities: {e}")

        try:
            G, leiden_mod = detect_leiden_communities(G, return_modularity=True)
            ld = True
        except Exception as e:
            logger.warning(f"Something went wrong detecting Leiden Communities: {e}")

        try:
            G = compute_centralities(G)
            centralities = True
        except Exception as e:
            logger.warning(f"Something went wrong computing Centralities degrees on graph: {e}")

        try:
            self.update_properties(G, centralities, ld, lv, leiden_mod, louvain_mod)
        except Exception as e:
            logger.warning(f"Something went wrong while updating properties on graph nodes: {e}")

        # In entity-subgraph mode only entities carry a community id; push it
        # down onto the Chunks they mention so community retrieval still works.
        if source == "entities":
            self._propagate_communities_to_chunks()

    def _propagate_communities_to_chunks(self):
        """Give each Chunk the majority community of the entities it MENTIONS.

        In `community_source="entities"` mode only entities carry a community id.
        QA retrieves Chunks by community, so we vote the community down the
        MENTIONS edges::

            EntityA community=3    EntityB community=3    EntityC community=7
                ▲   :MENTIONS          ▲   :MENTIONS          ▲  :MENTIONS
                │                      │                      │
                └────────── (:Chunk) ──┴──────────────────────┘
                              │
                              ▼   majority vote (3 and 3 beat 7)
                       chunk.community_leiden = 3

        Done for both algorithms (``leiden`` and ``louvain``).
        """
        with self._driver.session() as session:
            for comm_type in ("leiden", "louvain"):
                try:
                    session.execute_write(cypher.propagate_communities_to_chunks, comm_type)
                    logger.info(f"Propagated {comm_type} communities to chunks")
                except Exception as e:
                    logger.warning(f"Issue propagating {comm_type} communities to chunks: {e}")

    def get_communities(self, comm_type: str = "leiden") -> List[Community]:
        """Read communities back out of the graph, each bundled with its chunks.

        Groups nodes by their ``community_<comm_type>`` property, then for every
        community collects its entities, relationships, and the Chunks that
        mention those entities::

            GROUP BY community_leiden
            ┌─── community 1 ───┐   ┌─── community 2 ───┐
            │ entities: A, B, C  │   │ entities: D, E    │
            │ rels:    KNOWS(x2) │   │ rels:    WORKS_AT │
            │ chunks:  [c1, c3]  │   │ chunks:  [c2]     │   ->  List[Community]
            └────────────────────┘   └───────────────────┘        (chunks hydrated
                                                                          with text)

        ``GraphMetric`` rows (the stored modularity scores) are skipped — they
        masquerade as communities in the raw result.
        """
        if comm_type not in ["leiden", "louvain"]:
            raise NotImplementedError("This Community type has not been implemented.")

        communities = []
        results = []

        with self._driver.session() as session:

            try:
                results = session.execute_read(cypher.fetch_communities, comm_type)
            except Exception as e:
                logger.warning(f"Issue fetching communities for type {comm_type}: {e}")

            for r in results:

                if r['names'] in [["leiden_modularity"], ["louvain_modularity"]]:  # skip GraphMetric rows
                    continue

                comm = Community(
                    community_type=comm_type,
                    community_id=r["community_id"],
                    community_size=r["community_size"],
                    entity_ids=r["entity_ids"],
                    entity_names=r["names"],
                    relationship_ids=r["relationship_ids"],
                    relationship_types=r["relationship_types"],
                )

                # attach the chunks that mention this community's entities
                comm.chunks = []
                for element_id in r["chunk_ids"]:
                    try:
                        c_res = session.execute_read(cypher.fetch_chunk, element_id=element_id)
                        comm.chunks.append(Chunk(chunk_id=c_res[0]["chunk_id"], text=c_res[0]["text"]))
                    except Exception as e:
                        logger.warning(f"Issue fetching chunk with elementId {element_id}: {e}")

                communities.append(comm)

            return communities

    def store_community_reports(self, reports: List[CommunityReport]):
        """Persist LLM-generated community summaries into the graph + vector index.

        Each in-memory `CommunityReport` becomes a searchable
        ``(:CommunityReport)`` node carrying its summary, embedding, and the
        community metadata the QA agents filter on::

            CommunityReport (in-memory)        Neo4j
            ┌────────────────────┐              (:CommunityReport {
            │ summary            │  add_embeddings>   summary,
            │ summary_embeddings │ ──────────────>    summary_embeddings,
            │ community_type     │                    community_type,
            │ community_id       │                    community_id,
            │ community_size     │                    community_size })
            └────────────────────┘                        │
                                                         ▼
                                              vector index "reports" (cr_store)
                                              -> similarity_search in QA

        After all reports are written, the ``reports`` vector index is (re)created.
        """
        for report in reports:

            if report is None:
                continue

            metadatas = {
                "community_type": report.community_type,
                "community_id": report.community_id,
                "community_size": report.community_size,
            }

            try:
                self.cr_store.add_embeddings(
                    texts=[report.summary],
                    embeddings=[report.summary_embeddings],
                    metadatas=[metadatas],
                )
            except Exception as e:
                logger.warning(f"Error saving Community Report: {e}")

        try:
            self.cr_store.create_new_index()
        except Exception as e:
            logger.warning(f"Error creating Index for CommunityReports: {e}")
