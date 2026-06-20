"""
analysis.py — AnalysisMixin
===========================

The graph-analysis side of `KnowledgeGraph`:

    · get_digraph                         snapshot the whole graph as networkx
    · update_centralities_and_communities run community detection + centralities,
                                          write results back onto nodes
    · update_properties                   low-level node-property writer
    · get_communities                     read communities (with their chunks) back out
    · store_community_reports             persist LLM-generated community summaries

The pure algorithms (Leiden/Louvain/centralities) live in
`graph_data_structure.py`; the Cypher callbacks live in `cypher.py`.
"""

from typing import List, Optional

import networkx as nx

from graphrag.graph import cypher
from graphrag.graph.graph_data_structure import (
    build_update_query,
    compute_centralities,
    detect_leiden_communities,
    detect_louvain_communities,
    update_modularity,
)
from graphrag.graph.graph_model import Community, CommunityReport
from graphrag.schema import Chunk
from graphrag.utils.logger import get_logger


logger = get_logger(__name__)


class AnalysisMixin:
    """Provides `KnowledgeGraph` with community / centrality analysis methods."""

    def get_digraph(self) -> nx.DiGraph:
        """Returns the Knowledge Graph under its `networkx.DiGraph` representation."""
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
        """
        Returns the entity-only subgraph as a `networkx.DiGraph`:
        `__Entity__` nodes and the relationships directly between entities.

        Used in `community_source="entities"` mode so that community detection
        reflects the thematic structure of entities rather than being diluted
        by the (much more numerous) Chunk nodes and their NEXT/PART_OF edges.
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
        """Update Neo4j nodes with Leiden/Louvain communities and centrality scores."""
        with self._driver.session() as session:

            if any([centralities, leiden_communities, louvain_communities]):

                for node, data in G.nodes(data=True):
                    query, params = build_update_query(
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
                update_modularity(session, leiden_modularity, "leiden")
                logger.info("Updated Leiden Modularity property in Graph")

            if louvain_modularity is not None:
                update_modularity(session, louvain_modularity, "louvain")
                logger.info("Updated Louvain Modularity property in Graph")

    def update_centralities_and_communities(self):
        """Computes centralities measures and detects communities in nodes across the Knowledge Graph."""
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
        """Assign each Chunk the majority community of the entities it MENTIONS (both algorithms)."""
        with self._driver.session() as session:
            for comm_type in ("leiden", "louvain"):
                try:
                    session.execute_write(cypher.propagate_communities_to_chunks, comm_type)
                    logger.info(f"Propagated {comm_type} communities to chunks")
                except Exception as e:
                    logger.warning(f"Issue propagating {comm_type} communities to chunks: {e}")

    def get_communities(self, comm_type: str = "leiden") -> List[Community]:
        """Fetches communities from the Knowledge Graph."""
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
        """Stores Community Reports in the Graph, to make them available for GraphRAG strategies."""
        for report in reports:

            if report is None:
                continue

            metadatas = {
                "community_type": report.communtiy_type,
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
