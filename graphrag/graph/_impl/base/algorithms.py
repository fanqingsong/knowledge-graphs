"""
algorithms.py — Graph Algorithm Core (graph data structure)
=====================================================================

Responsibility: run "community detection" and "centrality computation" on the
in-memory networkx graph. Pure algorithms only — no Neo4j access here.
Called by knowledge_graph.py (via the AnalysisMixin).

Position of this module in the project:

    knowledge_graph.py  (orchestrator)
            │ calls
            ▼
    algorithms.py  (the "algorithm engineer" doing the actual work)
        detect_louvain_communities  community detection (split graph into groups)
        detect_leiden_communities   community detection (alternative algorithm)
        compute_centralities        how "important" each node is

The DB-facing side of analysis (writing the computed values back to Neo4j:
node-property updates via `build_update_query`, graph-level modularity via
`update_modularity`) lives in `cypher.py`, so this module stays a pure
networkx-only algorithm layer.

Key concepts:
    · Community Detection: cluster tightly-connected nodes into groups, each
      group is a "community". Louvain / Leiden are two such algorithms.
    · Modularity: a score measuring "how good the clustering is", roughly in
      [-0.5, 1]. Closer to 1 means "dense inside each group, sparse between
      groups" → a better partition.
    · Centrality: how important a single node is, measured from several angles.

Data flow overview:
    ① detect_louvain / ② detect_leiden / ③ compute_centralities  → "compute" in memory
        (each node is now decorated with the computed values; the caller then
         persists them to Neo4j via the callbacks in `cypher.py`)
"""

import community
import networkx as nx

from igraph import Graph
from leidenalg import find_partition, ModularityVertexPartition
from graphrag.core.logger import get_logger
from typing import Tuple


logger = get_logger(__name__)


def detect_louvain_communities(G: nx.DiGraph, return_modularity:bool=True) -> nx.DiGraph | Tuple[nx.DiGraph, float]:
    """
    Detects Louvain communities for a `networkx` Directed Graph.
    If `return_modularity`, also return the modularity of the Graph according to
    the Louvain distance measure.

    Purpose: cluster the graph with the Louvain algorithm and tag each node with
    the community it belongs to.

        Before (loose nodes, some closer together)      After (auto-split into groups)
            A──B   E──F                                     [A──B]   [E──F]
            │  │   │  │            ──►                      [│  │]   [│  │]
            C──D   G──H                                     [C──D]   [G──H]
                                                        ──group1── ──group2──
    """
    # Step A: Louvain only works on undirected graphs (edges have no direction),
    #         so convert the directed graph to undirected first. This treats both
    #         A→B and A←B as simply "A and B are connected".
    G_undirected = G.to_undirected()

    # Step B: Actually run the Louvain algorithm. Returns `partition` — a dict
    #         {node: group_id}, e.g. {A:0, B:0, C:0, E:1, ...}
    partition = community.best_partition(G_undirected)  # Louvain method

    # Step C: Attach the community id to each node as the attribute
    #         `community_louvain` — like handing each person a "badge" showing
    #         which department they belong to.
    nx.set_node_attributes(G, partition, "community_louvain")  # Store communities in node attributes

    if not return_modularity:

        return G

    else:
        # Step D: Compute the modularity score (clustering quality) of this
        #         partition — the higher, the better.
        modularity = community.modularity(partition, G_undirected)

        logger.info(f"Modularity based on Louvain communities: {modularity}")

        return G, modularity


def detect_leiden_communities(G: nx.DiGraph, return_modularity:bool=True) -> nx.DiGraph | Tuple[nx.DiGraph, float]:
    """
    Detects Leiden communities for a `networkx` Directed Graph.
    If `return_modularity`, also return the modularity of the Graph according to
    the Louvain distance measure.

    Purpose: same goal as detect_louvain_communities (clustering), but using the
    more advanced Leiden algorithm.

    Why is the code longer? Because the Leiden library (leidenalg) only works
    with igraph graphs, not networkx graphs, so the graph must first be
    "translated" into igraph format:

        networkx graph (used by the project)      igraph graph (required by leidenalg)
            ┌───┐         convert                      ┌───┐
            │ A │  (nodes are string names)   ──►      │ 0 │  (nodes are integers 0,1,2...)
            └───┘                                      └───┘

    In short: Louvain may produce groups that are internally disconnected,
    whereas Leiden adds a "refinement" step guaranteeing each group is internally
    connected, so it is more stable. Both are run so the results can be compared.
    """

    # Step A: Build a mapping node-name → integer id (igraph requires integers),
    #         e.g. A→0, B→1, C→2 ...
    # Convert networkx to igraph
    mapping = {node: i for i, node in enumerate(G.nodes())}  # Node mapping
    # Step B: Build the reverse mapping integer-id → node-name, used later to
    #         translate results back into node names.
    reverse_mapping = {i: node for node, i in mapping.items()}

    # Step C: Rebuild an equivalent graph inside igraph using integer ids
    #         (keeping the directed edges).
    # Create igraph graph
    ig_G = Graph(directed=True)
    ig_G.add_vertices(len(G.nodes()))
    ig_G.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])

    # Step D: Run the Leiden algorithm. Returns a list of node ids per community.
    partition = find_partition(ig_G, ModularityVertexPartition)

    # Step E: Write the community labels back to the original networkx graph.
    #         Use reverse_mapping to turn integers back into node names, then tag
    #         each node with `community_leiden`.
    # Assign community labels back to NetworkX
    for i, comm in enumerate(partition):
        for node in comm:
            G.nodes[reverse_mapping[node]]["community_leiden"] = i

    if not return_modularity:
        return G

    else:
        # Step F: Leiden directly provides the modularity score, no need to
        #         compute it separately.
        modularity = partition.modularity

        logger.info(f"Modularity based on Leiden communities: {modularity}")

        return G, modularity


def compute_centralities(G: nx.DiGraph | nx.Graph) -> nx.DiGraph | nx.Graph:
    """
    Compute PageRank, Betweenness and Closeness Centralities and store them as metadata in the graph

    Purpose: measure how important each node is, from three different angles,
    and store the scores back on each node. Using a social-network analogy:

      ① PageRank    "prestige score" — being pointed to by important nodes
                      makes you important (the idea behind Google's page ranking).

      ② Betweenness "bridge score"  — remove you and many people can no longer
                      reach each other (an information hub / choke point).

      ③ Closeness   "front-row score" — you are close to everyone, so messages
                      spread fastest (good as a broadcast source).
    """

    # ① PageRank: nodes pointed to by "important" nodes score higher. alpha=0.85
    #    means an 85% chance to follow a link and a 15% chance to jump randomly
    #    (avoids dead ends).
    pr = nx.pagerank(G, alpha=0.85)
    # ② Betweenness: the more "shortest paths" passing through this node, the
    #    higher its value — it is the "bridge" connecting different groups.
    bc = nx.betweenness_centrality(G)
    # ③ Closeness: the shorter the average distance from this node to all other
    #    nodes, the higher its value.
    cc = nx.closeness_centrality(G)

    # Attach the three scores to each node as attributes.
    nx.set_node_attributes(G, pr, "pagerank")
    nx.set_node_attributes(G, bc, "betweenness")
    nx.set_node_attributes(G, cc, "closeness")

    return G
