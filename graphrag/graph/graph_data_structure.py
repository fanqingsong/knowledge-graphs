"""
graph_data_structure.py — Graph Algorithm Core (graph data structure)
=====================================================================

Responsibility: run "community detection" and "centrality computation" on the
in-memory networkx graph, then write the results back to Neo4j.
Called by knowledge_graph.py.

Position of this module in the project:

    knowledge_graph.py  (orchestrator)
            │ calls
            ▼
    graph_data_structure.py  (the "algorithm engineer" doing the actual work)
        detect_louvain_communities  community detection (split graph into groups)
        detect_leiden_communities   community detection (alternative algorithm)
        compute_centralities        how "important" each node is
        update_modularity           store the global quality metric to the DB
        build_update_query          assemble the DB update statement

Key concepts:
    · Community Detection: cluster tightly-connected nodes into groups, each
      group is a "community". Louvain / Leiden are two such algorithms.
    · Modularity: a score measuring "how good the clustering is", roughly in
      [-0.5, 1]. Closer to 1 means "dense inside each group, sparse between
      groups" → a better partition.
    · Centrality: how important a single node is, measured from several angles.

Data flow overview:
    ① detect_louvain / ② detect_leiden / ③ compute_centralities  → "compute" in memory
                              │ (each node is now decorated with the computed values)
                              ▼
              ⑤ build_update_query → assemble UPDATE statement → write back to each node in Neo4j
                              ▼
              ④ update_modularity → store the whole-graph quality score in a GraphMetric node
"""

import community
import networkx as nx

from igraph import Graph
from leidenalg import find_partition, ModularityVertexPartition
from graphrag.utils.logger import get_logger
from neo4j import Query, Session
from typing import Any, Dict, Tuple


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


def update_modularity(session: Session, mod: float, mod_type: str="leiden"):
    """
    Save Leiden or Louvain modularity score as a graph-wide property (inside a node).

    Purpose: store the whole-graph modularity score (clustering quality) into
    Neo4j as a global metric of the graph.

    A dedicated node is created in the database to hold this score:

        ┌──────────────────────────────┐
        │ (:GraphMetric)               │
        │   name: "leiden_modularity"  │  ← which algorithm's score
        │   value: 0.72                │  ← clustering quality score
        └──────────────────────────────┘

    params:
    -------
    `session`: `Session`
        Neo4j Session
    `mod`: `float`
        Modularity score
    `mod_type`: `str`
        Either `leiden` or `louvain`
    """
    # Step A: Only leiden / louvain are supported; anything else raises.
    if mod_type in ["leiden", "louvain"]:
        try:
            # Steps B+D: MERGE = create the GraphMetric node if it doesn't exist,
            #            otherwise reuse it, then SET the score.
            #            Note: the value is passed via the `$modularity` parameter
            #            instead of string interpolation → prevents Cypher
            #            injection and avoids special characters breaking the query.
            session.run(
                f"""MERGE (m:GraphMetric {{name: '{mod_type}_modularity'}}) SET m.value = $modularity""",
                modularity=mod
            )
        except Exception as e:
            # Only log on failure, don't crash — modularity is just an auxiliary
            # metric and should not take down the whole pipeline.
            logger.warning(f"Issue updating Leiden modularity property: {e}")
    else:
        raise NotImplementedError("This Modularity type has not been implemented.")


def build_update_query(
        node_id,
        centralities=False,
        leiden_communities=False,
        louvain_communities=False,
        community_leiden: int=-1,
        community_louvain: int=-1,
        pagerank: float=0.0,
        betweenness: float=0.0,
        closeness: float=0.0
    ) -> Tuple[Query, Dict[str, Any]]:
    """
    Returns `Query` and `dict`with parameters to update node properies

    Purpose: dynamically assemble a Cypher update statement. The centralities
    and community labels computed by the functions above are ultimately written
    back to each node in the database through this one.

    Why "dynamic"? Because not every call needs to update all fields (sometimes
    only communities, sometimes only centralities), so clauses are added on
    demand. The assembled statement looks like:

        MATCH (n) WHERE elementId(n) = $node_id          -- locate the target node
        SET n.community_leiden = $community_leiden,      -- fields to update
            n.community_louvain = $community_louvain,
            n.pagerank = $pagerank,
            n.betweenness = $betweenness,
            n.closeness = $closeness

    With a matching parameters dict: {"node_id": "...", "community_leiden": 0, ...}
    """

    # Step A: Base statement — locate the node to update by its internal id.
    # Base query
    query = "MATCH (n) WHERE elementId(n) = $node_id\n"

    # Step B: Prepare an empty list; append a SET clause for each field to update.
    # List to hold SET clauses
    set_clauses = []
    parameters = {"node_id": node_id}

    # Step C: On demand — if the caller wants to update Leiden communities, add
    #         this SET clause.
    if leiden_communities:
        set_clauses.append("n.community_leiden = $community_leiden")
        parameters["community_leiden"] = community_leiden

    # Step D: On demand — same for Louvain communities.
    if louvain_communities:
        set_clauses.append("n.community_louvain = $community_louvain")
        parameters["community_louvain"] = community_louvain

    # Step E: On demand — add the three centralities together (they are usually
    #         computed together).
    if centralities:
        set_clauses.append("n.pagerank = $pagerank")
        set_clauses.append("n.betweenness = $betweenness")
        set_clauses.append("n.closeness = $closeness")
        parameters.update(
            {"pagerank": pagerank,
             "betweenness": betweenness,
             "closeness": closeness
            }
        )

    # Step F: Join all clauses into a single SET clause with commas (only if there
    #         is something to update).
    # Only add SET if there's something to update
    if set_clauses:
        query += "SET " + ",\n    ".join(set_clauses)  # Join clauses with proper formatting

    # Step G: Return the assembled statement plus its parameters dict, ready to be
    #         executed by Neo4j. Values are passed via $placeholders → safe & clear.
    return query, parameters
