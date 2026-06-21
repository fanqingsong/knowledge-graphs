"""
cypher.py — Cypher queries & transaction callbacks
===================================================

All Neo4j Cypher statements and the `tx`-callback functions used by the
`KnowledgeGraph` write/read paths live here, so the graph mixins stay focused
on orchestration rather than on SQL-like string blobs.

The callbacks keep the `(tx, ...)` signature expected by Neo4j's
`session.execute_write` / `execute_read`.
"""

from typing import Any, Dict, List, Tuple

from neo4j import ManagedTransaction, Query, Session

from graphrag.core.models import Chunk, ProcessedDocument
from graphrag.core.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Graph-wide read queries (executed directly with session.run)
# --------------------------------------------------------------------------- #
LABELS_QUERY = "CALL db.labels() YIELD label RETURN COLLECT(label) AS labels"
RELATIONSHIP_TYPES_QUERY = (
    "CALL db.relationshipTypes() YIELD relationshipType "
    "RETURN COLLECT(relationshipType) AS relationship_types"
)
COUNT_NODES_QUERY = "MATCH (n) RETURN COUNT(n) AS value"
COUNT_LABELS_QUERY = "CALL db.labels() YIELD label RETURN COUNT(label) AS value"
COUNT_RELATIONSHIPS_QUERY = "MATCH ()-[r]-() RETURN COUNT(r) AS value"
COUNT_DOCS_QUERY = "MATCH (n:Document) RETURN COUNT(n) AS value"

LEIDEN_MODULARITY_QUERY = (
    "MATCH (m:GraphMetric WHERE m.name = 'leiden_modularity') RETURN m.value AS value"
)
LOUVAIN_MODULARITY_QUERY = (
    "MATCH (m:GraphMetric WHERE m.name = 'louvain_modularity') RETURN m.value AS value"
)

COUNT_LEIDEN_COMMUNITIES_QUERY = """
    MATCH (n)
    WHERE n.community_leiden IS NOT NULL
    RETURN count(DISTINCT n.community_leiden) AS value
"""
COUNT_LOUVAIN_COMMUNITIES_QUERY = """
    MATCH (n)
    WHERE n.community_louvain IS NOT NULL
    RETURN count(DISTINCT n.community_louvain) AS value
"""

# Whole-graph extraction into a networkx DiGraph.
DIGRAPH_NODES_QUERY = """
    MATCH (n)
    RETURN elementId(n) AS node_id, labels(n) AS labels, properties(n) AS properties;
"""
DIGRAPH_RELS_QUERY = """
    MATCH (n)-[r]->(m)
    RETURN elementId(n) AS source, elementId(m) AS target, type(r) AS rel_type, properties(r) AS properties;
"""

# Entity-only subgraph extraction (community_source="entities"): only
# __Entity__ nodes and the relationships directly between entities.
ENTITY_DIGRAPH_NODES_QUERY = """
    MATCH (n:__Entity__)
    RETURN elementId(n) AS node_id, labels(n) AS labels, properties(n) AS properties;
"""
ENTITY_DIGRAPH_RELS_QUERY = """
    MATCH (n:__Entity__)-[r]->(m:__Entity__)
    RETURN elementId(n) AS source, elementId(m) AS target, type(r) AS rel_type, properties(r) AS properties;
"""


# --------------------------------------------------------------------------- #
# Transaction callbacks (write)
# --------------------------------------------------------------------------- #
def create_document_node(tx: ManagedTransaction, doc: ProcessedDocument):
    """Create the `:Document` source node for an ingested file."""
    query = """
        CREATE (d:Document {
            filename: $filename,
            document_version: $document_version
        })
    """
    try:
        tx.run(
            query,
            filename=doc.filename,
            document_version=doc.document_version,
            metadata=doc.metadata,
        )
        logger.info(f"Document node created for file: {doc.filename}")
    except Exception as e:
        logger.warning(f"Error creating Document node for file: {doc.filename}: {e}")


def create_part_of_relationships(tx: ManagedTransaction, filename: str, document_version: int):
    """Link every Chunk of a document to its Document node via `:PART_OF`."""
    query = """
        MATCH (d:Document {filename: $filename, document_version: $document_version})
        MATCH (c:Chunk {filename: $filename, document_version: $document_version})
        MERGE (c)-[:PART_OF]->(d)
    """
    try:
        tx.run(query, filename=filename, document_version=document_version)
        logger.info(f"PART_OF relationships created for Document {filename} version {document_version}")
    except Exception as e:
        logger.warning(f"Error creating PART_OF relationships for Document {filename}: {e}")


def create_next_relationships(tx: ManagedTransaction, filename: str, document_version: int):
    """Chain consecutive Chunks of a document with `:NEXT` edges (chunk_id → chunk_id+1)."""
    query = """
        MATCH (c1:Chunk {filename: $filename, document_version: $document_version})
        WITH c1
        MATCH (c2:Chunk {filename: $filename, document_version: $document_version, chunk_id: c1.chunk_id + 1})
        MERGE (c1)-[:NEXT]->(c2)
    """
    try:
        tx.run(query, filename=filename, document_version=document_version)
    except Exception as e:
        logger.warning(f"Error creating NEXT relationships for chunks in Document {filename}: {e}")


def create_mentions_relationships(
    tx: ManagedTransaction,
    node_id: str,
    chunk_id: int,
    filename: str,
    document_version: int,
):
    """Link a Chunk to an extracted Entity via `:MENTIONS`."""
    query = """
        MATCH (c:Chunk {chunk_id: $chunk_id, filename: $filename, document_version: $document_version})
        MATCH (e:__Entity__ {id: $node_id})
        MERGE (c)-[:MENTIONS]->(e)
    """
    try:
        tx.run(
            query,
            node_id=node_id,
            chunk_id=chunk_id,
            filename=filename,
            document_version=document_version,
        )
    except Exception as e:
        logger.warning(f"Error creating MENTIONS relationships for {node_id}: {e}")


# --------------------------------------------------------------------------- #
# Transaction callbacks (read)
# --------------------------------------------------------------------------- #
def fetch_communities(tx: ManagedTransaction, comm_type: str = "leiden"):
    """
    Aggregate every community of the given type into one row per community,
    collecting its entities, relationships and the chunks that mention them.
    `comm_type` is validated by the caller (`get_communities`) to be in
    {"leiden", "louvain"}, so interpolating it into the query is safe.
    """
    query = f"""
        MATCH (n)-[r]-(m)
        WHERE n.community_{comm_type} IS NOT NULL
        OPTIONAL MATCH (chunk:Chunk) WHERE chunk.community_{comm_type} = n.community_{comm_type}
        WITH
            '{comm_type}' AS community_type,
            n.community_{comm_type} AS community_id,
            count(DISTINCT n) AS community_size,
            collect(DISTINCT elementId(n)) AS entity_ids,
            collect(DISTINCT n.name) AS names,
            collect(DISTINCT elementId(r)) AS relationship_ids,
            collect(DISTINCT type(r)) AS relationship_types,
            collect(DISTINCT elementId(chunk)) AS chunk_ids
        RETURN
            community_type,
            community_id,
            community_size,
            entity_ids,
            names,
            relationship_ids,
            relationship_types,
            chunk_ids
        ORDER BY community_size DESC
    """
    return list(tx.run(query))


def fetch_chunk(tx: ManagedTransaction, element_id: str):
    """
    Fetch a single Chunk's id and text by its Neo4j elementId.
    Uses a parameterized query (no string interpolation) to avoid injection.
    """
    query = """
        MATCH (c:Chunk)
        WHERE elementId(c) = $element_id
        RETURN elementId(c) AS chunk_id, c.text AS text
    """
    return list(tx.run(query, element_id=element_id))


def propagate_communities_to_chunks(tx: ManagedTransaction, comm_type: str):
    """
    Assign each Chunk the community of the majority of the entities it MENTIONS.

    Used only in `community_source="entities"` mode, where detection runs over
    the entity subgraph (so only entities carry a community id). This step
    "pushes" that community down onto the Chunks so that community-based
    retrieval (`filter={community_<type>: ...}`) keeps working. Ties are broken
    by the lowest community id. Chunks that mention no entity keep
    `community_<type> = null` (i.e. they take no part in community retrieval).

    `comm_type` is validated by the caller to be in {"leiden", "louvain"}.
    """
    query = f"""
        MATCH (c:Chunk)-[:MENTIONS]->(e:__Entity__)
        WHERE e.community_{comm_type} IS NOT NULL
        WITH c, e.community_{comm_type} AS cid, count(*) AS freq
        ORDER BY freq DESC, cid
        WITH c, collect(cid)[0] AS top_cid
        SET c.community_{comm_type} = top_cid
    """
    return list(tx.run(query))


# --------------------------------------------------------------------------- #
# Persistence helpers (write) — node-property & graph-metric updates
#
# These assemble / run the Cypher that writes computed analysis results
# (community ids, centralities, modularity) back onto Neo4j. The pure
# algorithms live in `algorithms.py`; only the DB-facing callbacks
# live here, keeping the split "pure algorithms" vs "Neo4j access" clean.
# --------------------------------------------------------------------------- #
def update_modularity(session: Session, mod: float, mod_type: str = "leiden"):
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
    # is something to update).
    # Only add SET if there's something to update
    if set_clauses:
        query += "SET " + ",\n    ".join(set_clauses)  # Join clauses with proper formatting

    # Step G: Return the assembled statement plus its parameters dict, ready to be
    # executed by Neo4j. Values are passed via $placeholders → safe & clear.
    return query, parameters


# --------------------------------------------------------------------------- #
# Traversal reads (multi-record) — used by the QA QueryMixin
#
# These walk the graph beyond a single node: adjacent chunks (via NEXT),
# entities mentioned by a chunk (via MENTIONS), and community subgraphs.
# Unlike the `fetch_*` single-record reads above, these return assembled
# structures (chunk triples / entity dicts / subgraph edge triples).
# (Migrated from the former `graph_queries.py` so all Cypher lives in one place.)
# --------------------------------------------------------------------------- #
def get_adjacent_chunks(
    session: Session,
    chunk: Chunk,
    use_elementId: bool=False
    ) -> Tuple[Chunk | None, Chunk , Chunk | None]:
    """
    Returns a tuple with the previous , current and following `Chunk`
    given an initial node characterised by a `filename` and a `chunk_id`.
    If `use_elementId` is set to `True`, will use the elementId of the chunk instead.
    """
    if use_elementId:
        base_query = """
            MATCH (current:Chunk)
            WHERE elementId(current) = $elementId

            OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(current)
            OPTIONAL MATCH (current)-[:NEXT]->(next:Chunk)

            RETURN prev AS previous_chunk, current, next AS next_chunk
        """
        try:
            result = session.run(base_query, elementId=chunk.chunk_id)
            record = result.single()
        except Exception as e:
            logger.warning(f"Unable to retrieve adjacent chunks for Chunk: {chunk.chunk_id}")
            return None, chunk, None

    else:
        base_query = """
            MATCH (current:Chunk)
            WHERE current.chunk_id = $chunk_id AND current.filename = $filename

            OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(current)
            OPTIONAL MATCH (current)-[:NEXT]->(next:Chunk)

            RETURN prev AS previous_chunk, current, next AS next_chunk
        """

        try:
            result = session.run(base_query, chunk_id=chunk.chunk_id, filename=chunk.filename)
            record = result.single()
        except Exception as e:
            logger.warning(f"Unable to retrieve adjacent chunks for Chunk: {chunk.chunk_id}")
            return None, chunk, None

    previous_chunk = dict(record["previous_chunk"]) if record["previous_chunk"] else None
    if previous_chunk:
        previous_chunk = Chunk(
            chunk_id=previous_chunk['chunk_id'],
            filename=previous_chunk['filename'],
            text=previous_chunk["text"],
        )
        chunk.chunk_id = previous_chunk.chunk_id + 1 # original chunk id
    next_chunk = dict(record["next_chunk"]) if record["next_chunk"] else None
    if next_chunk:
        next_chunk = Chunk(
            chunk_id=next_chunk['chunk_id'],
            filename=next_chunk['filename'],
            text=next_chunk["text"],
        )
        chunk.chunk_id = next_chunk.chunk_id-1 # original chunk id

    return previous_chunk, chunk, next_chunk


def get_mentioned_entities(
    session: Session,
    chunk: Chunk,
    n_hops: int=1,
    use_elementId: bool = False
    ) -> List[Dict[str, Any]]:
    """
    Follows the `MENTIONS` relationships of a given Chunk in the Graph and collects mentioned entities.
    `n_hops` is used to indicate the number of relationship layers that could be done following entities linking.
    """
    nodes = []

    # TODO perform n-hops retrieval
    if use_elementId:
        base_query = """
            MATCH (c:Chunk)
            WHERE elementId(c) = $elementId
            MATCH (c)-[:MENTIONS]->(mentioned)
            RETURN collect(mentioned) AS mentioned_nodes
        """
        try:
            result= session.run(base_query, elementId=chunk.chunk_id)
            record = result.single()
            mentioned_nodes = record["mentioned_nodes"] if record else []
            for node in mentioned_nodes:
                nodes.append(dict(node))

            logger.info(f"Retrieved {len(nodes)} entities for chunk {chunk.chunk_id}")

            return nodes

        except Exception as e:
            logger.warning(f"No mentioned entities retrieved with exception: {e}")
            return []

    else:
        base_query = """
            MATCH (c:Chunk)
            WHERE c.chunk_id = $chunk_id AND c.filename = $filename
            MATCH (c)-[:MENTIONS]->(mentioned)
            RETURN collect(mentioned) AS mentioned_nodes
        """
        try:
            result= session.run(base_query, chunk_id=chunk.chunk_id, filename=chunk.filename)
            record = result.single()
            mentioned_nodes = record["mentioned_nodes"] if record else []
            for node in mentioned_nodes:
                nodes.append(dict(node))

            logger.info(f"Retrieved {len(nodes)} entities for chunk {chunk.chunk_id}")

            return nodes

        except Exception as e:
            logger.warning(f"No mentioned entities retrieved with exception: {e}")
            return []


def filter_graph_by_communities(session: Session, community_ids: List[int], community_type: str="leiden") -> List[Dict[str, Any]]:
    """
    Creates a temporary  view of the Knowledge Graph to filter it into subgraphs given community ids.
    """
    query = f"""
        MATCH (n)-[r]->(m)
        WHERE n.community_{community_type} IN $community_values
            AND NOT n:Chunk
            AND NOT m:Chunk
        RETURN n, r, m
    """

    keys_to_remove = {
        'community_louvain', 'community_leiden', 'pagerank',
        'id', 'betweenness', 'closeness'
    }

    try:
        result = session.run(query, community_values=community_ids)

        subgraph = []

        for record in result:
            node_1 = {k: v for k, v in dict(record["n"]).items() if k not in keys_to_remove}
            node_2 = {k: v for k, v in dict(record["m"]).items() if k not in keys_to_remove}
            relationship = dict(record["r"])

            subgraph.append({
                "node_1": node_1,
                "relationship": relationship,
                "node_2": node_2
            })

        return subgraph

    except Exception as e:
        logger.warning(f"Error while fetching subgraph: {e}")
        return []

