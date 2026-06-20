"""
cypher.py — Cypher queries & transaction callbacks
===================================================

All Neo4j Cypher statements and the `tx`-callback functions used by the
`KnowledgeGraph` write/read paths live here, so the graph mixins stay focused
on orchestration rather than on SQL-like string blobs.

The callbacks keep the `(tx, ...)` signature expected by Neo4j's
`session.execute_write` / `execute_read`.
"""

from neo4j import ManagedTransaction

from graphrag.schema import ProcessedDocument
from graphrag.utils.logger import get_logger


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

