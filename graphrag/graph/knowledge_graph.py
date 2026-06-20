"""
knowledge_graph.py — KnowledgeGraph facade
===========================================

`KnowledgeGraph` is the single entry point for the graph layer. It used to be a
~700-line god class; the behaviour now lives in focused mixins:

    · MetadataMixin   (metadata.py)   read-only graph statistics & modularity
    · IngestionMixin  (ingestion.py)  write path: docs/chunks/extracted-graph
    · AnalysisMixin   (analysis.py)   community detection, centralities, reports

This file only keeps construction: credentials, the two Neo4jVector stores
(`vector_store` for Chunks, `cr_store` for CommunityReports) and the
`Neo4jGraph` super-init.

The public API (constructor signature, attributes `_driver` / `vector_store` /
`cr_store`, all properties and methods) is unchanged.
"""

from langchain_core.embeddings import Embeddings
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from graphrag.config import KnowledgeGraphConfig
from graphrag.graph.analysis import AnalysisMixin
from graphrag.graph.ingestion import IngestionMixin
from graphrag.graph.metadata import MetadataMixin
from graphrag.utils.logger import get_logger


logger = get_logger(__name__)

BASE_ENTITY_LABEL = "__Entity__"


class KnowledgeGraph(MetadataMixin, IngestionMixin, AnalysisMixin, Neo4jGraph):
    """
    Class used to represent a Knowledge Base under graph representation,
    using `neo4j` as the backend for querying operations.

    If an `Ontology` is provided (see `KnowledgeGraphConfig.ontology`), will not allow for nodes and relationships
    to be created outside of the given sets of allowed labels and relationships.
    """

    def __init__(
            self,
            conf: KnowledgeGraphConfig,
            embeddings_model: Embeddings,
            sanitize=False,
            refresh_schema=True,
            enhanced_schema=False,
        ):

        self.url = conf.uri if conf.uri is not None else f"{conf.db_schema}://{conf.host_name}:{conf.port}"
        self.username = conf.user
        self.password = conf.password
        self.database = conf.database
        self.timeout = conf.timeout
        self.index_name = conf.index_name
        # Which graph community detection runs over: "full" | "entities".
        self.community_source = conf.community_source

        if conf.ontology:  # TODO
            self.allowed_labels = conf.ontology.allowed_labels
            self.allowed_relationships = conf.ontology.allowed_relations

        self.embeddings = embeddings_model

        # Vector store over Chunk nodes (used for similarity search in QA).
        try:
            self.vector_store = Neo4jVector(
                embedding=self.embeddings,
                url=self.url,
                username=self.username,
                database=self.database,
                password=self.password,
                index_name=self.index_name,
                node_label="Chunk",
                embedding_node_property="embedding",
                text_node_property="text",
            )
        except Exception as e:
            logger.warning(f"Error connecting to Neo4jVector: {e}")

        # Vector store over CommunityReport nodes (used by Communities/Subgraph QA).
        try:
            self.cr_store = Neo4jVector(
                embedding=self.embeddings,
                url=self.url,
                username=self.username,
                database=self.database,
                password=self.password,
                index_name="reports",
                node_label="CommunityReport",
                embedding_node_property="summary_embeddings",
                text_node_property="summary",
            )
        except Exception as e:
            logger.warning(f"Error connecting to Neo4jVector: {e}")

        super().__init__(
            url=self.url,
            username=self.username,
            password=self.password,
            database=self.database,
            timeout=self.timeout,
            sanitize=sanitize,
            refresh_schema=refresh_schema,
            enhanced_schema=enhanced_schema,
        )
