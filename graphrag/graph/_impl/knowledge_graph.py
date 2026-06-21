"""
knowledge_graph.py — KnowledgeGraph facade
===========================================

`KnowledgeGraph` is the single entry point for the graph layer. Its behaviour
is composed from focused mixins (under `mixins/`):

    · MetadataMixin   (mixins/metadata.py)   read-only stats & modularity
    · IngestionMixin  (mixins/ingestion.py)  write path: docs/chunks/graph
    · AnalysisMixin   (mixins/analysis.py)   community detection, centralities
    · QueryMixin      (mixins/queries.py)    read path: vector search + traversal

The mixins build on `base/` (`cypher.py` Cypher statements, `algorithms.py`
networkx algorithms). This file only keeps construction: credentials, the two
Neo4jVector stores and the `Neo4jGraph` super-init.
"""

from langchain_core.embeddings import Embeddings
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from graphrag.core.config import KnowledgeGraphConfig
from .mixins.analysis import AnalysisMixin
from .mixins.ingestion import IngestionMixin
from .mixins.metadata import MetadataMixin
from .mixins.queries import QueryMixin
from graphrag.core.logger import get_logger


logger = get_logger(__name__)

BASE_ENTITY_LABEL = "__Entity__"


class KnowledgeGraph(MetadataMixin, IngestionMixin, AnalysisMixin, QueryMixin, Neo4jGraph):
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

        if conf.ontology:
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

    def verify_connection(self) -> bool:
        """Return True if the Neo4j driver authenticates successfully.

        Wraps the underlying `_driver.verify_authentication()` so callers don't
        reach into the langchain private attribute.
        """
        try:
            return self._driver.verify_authentication()
        except Exception:
            return False
