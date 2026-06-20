"""
ingestion.py — IngestionMixin
=============================

The *write* path of `KnowledgeGraph`: vector-index management and writing
documents/chunks/extracted-graph into Neo4j.

The transaction callbacks themselves live in `cypher.py`; this mixin only
orchestrates the sessions and the per-chunk bookkeeping.
"""

from typing import List

from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_document import GraphDocument

from graphrag.graph import cypher
from graphrag.schema import ProcessedDocument
from graphrag.utils.logger import get_logger


logger = get_logger(__name__)


class IngestionMixin:
    """Provides `KnowledgeGraph` with ingestion / write methods."""

    def create_document_node(self, doc: ProcessedDocument):
        """Creates a Document node in the Knowledge Graph."""
        with self._driver.session(database=self._database) as session:
            session.execute_write(cypher.create_document_node, doc)
            session.execute_write(
                cypher.create_part_of_relationships,
                doc.filename,
                doc.document_version,
            )
            logger.info(f"Document node created for file: {doc.filename}")

    def create_next_relationships(self, filename: str, doc_version: int):
        """Creates NEXT relationships between Chunk Nodes from a Document."""
        with self._driver.session(database=self._database) as session:
            session.execute_write(cypher.create_next_relationships, filename, doc_version)
            logger.info(f"NEXT relationships created for Document {filename} version {doc_version}")

    def create_mentions_relationships(
        self,
        node_id: str,
        chunk_id: int,
        filename: str,
        document_version: int,
    ):
        """Creates MENTIONS relationships between Chunk and __Entity__ nodes."""
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                cypher.create_mentions_relationships,
                node_id,
                chunk_id,
                filename,
                document_version,
            )
            logger.info("MENTIONS relationships created!")

    def store_chunks_for_doc(self, doc: ProcessedDocument):
        """
        Stores Chunk nodes for a `ProcessedDocument` into the Knowledge Graph and updates the
        Knowledge Graph itself with the graphs extracted from each chunk, if any.
        """

        for chunk in doc.chunks:

            # doc level metadata
            metadata = doc.metadata if doc.metadata else {}
            metadata["filename"] = doc.filename
            metadata["document_version"] = doc.document_version
            # chunk level metadata
            metadata["chunk_id"] = chunk.chunk_id
            metadata["chunk_size"] = chunk.chunk_size
            metadata["chunk_overlap"] = chunk.chunk_overlap
            metadata["embeddings_model"] = chunk.embeddings_model

            try:
                self.vector_store.add_embeddings(
                    texts=[chunk.text],
                    embeddings=chunk.embedding,
                    metadatas=[metadata],
                )
            except Exception as e:
                logger.warning(f"Error storing chunk for document {doc.filename}: {e}")

            # store chunk's graph
            if chunk.nodes is not None:

                graph_doc: GraphDocument = GraphDocument(
                    nodes=chunk.nodes,
                    relationships=chunk.relationships if chunk.relationships is not None else [],
                    source=Document(page_content=chunk.text),
                )

                try:
                    self.add_graph_documents(
                        graph_documents=[graph_doc],
                        include_source=False,
                        baseEntityLabel=True,
                    )

                    for node in chunk.nodes:
                        self.create_mentions_relationships(
                            node_id=node.id,
                            chunk_id=chunk.chunk_id,
                            filename=doc.filename,
                            document_version=doc.document_version,
                        )
                except Exception as e:
                    logger.warning(f"Error storing graph for chunk {chunk.chunk_id} in document {doc.filename}: {e}")

        try:
            self.create_next_relationships(
                filename=doc.filename,
                doc_version=doc.document_version,
            )
        except Exception as e:
            logger.warning(f"Error creating NEXT relationships for chunks in Document {doc.filename}: {e}")

        try:
            self.create_document_node(doc=doc)
        except Exception as e:
            logger.warning(f"Error creating Document source node for file: {doc.filename}: {e}")

        try:
            self.vector_store.create_new_index()
        except Exception as e:
            logger.warning(f"Error creating Index for chunks: {e}")

    def add_documents(self, docs: List[ProcessedDocument]):
        for doc in docs:
            self.store_chunks_for_doc(doc)
