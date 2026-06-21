"""
queries.py — QueryMixin
=======================

Read-side helpers for the QA agents: thin wrappers around the two vector
stores (`vector_store` over Chunks, `cr_store` over CommunityReports) and the
`cypher` traversal functions.

Without this mixin every agent had to open its own `_driver.session()` and
poke at `vector_store` / `cr_store` directly. Agents now call these methods
instead, so the driver and store handles stay inside the graph layer.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from graphrag.graph._impl.base import cypher
from graphrag.core.models import Chunk
from graphrag.core.logger import get_logger


logger = get_logger(__name__)


class QueryMixin:
    """Provides `KnowledgeGraph` with the read methods used by QA agents."""

    # --- vector search -------------------------------------------------

    def search_chunks(
        self, query: str, filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Vector similarity search over `Chunk` nodes."""
        return self.vector_store.similarity_search(query=query, filter=filter)

    def search_reports(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        with_scores: bool = False,
        score_threshold: Optional[float] = None,
    ):
        """Vector search over `CommunityReport` nodes.

        When `with_scores=True`, returns `(Document, score)` tuples and
        optionally applies `score_threshold`; otherwise returns plain Documents.
        """
        if with_scores:
            return self.cr_store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter,
                score_threshold=score_threshold,
            )
        return self.cr_store.similarity_search(query=query, k=k, filter=filter)

    # --- graph traversal ----------------------------------------------

    def adjacent_chunks(
        self, chunk: Chunk, use_elementId: bool = False
    ) -> Tuple[Optional[Chunk], Chunk, Optional[Chunk]]:
        """Return the (previous, current, next) `Chunk` triple for `chunk`."""
        with self._driver.session(database=self._database) as session:
            return cypher.get_adjacent_chunks(
                session, chunk, use_elementId=use_elementId
            )

    def mentioned_entities(
        self, chunk: Chunk, use_elementId: bool = False
    ) -> List[Dict[str, Any]]:
        """Entities mentioned by `chunk` via the `MENTIONS` relationship."""
        with self._driver.session(database=self._database) as session:
            return cypher.get_mentioned_entities(
                session, chunk, use_elementId=use_elementId
            )

    def community_subgraph(
        self, community_ids: List[int], community_type: str = "leiden"
    ) -> List[Dict[str, Any]]:
        """Subgraph (as a list of `{node_1, relationship, node_2}` dicts)."""
        with self._driver.session(database=self._database) as session:
            return cypher.filter_graph_by_communities(
                session, community_ids=community_ids, community_type=community_type
            )
