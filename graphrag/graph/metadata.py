"""
metadata.py — MetadataMixin
============================

Read-only graph statistics exposed as properties on `KnowledgeGraph`
(node/label/relationship/document counts, community counts, modularity scores).

All of them follow the same shape ("run one scalar query, return the value"),
which is collapsed into the `_scalar` helper so each property is a one-liner
instead of repeating the session/try/return boilerplate.
"""

from typing import List

from graphrag.graph import cypher
from graphrag.utils.logger import get_logger


logger = get_logger(__name__)


class MetadataMixin:
    """Provides `KnowledgeGraph` with its read-only metric properties."""

    def _scalar(self, query: str):
        """Run a query that returns a single row with a single `value` column."""
        with self._driver.session(database=self._database) as session:
            return session.run(query).single()["value"]

    @property
    def labels(self) -> List[str]:
        """Returns a list of labels in the Knowledge Graph."""
        with self._driver.session(database=self._database) as session:
            return session.run(cypher.LABELS_QUERY).single()["labels"]

    @property
    def relationships(self) -> List[str]:
        """Returns a list of relationships in the Knowledge Graph."""
        with self._driver.session(database=self._database) as session:
            return session.run(cypher.RELATIONSHIP_TYPES_QUERY).single()["relationship_types"]

    @property
    def number_of_nodes(self) -> int:
        """Returns the total number of nodes in the Knowledge Graph."""
        return self._scalar(cypher.COUNT_NODES_QUERY)

    @property
    def number_of_labels(self) -> int:
        """Returns the number of labels in the Knowledge Graph."""
        return self._scalar(cypher.COUNT_LABELS_QUERY)

    @property
    def number_of_relationships(self) -> int:
        """Returns the total number of relationships in the Knowledge Graph."""
        return self._scalar(cypher.COUNT_RELATIONSHIPS_QUERY)

    @property
    def number_of_docs(self) -> int:
        """Returns the current number of documents collected in the Knowledge Graph."""
        return self._scalar(cypher.COUNT_DOCS_QUERY)

    @property
    def leiden_modularity(self):
        """Returns the Leiden modularity score stored on the graph (None if not computed)."""
        try:
            return self._scalar(cypher.LEIDEN_MODULARITY_QUERY)
        except Exception:
            logger.warning("Leiden Modularity has not been computed")

    @property
    def louvain_modularity(self):
        """Returns the Louvain modularity score stored on the graph (None if not computed)."""
        try:
            return self._scalar(cypher.LOUVAIN_MODULARITY_QUERY)
        except Exception:
            logger.warning("Louvain Modularity has not been computed")

    @property
    def number_of_leiden_communities(self):
        """Number of distinct Leiden communities on the graph (None if not detected)."""
        try:
            return self._scalar(cypher.COUNT_LEIDEN_COMMUNITIES_QUERY)
        except Exception:
            logger.warning("Leiden communities have not been detected yet")

    @property
    def number_of_louvain_communities(self):
        """Number of distinct Louvain communities on the graph (None if not detected)."""
        try:
            return self._scalar(cypher.COUNT_LOUVAIN_COMMUNITIES_QUERY)
        except Exception:
            logger.warning("Louvain communities have not been detected yet")
