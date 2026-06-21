"""graphrag.graph — Neo4j backend (public interface).

The single public entry point is `KnowledgeGraph`; import it from the package,
not from any internal module::

    from graphrag.graph import KnowledgeGraph

Everything under ``_impl/`` — the mixins `KnowledgeGraph` composes from
(``metadata`` / ``ingestion`` / ``analysis`` / ``queries``), the Cypher
statements & callbacks (``cypher``), and the networkx algorithms
(``algorithms``) — is a private implementation detail. The package ``__init__``
intentionally re-exports only `KnowledgeGraph`; do not import from ``_impl``.
"""

from graphrag.graph._impl.knowledge_graph import KnowledgeGraph

__all__ = ["KnowledgeGraph"]
