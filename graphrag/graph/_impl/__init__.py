"""graphrag.graph._impl — private implementation of the graph backend.

Everything here is an internal implementation detail of `KnowledgeGraph`:
the behaviour mixins it composes from (`metadata` / `ingestion` / `analysis` /
`queries`), the Cypher statements & callbacks (`cypher`), and the networkx
algorithms (`algorithms`).

External code must import `KnowledgeGraph` from `graphrag.graph`, never
anything from this subpackage. The leading underscore is the contract: this
package's layout is not a public API and may change.
"""
