"""graphrag.graph._impl.mixins — behaviour mixins composed into KnowledgeGraph.

Each mixin contributes one slice of `KnowledgeGraph`'s behaviour and is
composed into the facade defined in the parent ``knowledge_graph.py``:

· ``metadata``  — MetadataMixin:  read-only graph statistics & modularity
· ``ingestion`` — IngestionMixin: write path (docs/chunks/extracted-graph)
· ``analysis``  — AnalysisMixin:  community detection, centralities, reports
· ``queries``   — QueryMixin:     read path (vector search + traversal for QA)

The mixins build on the building blocks in ``../base/`` (`cypher`, `algorithms`).
"""
