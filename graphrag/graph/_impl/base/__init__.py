"""graphrag.graph._impl.base — building blocks the mixins build on.

· ``cypher``     — every Cypher statement & transaction callback (the single
                   Cypher entry point used by all mixins).
· ``algorithms`` — pure networkx algorithms (community detection, centralities);
                   no Neo4j access.
"""
