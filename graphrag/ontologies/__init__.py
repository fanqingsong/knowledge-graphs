"""graphrag.ontologies — pre-defined `Ontology` constants.

Each built-in ontology is a hand-authored `Ontology` instance (from
`graphrag.core.models`) that pins the allowed node labels and relationship
types for a domain. Import the constant and pass it directly to
`KnowledgeGraphConfig(ontology=...)`::

    from graphrag.ontologies import beiyin_ontology

    kg_conf = KnowledgeGraphConfig(..., ontology=beiyin_ontology)

Hand-authored ontologies give a clean, predictable graph schema. (The old
sampling-based `OntologyExplorer` agent and the name-based `ONTOLOGIES` /
`get_ontology()` resolver have been removed — selection is now by passing the
`Ontology` instance in code, not by a `.env` name string.)
"""

from graphrag.ontologies._impl.beiyin import beiyin_ontology

__all__ = ["beiyin_ontology"]
