"""graphrag.graph package — convenience re-exports for the graph layer."""

from graphrag.graph.knowledge_graph import KnowledgeGraph
from graphrag.graph.graph_model import Community, CommunityReport, Ontology

__all__ = ["KnowledgeGraph", "Community", "CommunityReport", "Ontology"]
