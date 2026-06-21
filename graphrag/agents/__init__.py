"""
graphrag.agents — LLM agents that build and query the knowledge graph
=====================================================================

导读 of the agent layer; import these names directly from the package::

    from graphrag.agents import GraphExtractor, CommunitiesSummarizer, GraphAgentResponder

· GraphExtractor        — text -> graph (nodes + relationships), constrained by
                          an `Ontology` (allowed labels / relations).
· CommunitiesSummarizer — produces per-community summaries from Chunks and
                          embeds them so they are retrievable.
· GraphAgentResponder   — the QA entry point: answers via Cypher, vector RAG,
                          community reports, or community subgraphs.
"""

from graphrag.agents.graph_extractor import GraphExtractor
from graphrag.agents.community_summarizer import CommunitiesSummarizer
from graphrag.agents.graph_qa import GraphAgentResponder

__all__ = ["GraphExtractor", "CommunitiesSummarizer", "GraphAgentResponder"]
