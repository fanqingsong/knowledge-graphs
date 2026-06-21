"""
graphrag — knowledge-graph RAG toolkit
======================================

This top-level package stays **import-light**: importing it does not pull in the
graph backend, the ingestion pipeline, or any provider SDK. Each subpackage
exposes its own curated interface (导读) via its ``__init__`` — import the one
you need::

    from graphrag.core.config import Configuration, KnowledgeGraphConfig, LLMConf, EmbedderConf
    from graphrag.core.models import Chunk, ProcessedDocument, Ontology, Community, CommunityReport
    from graphrag.core.llm import fetch_llm
    from graphrag.core.embeddings import get_embeddings
    from graphrag.core.logger import get_logger
    from graphrag.graph import KnowledgeGraph
    from graphrag.agents import GraphExtractor, CommunitiesSummarizer, GraphAgentResponder
    from graphrag.ingestion import IngestionPipeline
    from graphrag.ontologies import beiyin_ontology

The core layer (`graphrag.core`) holds the pure data/config models plus the
LLM/embedding factories and the logger; everything else builds on it.
"""
