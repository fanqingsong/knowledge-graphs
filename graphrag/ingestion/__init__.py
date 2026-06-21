"""graphrag.ingestion — document ingestion pipeline
=================================================

The public interface is `IngestionPipeline`, the high-level entry point that
orchestrates the full run (load → clean → chunk → embed → graph-mine → graph
write → community reports)::

    from graphrag.ingestion import IngestionPipeline
    pipeline = IngestionPipeline(conf=conf, knowledge_graph=kg)
    docs = pipeline.run(on_progress=print)

All stages (`Cleaner`, `Chunker`, `ChunkEmbedder`, `GraphMiner`,
`LocalIngestor`), the `Stage` protocol, and the `Ingestor` base live under
`_impl/` and are NOT re-exported — they are `IngestionPipeline`'s internals.

For an embedder instance outside the pipeline (e.g. to build a
`KnowledgeGraph` / `CommunitiesSummarizer`), use the factory in the core
layer — `from graphrag.core.embeddings import get_embeddings` — not
`ChunkEmbedder` (which is a pipeline stage, not an embedder provider).
"""

from graphrag.ingestion._impl.pipeline import IngestionPipeline

__all__ = ["IngestionPipeline"]
