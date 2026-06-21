"""
pipeline.py — IngestionPipeline
===============================

Orchestrates the full ingestion run end to end::

    load → clean → chunk → embed → graph-mine
    → KnowledgeGraph.add_documents → communities → community reports

It composes the pipeline ``Stage``s (each with the unified ``process``
interface) plus the graph-write / community steps that don't return a document
stream. Moving this orchestration out of the Streamlit UI lets it be reused by
scripts, tests, or any other entry point — the UI only supplies a progress
callback::

    from graphrag.ingestion import IngestionPipeline
    pipeline = IngestionPipeline(conf=conf, knowledge_graph=kg)
    docs = pipeline.run(on_progress=print)
"""

from typing import Callable, List, Optional

from graphrag.core.config import Configuration
from graphrag.core.models import ProcessedDocument
from graphrag.core.logger import get_logger
from graphrag.agents.community_summarizer import CommunitiesSummarizer
from graphrag.graph import KnowledgeGraph

from graphrag.ingestion._impl.ingestor import LocalIngestor
from graphrag.ingestion._impl.cleaner import Cleaner
from graphrag.ingestion._impl.chunker import Chunker
from graphrag.ingestion._impl.embedder import ChunkEmbedder
from graphrag.ingestion._impl.graph_miner import GraphMiner

logger = get_logger(__name__)


class IngestionPipeline:
    """End-to-end ingestion orchestrator (stages → graph write → community reports)."""

    def __init__(self, conf: Configuration, knowledge_graph: KnowledgeGraph):
        self.conf = conf
        self.knowledge_graph = knowledge_graph
        self.ingestor = LocalIngestor(source=conf.source_conf)
        self.cleaner = Cleaner()
        self.chunker = Chunker(conf=conf.chunker_conf)
        self.embedder = ChunkEmbedder(conf=conf.embedder_conf)
        self.graph_miner = GraphMiner(
            conf=conf.re_model_conf,
            ontology=conf.database.ontology,
        )

    def run(self, on_progress: Optional[Callable[[str], None]] = None) -> List[ProcessedDocument]:
        """Run the full pipeline; ``on_progress`` receives a status message per step."""

        def _step(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        _step("Loading..")
        docs = self.ingestor.batch_ingest()

        _step("Cleaning..")
        docs = self.cleaner.process(docs)

        _step("Chunking..")
        docs = self.chunker.process(docs)

        _step("Embedding..")
        docs = self.embedder.process(docs)

        _step("Extracting a Knowledge Graph from each file..")
        docs = self.graph_miner.process(docs)

        _step("Uploading Data to Knowledge Graph..")
        self.knowledge_graph.add_documents(docs)

        _step("Updating Communities and computing Centralities in the Graph..")
        self.knowledge_graph.update_centralities_and_communities()

        _step("Summarizing Communities into Reports..")
        summarizer = CommunitiesSummarizer(
            llm_conf=self.conf.qa_model,
            embeddings_conf=self.conf.embedder_conf,
        )
        for comm_type in ["leiden", "louvain"]:
            communities = self.knowledge_graph.get_communities(comm_type=comm_type)
            reports = summarizer.get_reports(communities)
            self.knowledge_graph.store_community_reports(reports)
            _step(f"  · {comm_type}: {len(reports)} community reports stored")

        return docs
