"""
stage.py — unified pipeline interface
=====================================

Every document-transformation step of the ingestion pipeline implements this
single ``process`` contract, so the orchestrator (``streamlit_pages/upload.py``)
depends on one stable interface instead of each stage's bespoke method name
(``clean_documents`` / ``chunk_documents`` / ``embed_documents_chunks`` /
``mine_graph_from_docs``).

Why this exists (SOLID):

· Open/Closed      — a new step implements `Stage` and is appended to the
                     pipeline; the orchestrator does not change.
· Interface Seg.   — the orchestrator talks to one `process` method, not a
                     different verb per stage.
· Testability      — stages are interchangeable / mockable behind the protocol.

A `Stage` takes a list of `ProcessedDocument` and returns a list of
`ProcessedDocument` (documents flow through, mutated in place or replaced)::

    class MyStage:
        def process(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
            ...

Side-effecting graph writes (`KnowledgeGraph.add_documents`,
`update_centralities_and_communities`, `store_community_reports`) are NOT
`Stage`s — they don't return a document stream — and stay in the orchestrator.
"""

from typing import List, Protocol

from graphrag.core.models import ProcessedDocument


class Stage(Protocol):
    """Document-transformation step of the ingestion pipeline."""

    def process(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        ...
