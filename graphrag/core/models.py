"""
models.py — pydantic data models for the whole project
=======================================================

Single home for every data model, so callers import from one place::

    from graphrag.core.models import (
        Chunk, ProcessedDocument,                 # ingestion / retrieval carriers
        Ontology,                                 # extraction schema constraints
        _Node, _Relationship, _Graph,             # GraphExtractor structured output
        Community, CommunityReport,               # community detection + summaries
    )

These are pure *shape* (pydantic ``BaseModel`` / langchain ``Serializable``);
no graph backend, no config, no provider SDKs. Keeping this module
side-effect-free is what lets ``graphrag.config`` and ``graphrag.ontologies``
depend on it without creating import cycles.

This module supersedes the former ``graphrag/schema.py`` and
``graphrag/graph/graph_model.py``.
"""

from typing import List, Dict, Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, AliasChoices
from langchain_core.load.serializable import Serializable
from langchain_neo4j.graphs.graph_document import Node, Relationship


# --------------------------------------------------------------------------- #
# Ingestion / retrieval carriers
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    chunk_id: int | str
    text: str
    filename: Optional[str] = None
    embedding: Optional[List[float]] = None
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embeddings_model: Optional[str] = None
    nodes: Optional[List[Node]] = None
    relationships: Optional[List[Relationship]] = None


class ProcessedDocument(BaseModel):
    filename: str = ""
    source: str = ""
    document_version: int = 1
    metadata: Optional[dict] = None
    chunks: Optional[List[Chunk]] = None


# --------------------------------------------------------------------------- #
# Graph-extraction structured output
# --------------------------------------------------------------------------- #
class _Node(Serializable):
    id: str
    type: str
    properties: Optional[Dict[str, str]] = None


class _Relationship(Serializable):
    source: str
    target: str
    type: str
    properties: Optional[Dict[str, str]] = None


class _Graph(Serializable):
    """
    Represents a graph consisting of nodes and relationships.

    Attributes:
        `nodes (List[_Node])`: A list of nodes in the graph.
        `relationships (List[_Relationship])`: A list of relationships in the graph.
    """
    nodes: List[_Node]
    relationships: List[_Relationship]


class Ontology(BaseModel):
    """
    Describes arbitrary, project-specific allowed labels and relationships.

    Labels should map to `Node.type` and relationships to `Relationship.type` from
    `langchain_neo4j.graphs.graph_document`. A functional description of what
    labels and relationships represent in the domain may also be provided.
    """
    allowed_labels: Optional[List[str]] = None
    labels_descriptions: Optional[Dict[str, str]] = None
    allowed_relations: Optional[List[str]] = None


# --------------------------------------------------------------------------- #
# Community detection + summaries
# --------------------------------------------------------------------------- #
class Community(BaseModel):
    """
    Describes a community in the Knowledge Graph.

    `community_type`: `str`   — `leiden` or `louvain`
    `community_id`: `int`     — the id of this community stored on graph nodes
    `community_size`: `Optional[int]` — number of nodes in the community
    `entity_ids` / `entity_names`: entities related to the community
    `relationship_ids` / `relationship_types`: relationships in the community
    `chunks`: chunks that belong to (or mention entities of) the community
    `attributes`: `Optional[Dict[str, Any]]` — any extra attributes
    `table_repr`: `Optional[pd.DataFrame]`   — tabular representation
    """
    community_type: str
    community_id: int
    community_size: Optional[int] = None
    entity_ids: Optional[List[str]] = None
    entity_names: Optional[List[str]] = None
    relationship_ids: Optional[List[str]] = None
    relationship_types: Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = None
    chunks: Optional[List[Chunk]] = None
    table_repr: Optional[pd.DataFrame] = None  # TODO how to fetch this?

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CommunityReport(BaseModel):
    """
    Summary report from a given `Community`.

    `community_type` still accepts the legacy (misspelled) key `communtiy_type`
    at construction for backward compatibility; attribute access is always
    `community_type`.
    """
    community_type: str = Field(
        validation_alias=AliasChoices("community_type", "communtiy_type")
    )
    community_id: int
    summary: str = ""
    rank: float = 0.0
    community_size: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None
    summary_embeddings: Optional[List[float]] = None
