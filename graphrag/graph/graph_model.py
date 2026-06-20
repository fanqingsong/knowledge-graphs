"""
graph_model.py — pydantic data models for the graph layer
==========================================================

Only *shape* lives here: the entity/relationship/graph containers, the
optional `Ontology`, and the community + community-report models.

All graph *transformation* helpers (networkx <-> dict <-> langchain
GraphDocument, node normalization, property key formatting) have moved to
`graph_converters.py`.
"""

import pandas as pd

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

from langchain.load.serializable import Serializable

from graphrag.schema import Chunk


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

    -----------
    Attributes:
    -----------
        `nodes (List[_Node])`: A list of nodes in the graph.
        `relationships (List[_Relationship])`: A list of relationships in the graph.
    """
    nodes: List[_Node]
    relationships: List[_Relationship]


class Ontology(BaseModel):
    """
    Used to describe arbitrary, project-specific allowed labels and relationships.

    Labels should map to `Node.type` and relationships to `Relationship.type` from
    `langchain_neo4j.graphs.graph_document`. It is allowed to provide a functional
    description of what labels and relationships represent in the domain.
    """
    allowed_labels: Optional[List[str]] = None
    labels_descriptions: Optional[Dict[str, str]] = None
    allowed_relations: Optional[List[str]] = None


class Community(BaseModel):
    """
    Describes a community in the Knowledge Graph.

    -----------
    Attributes:
    -----------
    `community_type`: `str`
        The type of community, such as `leiden` or `louvain`
    `community_id`: `int`
        The identifier of this community in the graph nodes properties
    `community_size`: `Optional[int]`
        The number of nodes in the graph with attribute 'community_type: community_id'
    `entity_ids`: `Optional[List[str]]`
        List of entity IDs related to the community
    `relationship_ids`: `Optional[List[str]]`
        List of relationship IDs related to the community
    `table_repr`: `Optional[pd.DataFrame]`
        Table Representation of the community
    `attributes`: `Optional[Dict[str, Any]]`
        A dictionary of additional attributes associated with the community
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
    Summary report from a given `Community`

    -----------
    Attributes:
    -----------
    `community_type`: `str`
        The type of community, such as `leiden` or `louvain`
    `community_id`: `int`
        The identifier of this community in the graph nodes properties
    `summary`: `str`
        Summary of the report
    community_size`: `Optional[int]`
        The number of nodes in the graph with attribute 'community_type: community_id'
    `rank`: `float`
        Used for sorting. The higher the better.
    `attributes`: `Optional[Dict[str, Any]]`
        A dictionary of additional attributes associated with the report
    """
    communtiy_type: str
    community_id: int
    summary: str = ""
    rank: float = 0.0
    community_size: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None
    summary_embeddings: Optional[List[float]] = None
