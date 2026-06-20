"""
graph_converters.py — graph format conversion utilities
========================================================

Helpers that translate the project's `_Graph` model (entity/relationship
containers produced by the extractor) into langchain_neo4j `GraphDocument`s
ready to be written into Neo4j.

(Other historical conversions — networkx<->dict round-trips, node-name
normalization, Neo4j property-key formatting — were unused and have been
removed. If you need them back, see git history.)
"""

from typing import List

from langchain.schema import Document
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

from graphrag.graph.graph_model import _Graph, _Node, _Relationship


def map_to_lc_node(node: _Node) -> Node:
    """Maps the `_Graph` `_Node` to the `langchain_neo4j.graphs.graph_document.Node`"""
    properties = node.properties if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return Node(
        id=node.id.title(),
        type=node.type.capitalize(),
        properties=properties
    )


def map_to_lc_relationship(rel: _Relationship, nodes: List[_Node]) -> Relationship:
    """Maps the `_Graph` `_Relationship`  to the `langchain_neo4j.graphs.graph_document.Relationship`"""

    source_node = [node for node in nodes if node.id == rel.source][0]
    target_node = [node for node in nodes if node.id == rel.target][0]

    source = map_to_lc_node(source_node)
    target = map_to_lc_node(target_node)

    properties = rel.properties if rel.properties else {}

    return Relationship(
        source=source,
        target=target,
        type=rel.type,
        properties=properties
    )


def map_to_lc_graph(graph: _Graph, source_content: str) -> GraphDocument:
    """
    Maps the `_Graph` class to the
    `langchain_neo4j.graphs.graph_document.GraphDocument` class
    """
    nodes = [map_to_lc_node(node) for node in graph.nodes]

    relationships = [map_to_lc_relationship(rel, graph.nodes) for rel in graph.relationships]

    graph_doc = GraphDocument(
        nodes=nodes,
        relationships=relationships,
        source=Document(page_content=source_content)
    )

    return graph_doc
