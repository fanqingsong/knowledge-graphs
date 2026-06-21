from graphrag.core.logger import get_logger
from typing import Optional

# from langchain_neo4j.graphs.graph_document import Relationship, Node
from langchain.schema import Document

from graphrag.core.llm import fetch_llm
from graphrag.core.config import LLMConf
from graphrag.core.models import Ontology, _Graph
from langchain.prompts import PromptTemplate


logger = get_logger(__name__)


def get_graph_extractor_prompt() -> PromptTemplate:
    """
    Parses the instructions to give as input to the LLM in charge of
    Relations Extractions.
    """
    prompt= """
        You are a top-tier algorithm designed for extracting information in structured formats to build a Knowledge Graph.

        Your task is to extract informations in the form of Nodes and Relationships from an INPUT TEXT.

        - NODES represent entities and concepts.
        - RELATIONSHIPS represents the connections between nodes.
        - PROPERTIES characterize nodes or relationships.

        ------
        RULES:
        ------

        1. FORMAT
        - You MUST return ONLY the Graph extracted from the INPUT TEXT. Do not add anything else.
        - Remember that a Graph is defined as

        ````
        class Graph(Serializable):
            '''
            Represents a graph consisting of nodes and relationships.

            Attributes:
                nodes (List[Node]): A list of nodes in the graph.
                relationships (List[Relationship]): A list of relationships in the graph.
            '''
            nodes: List[Node]
            relationships: List[Relationship]
        ````

        where Nodes and Relationships are defined as

        ````
        class Node(Serializable):
            '''
            Represents a node in a graph with associated properties.

            Attributes:
                id (str): A unique identifier for the node.
                type (str): The type or label of the node.
                properties (Optional[Dict[str, str]]): Additional properties associated with the node.
            '''
            id: str
            type: str
            properties: Optional[Dict[str, str]]
        ````

        and

        ````
        class Relationship(Serializable):
            '''
            Represents a directed relationship between two nodes in a Graph.

            Attributes:
                source (str): The source node of the relationship.
                target (str): The target node of the relationship.
                type (str): The type of the relationship.
                properties (Optional[Dict[str, str]]): Additional properties associated with the relationship.
            '''
            source: str
            target: str
            type: str
            properties: Optional[Dict[str, str]]
        ````

        2. ALLOWED LABELS AND RELATIONSHIPS
        - If provided with allowed labels and relationship types then you MUST use only those as possible outcomes.
        - If labels and relationships are not provided, you are free to use any label and relationship you see fit.

        ------------

        ALLOWED NODE LABELS: {allowed_labels}
        LABELS DESCRIPTIONS: {labels_descriptions}
        ALLOWED RELATIONSHIPS TYPES: {allowed_relationships}

        ## Begin Extraction!
        INPUT TEXT: {input_text}
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['input_text', 'allowed_labels', 'labels_descriptions', 'allowed_relationships']

    return template


class GraphExtractor:
    """ Agent able to extract informations in a graph representation format from a given text.
    """

    def __init__(self, conf: LLMConf, ontology: Optional[Ontology]=None):
        self.conf = conf
        self.llm = fetch_llm(conf)
        self.prompt = get_graph_extractor_prompt()

        self.prompt.partial_variables = {
            'allowed_labels':ontology.allowed_labels if ontology and ontology.allowed_labels else "", 
            'labels_descriptions': ontology.labels_descriptions if ontology and ontology.labels_descriptions else "", 
            'allowed_relationships': ontology.allowed_relations if ontology and ontology.allowed_relations else ""
        }


    def extract_graph(self, text: str) -> _Graph:
        """ 
        Extracts a graph from a text.
        """

        if self.llm is not None:
            try:
                graph: _Graph = self.llm.with_structured_output(
                    schema=_Graph
                    ).invoke(
                        input=self.prompt.format(input_text=text)
                    )

                return graph 
                
            except Exception as e:
                logger.warning(f"Error while extracting graph: {e}")