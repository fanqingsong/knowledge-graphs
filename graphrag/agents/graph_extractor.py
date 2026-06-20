from graphrag.utils.logger import get_logger
from typing import Optional

# from langchain_neo4j.graphs.graph_document import Relationship, Node
from langchain.schema import Document

from graphrag.factory.llm import fetch_llm
from graphrag.config import LLMConf
from graphrag.graph.graph_model import Ontology, _Graph
from graphrag.prompts.graph_extractor import get_graph_extractor_prompt


logger = get_logger(__name__)


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