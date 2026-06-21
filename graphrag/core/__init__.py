"""graphrag.core — core layer: data models, configuration, and factories.

Pure / low-level modules the rest of the toolkit depends on. All are
"module-as-interface" — import symbols directly from each module:

· ``config``     — pydantic configuration models (``Configuration``,
                   ``KnowledgeGraphConfig``, ``LLMConf``, ``EmbedderConf``,
                   ``ChunkerConf``, ``Source``, ``ModelType``, ``ChunkerType``).
· ``models``     — pydantic data models (``Chunk``, ``ProcessedDocument``,
                   ``Ontology``, ``Community``, ``CommunityReport``,
                   ``_Node`` / ``_Relationship`` / ``_Graph``).
· ``logger``     — ``get_logger``, the project-wide logging helper.
· ``llm``        — ``fetch_llm``, builds LangChain LLM clients from config
                   (provider SDKs imported lazily).
· ``embeddings`` — ``get_embeddings``, builds embedding clients from config.

These stay side-effect-free / lazy so they can be imported anywhere without
pulling in the graph backend or a provider SDK::

    from graphrag.core.config import Configuration, LLMConf
    from graphrag.core.models import Chunk, Ontology
    from graphrag.core.logger import get_logger
    from graphrag.core.llm import fetch_llm
    from graphrag.core.embeddings import get_embeddings
"""
