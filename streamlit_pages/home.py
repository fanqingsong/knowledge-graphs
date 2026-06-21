import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)

st.write("## Welcome to the Knowledge Graphs Demo! 👋")

st.markdown(
    """
    ## Graph RAG VS 'Normal' RAG
    This is a Demo Application to showcase the possibilities enabled by the
    Graph RAG approach.  
    For "Graph RAG" we mean [Retrieval Augmented Generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) 
    grounded and contextualized not only via vector search or hybrid search 
    but also by querying a graph of entities and relationships using an LLM-powered agent 
    and then feeding the resulting context to another agent in charge of answering the 
    user's original question.  

    Explaining here the advantages of Retrieval Augmented Generation enriched with a Knowledge Graph 
    would be too long, but you can follow this [link](https://neo4j.com/developer-blog/knowledge-graphs-llms-multi-hop-question-answering/)
    to the Neo4j's Developer Blog.

    ## What's in the Demo?
    In this demo, pages of the Web App are used to serve different purposes: 
    """
)


st.page_link(
    page=st.Page("streamlit_pages/upload.py"),
    icon="🗳️",
    label="Upload one document and ingest it into a Knowledge Graph representation in your Neo4j database;"
)
st.page_link(
    page=st.Page("streamlit_pages/chat.py"),
    icon="🦜",
    label="Chat with the Knowledge Graph using LLM-powered Agents"
)
# st.page_link(
#     page=st.Page("streamlit_pages/display.py"),
#     icon="🕸️",
#     label="Display the Knowledge Graph or parts of it asking in natural language"
# )
# st.page_link(
#     page=st.Page("streamlit_pages/config.py"),
#     icon="⚙️",
#     label="Update/customize the Configuration for this web app"
# )

st.markdown(
    """
    ## Prerequisites
    In order to showcase this approach to RAG, we will need some tools. 
    If you are using the Dockerized version of this app, some of them are already set up for you in the DockerFile. 
    * **Neo4j**: in this demo app, [Neo4j](https://neo4j.com/) is used both as a Vector Store as well as a Graph Database; 
        in fact, during the ingestion process, each Document is transformed in a node, 
        and from it `Chunk` nodes are extracted (with `embeddings` as metadata for that node), 
        while an agent is used to produce a graph representation of the content of the document. 
    * **LLM & Embeddings APIs**: To power agents you need an LLM and an embeddings model.
        The demo defaults to **Zhipu GLM** (`glm-4.7`) for LLMs and **SiliconFlow** (`BAAI/bge-m3`) for embeddings — both OpenAI-compatible endpoints. Other configured providers (OpenAI, Azure OpenAI, Groq, Google, Ollama) are also supported via the `ModelType` enum in `graphrag/core/config.py`.
    * **Documents**: Documents coming from a specific domain to ingest; available formats are `.pdf`, `.docx`, `.txt`, `.html`.
    * **Configuration**: in order for this Demo to work, you should either have all the settings for Neo4j, LLMs..
        inside your environment or a configuration file at the following path: `knowledge-graphs/config_example.env`.
        See the repository's README for further details on the settings.
    """
)
