import os
import streamlit as st

from graphrag.ingestion import IngestionPipeline

from streamlit_pages.utils import get_configuration, get_embedder, get_knowledge_graph

st.set_page_config(
    page_title="Upload",
    page_icon="🗳️",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    ## 🗳️ Ingestion of Files in the Graph 🕸️

    Use the box below to upload Files in `.pdf`, `.docx`, `.txt` or `.html` format.
    They will be saved to the app's Source Folder and then ingested into your Knowledge Graph.

    Each file is passed through a pipeline that:
    * loads and cleans its text;
    * splits it into chunks and embeds them;
    * uses an LLM to extract a graph of entities/relationships from each chunk;
    * writes vectors and entities into the Knowledge Graph;
    * runs community detection (Leiden/Louvain) and summarizes each community.
    """
)

SOURCE_FOLDER = f"{os.getcwd()}/source_docs"

conf = get_configuration()

uploaded_files = st.file_uploader(
    label="Upload Files to ingest",
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(SOURCE_FOLDER, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(SOURCE_FOLDER, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

    if st.button(label="Ingest into Knowledge Graph", icon="🗳️"):
        embedder = get_embedder(conf.embedder_conf)
        knowledge_graph = get_knowledge_graph(conf, embedder)

        if not knowledge_graph.verify_connection():
            st.error("Check your Neo4j Configuration!")
        else:
            with st.status(f"Ingesting {len(uploaded_files)} Documents...", expanded=True) as status:
                pipeline = IngestionPipeline(conf=conf, knowledge_graph=knowledge_graph)
                docs = pipeline.run(on_progress=st.write)
                status.update(label="Done with the Ingestion", state="complete", expanded=False)

            st.success(f"Done with the Ingestion of {len(docs)} Files")

            if st.button(label="Cleanup Folder", help=f"Deletes files in {SOURCE_FOLDER}", icon="🗑️"):
                for filename in os.listdir(SOURCE_FOLDER):
                    file_path = os.path.join(SOURCE_FOLDER, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                st.info("Cleanup Completed!")
