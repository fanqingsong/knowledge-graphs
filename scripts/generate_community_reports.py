"""
One-off utility: generate Community Reports for already-ingested data.

The ingestion pipeline detects communities but, out of the box, never produces
the CommunityReport summaries that the `Communities` and `Subgraph` answering
modes retrieve from. This script back-fills those reports from existing
communities (Leiden + Louvain) without re-ingesting documents.

Run inside the app container:
    docker exec -it kg-app python scripts/generate_community_reports.py
"""
import os
from dotenv import load_dotenv

from graphrag.config import LLMConf, EmbedderConf, KnowledgeGraphConfig
from graphrag.graph.knowledge_graph import KnowledgeGraph
from graphrag.ingestion.embedder import ChunkEmbedder
from graphrag.agents.community_summarizer import CommunitiesSummarizer

# config_example.env is baked into the image; container env (from docker
# env_file=.env) takes precedence because load_dotenv does not override.
load_dotenv("config_example.env")


def main():
    db = KnowledgeGraphConfig(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        index_name=os.environ.get("INDEX_NAME", "vector"),
    )
    emb_conf = EmbedderConf(
        type=os.environ["EMBEDDINGS_TYPE"],
        model=os.environ["EMBEDDINGS_MODEL_NAME"],
        api_key=os.environ["EMBEDDINGS_API_KEY"],
        endpoint=os.environ["EMBEDDINGS_ENDPOINT"],
    )
    qa_conf = LLMConf(
        type=os.environ["QA_MODEL_TYPE"],
        model=os.environ["QA_MODEL_NAME"],
        api_key=os.environ["QA_API_KEY"],
        endpoint=os.environ["QA_MODEL_ENDPOINT"],
    )

    embedder = ChunkEmbedder(conf=emb_conf)
    kg = KnowledgeGraph(conf=db, embeddings_model=embedder.embeddings)

    if not kg._driver.verify_authentication():
        raise RuntimeError("Neo4j authentication failed — check NEO4J_* config.")

    summarizer = CommunitiesSummarizer(
        llm_conf=qa_conf,
        embeddings_conf=emb_conf,
    )

    for comm_type in ["leiden", "louvain"]:
        communities = kg.get_communities(comm_type=comm_type)
        reports = summarizer.get_reports(communities)
        stored = [r for r in reports if r is not None]
        kg.store_community_reports(reports)
        print(f"[{comm_type}] communities={len(communities)} reports_stored={len(stored)}")

    print("Done. Communities / Subgraph answering modes are now usable.")


if __name__ == "__main__":
    main()
