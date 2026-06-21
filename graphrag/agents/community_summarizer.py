from typing import List, Optional
from graphrag.graph import KnowledgeGraph
from graphrag.core.embeddings import get_embeddings
from graphrag.core.llm import fetch_llm
from graphrag.core.config import LLMConf, EmbedderConf
from graphrag.core.models import Community, CommunityReport
from langchain.prompts import PromptTemplate
from graphrag.core.logger import get_logger


logger = get_logger(__name__)


def get_summarize_community_prompt() -> PromptTemplate:

    prompt = """
        Your task is to synthetize a summary from the given context.

        Do not mention anything else, just summarize a precise, clear and helpful summary.
        Do not make things up or add any information on your own.

        CONTEXT: {context}

        SUMMARY:
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['context']

    return template


class CommunitiesSummarizer:
    """ 
    Agent in charge of producing summaries of Community Reports. 
    """
    
    def __init__(
        self, 
        llm_conf: LLMConf, 
        embeddings_conf: EmbedderConf
        ):
        self.llm = fetch_llm(llm_conf)
        self.embeddings = get_embeddings(embeddings_conf)
        self.summarize_community_prompt = get_summarize_community_prompt()
        
        
    def get_reports(self, communities: List[Community]) -> List[CommunityReport]:
        """
        Generate Community Reports for available communities in the Graph.
        """
        reports = []

        for comm in communities:
            report = self.get_community_report(comm)
            if report is not None:
                reports.append(report)

        return reports


    def get_community_report(self, community: Community) -> Optional[CommunityReport]:
        """
        Generates a CommunityReport for a given community, out of chunks available in said community.
        It will also embed the summary to make it retrievable
        """
        if not community.chunks:
            logger.warning(f"There are no Chunks to summarize for community {community.community_type}: {community.community_id}")
            return None

        chunks_content = "\n\n".join(
            chunk.text.replace("\n\n", "\n") for chunk in community.chunks
        )

        try:
            summary = self.llm.invoke(
                input=self.summarize_community_prompt.format(
                    context=chunks_content
                )
            ).content
        except Exception as e:
            logger.warning(f"Issue summarizing Chunks for community {community.community_type}: {community.community_id}: {e}")
            return None

        try:
            summary_embeddings = self.embeddings.embed_documents([summary])[0]
        except Exception as e:
            logger.warning(f"Issue embedding Summary for community {community.community_type}: {community.community_id}: {e}")
            return None

        report = CommunityReport(
            community_type=community.community_type,
            community_id=community.community_id,
            summary=summary,
            community_size=community.community_size,
            summary_embeddings=summary_embeddings
        )

        return report
        
    
        
        
        
        
        