from typing import List, Optional, Any, Dict, Tuple

from langchain_core.messages import BaseMessage
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain

from graphrag.core.config import LLMConf
from graphrag.graph import KnowledgeGraph
from graphrag.core.llm import fetch_llm
from langchain.prompts import PromptTemplate
from graphrag.core.models import Chunk
from graphrag.core.logger import get_logger


logger = get_logger(__name__)


def get_question_answering_prompt() -> PromptTemplate:

    prompt = """
        You are a helpful virtual assistant.

        Your task is provide a relevant and precise answer to the user's question, given context information.
        You might also be given the conversation record of your previous interactions with the user.

        Do not make things up or add any information on your own.
        If the context is not relevant to the user's question, just say that you don't know.
        Maintain the core information from the context.

        CHAT HISTORY: {history}

        QUESTION: {question}

        CONTEXT: {context}

        HELPFUL ANSWER:
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ["history", "question", "context"]

    return template


def get_rephrase_prompt() -> PromptTemplate:
    """Build the prompt that rewrites a user question to match the graph schema.

    Used when ``GraphAgentResponder`` is configured with ``rephrase_llm_conf``.
    ``_rephrase()`` invokes this template (with ``graph_labels`` / ``graph_relationships``
    bound at init) before ``answer_with_cypher()`` and ``answer()``, so the Cypher chain
    receives a question phrased in terms of Document, Chunk, and graph relationships
    instead of colloquial wording. The LLM must not invent facts—only rephrase.

    Examples:
        - User: "第三章讲了什么？"
          Rephrased: "What text appears in the Chunk reached by two NEXT hops from the
          first Chunk of the Document?"
        - User: "这篇文档里提到了哪些人？"
          Rephrased: "Which Person nodes are linked to Chunk nodes via MENTIONS?"
        - User: "刚才那份 PDF 的结论是什么？"  (history mentions 背影.pdf)
          Rephrased: "What do Chunk nodes with PART_OF to the Document whose filename
          is '背影.pdf' say about the conclusion?"
    """

    prompt = """
        Your task is to rephrase a user's question based on the schema of a Graph Database that will be given to you.
        You might also be given the conversation record of previous interaction between the user and a virtual assistant, to provide you additional context.

        The schema is made of node labels and relationships available in the Graph.

        Remember that in a Knowledge Graph there are Documents and Chunks.
        * a node with label `Document` always has a property `filename` (every Document has a name);
        * a node with label `Chunk` is connected via a `PART_OF` relationship to a node with the `Document` label (Chunks are pieces of text coming from a Document);
        * a node with label `Chunk` always has a `text` property;
        * a node with label `Chunk` is usually connected to other nodes with label `Chunk` by a `NEXT` relationship (Chunks are ordered in a sequential order);
        * a node with label `Chunk` might be connected to other nodes in the Graph by a `MENTIONS` relationship (text in Chunks might mention some relevant entities).

        Do not mention anything else, just rephrase the question from the user to be as coherent as possible with the schema of the graph.
        Do not make things up or add any information on your own.

        CHAT HISTORY: {history}
        AVAILABLE NODE LABELS: {graph_labels}
        AVAILABLE RELATIONSHIPS: {graph_relationships}
        QUESTION: {question}

        REPHRASED_QUESTION:
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['history', 'graph_labels', 'graph_relationships', 'question']

    return template


def get_qa_prompt_with_subgraph() -> PromptTemplate:

    prompt = """
        You are a helpful virtual assistant.
        Your task is provide a relevant and precise answer to the user's question, given context information from a Knowledge Graph.
        You might also be given the conversation record of your previous interactions with the user.

        In the context you will find:
        * one or more SUMMARY OF COMMUNITY CHUNKS;
        * the COMMUNITY GRAPH represented as a list of dictionaries;
        * the CHUNKS in that community;
        * the MENTIONED ENTITIES in each chunk.

        Do not make things up or add any information on your own.
        If the context is not relevant to the user's question, just say that you don't know.
        Maintain the core information from the context.

        CHAT HISTORY: {history}

        QUESTION: {question}

        CONTEXT: {context}

        HELPFUL ANSWER:
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ["history", "context", "question"]

    return template


def get_summarization_prompt() -> PromptTemplate:

    prompt = """
        Your task is to synthetize a clear and helpful answer to a question.

        The sources of information to use for your task come from a Vector Database and from a Graph Database.
        You might also be given the conversation record of your previous interactions with the user.

        In your task, you MUST use the context obtained from a vector search on the Vector Database
        and the query results given running a Cypher Query on the Graph Database.
        If one of the sources is empty, just answer the question using the other source.

        Do not mention anything else, just summarize a precise, clear and helpful answer.
        Do not make things up or add any information on your own.

        CHAT HISTORY: {history}

        QUESTION: {question}

        RETRIEVED CONTEXT: {retrieved_context}

        QUERY RESULT ON GRAPH: {query_result}

        ANSWER:
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['history', 'question', 'retrieved_context', 'query_result']

    return template


class GraphAgentResponder:
    """
    Agent powered by up to three LLMs, is able to answer a user's question
    navigating the `KnowledgeGraph` via Cypher Queries as well as via Vector Search.
    """

    def __init__(
        self,
        qa_llm_conf: LLMConf,
        cypher_llm_conf: LLMConf,
        graph: KnowledgeGraph,
        rephrase_llm_conf: Optional[LLMConf]=None
    ):
        self.graph = graph
        self.qa_llm = fetch_llm(qa_llm_conf)
        self.cypher_llm = fetch_llm(cypher_llm_conf)
        self.qa_prompt = get_question_answering_prompt()
        self.qa_prompt_with_subgraph = get_qa_prompt_with_subgraph()

        self.summarize_prompt = get_summarization_prompt()

        self.graph_qa_chain = GraphCypherQAChain.from_llm(
            qa_llm=self.qa_llm,
            cypher_llm=self.cypher_llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True,
            validate_cypher=True,
            return_intermediate_steps=True
        )
        self.rephrase_llm = None
        if rephrase_llm_conf:
            self.rephrase_llm = fetch_llm(rephrase_llm_conf)
            self.rephrase_prompt = get_rephrase_prompt()
            self.rephrase_prompt.partial_variables = {
                "graph_labels": self.graph.labels,
                "graph_relationships": self.graph.relationships
            }


    # ------------------------------------------------------------------
    # context-building helpers
    # ------------------------------------------------------------------

    def _rephrase(self, query: str, history: Optional[str]) -> Optional[str]:
        """Rephrase the user question against the graph schema, if a rephrase LLM is configured."""
        if not self.rephrase_llm:
            return None
        try:
            rephrased = self.rephrase_llm.invoke(
                input=self.rephrase_prompt.format(question=query, history=history)
            ).content
            logger.info(f"Rephrased Question: {rephrased}")
            return rephrased
        except Exception as e:
            logger.warning(f"Failed to rephrase user question with exception: {e}")
            return None

    @staticmethod
    def _doc_to_chunk(doc) -> Chunk:
        return Chunk(
            chunk_id=doc.metadata["chunk_id"],
            text=doc.page_content,
            filename=doc.metadata["filename"],
        )

    def _chunks_context(self, context_docs, use_adjacent_chunks: bool=False) -> str:
        """Join retrieved chunk docs into a context string.

        When `use_adjacent_chunks=True`, expand each doc with its previous/next
        chunk by following the `NEXT` relationship in the graph (higher latency).
        """
        if not use_adjacent_chunks:
            return "\n".join(f"\n {doc.page_content}" for doc in context_docs)

        parts: List[str] = []
        for doc in context_docs:
            prev_chunk, current_chunk, next_chunk = self.graph.adjacent_chunks(
                self._doc_to_chunk(doc)
            )
            if prev_chunk is not None:
                parts.append(f"\n {prev_chunk.text}")
            parts.append(f"\n {current_chunk.text}")
            if next_chunk is not None:
                parts.append(f"\n {next_chunk.text}")
        return "".join(parts)


    # ------------------------------------------------------------------
    # answering strategies
    # ------------------------------------------------------------------

    def answer_with_cypher(
        self,
        query: str,
        intermediate_steps: bool=False,
        history: Optional[str]=None
    ) -> Optional[str | Tuple[str, List]]:
        """
        Uses only the Cypher chain to answer the user's question.
        """
        rephrased_question = self._rephrase(query, history)
        question = rephrased_question if rephrased_question is not None else query

        try:
            graph_qa_output = self.graph_qa_chain.invoke(inputs={"query": question})
            if intermediate_steps:
                return graph_qa_output["result"], graph_qa_output["intermediate_steps"]
            else:
                return graph_qa_output["result"]
        except Exception as e:
            logger.warning(f"Problem Answering with CYPHER chain: {e}")
            return None


    def answer_with_context(
        self,
        query: str,
        use_adjacent_chunks: bool=False,
        history: Optional[str]=None
    )-> str:
        """
        Uses only vanilla RAG to answer the user's question.
        If `use_adjacent_chunks=True` will query the graph for additional context
        compared to the Chunks retrieved by the similarity search. Latency will be higher due to expanded context.
        """
        try:
            context_docs = self.graph.search_chunks(query=query)
        except Exception as e:
            logger.warning(f"Failed to retrieve context with exception: {e}")
            context_docs = []

        context = self._chunks_context(context_docs, use_adjacent_chunks)

        answer: BaseMessage = self.qa_llm.invoke(
            input=self.qa_prompt.format(
                history=history,
                question=query,
                context=context
            )
        )

        return answer.content


    def answer_with_community_reports(
        self,
        query: str,
        use_adjacent_chunks: bool=False,
        community_type: str="leiden",
        history: Optional[str]=None
    ) -> str:
        """
        Queries two vector indexes to get the user's answer out of an ensemble of contexts:
            1. one made of a list of `CommunityReport`
            2. one made of a list of `Chunk` from the same communities of the reports.

        If `use_adjacent_chunks=True` will query the graph for additional context
        compared to the Chunks retrieved by the similarity search. Latency will be higher due to expanded context.
        """
        context = ""

        try:
            reports_and_scores = self.graph.search_reports(
                query=query,
                k=3,
                filter={"community_type": community_type},
                with_scores=True,
                score_threshold=0.8,
            )
            logger.info(f"Retrieved {len(reports_and_scores)} Community Reports")
        except Exception as e:
            logger.warning(f"Failed to retrieve Community Reports with exception: {e}")
            reports_and_scores = []

        for report, score in reports_and_scores:

            context += f"SUMMARY OF CHUNKS: \n {report.page_content} \n"

            try:
                # fetch only similar chunks in the community
                community_chunks = self.graph.search_chunks(
                    query=query,
                    filter={f"community_{community_type}": report.metadata['community_id']}
                )
                logger.info(f"Retrieved {len(community_chunks)} Chunks for community: {report.metadata['community_id']}")

            except Exception as e:
                logger.warning(f"Failed to enrich context with chunks from community: {report.metadata['community_id']}")
                community_chunks = []

            context += f"CHUNKS: \n"
            context += self._chunks_context(community_chunks, use_adjacent_chunks)

        answer: BaseMessage = self.qa_llm.invoke(
            input=self.qa_prompt.format(
                question=query,
                context=context,
                history=history
            )
        )

        return answer.content


    def answer_with_community_subgraph(
        self,
        query: str,
        community_type: str = "leiden",
        history: Optional[str] = None
    ) -> str:
        """Answer using community reports plus entity structure and chunk-level mentions.

        Retrieval flow:
            1. Vector-search the most relevant ``CommunityReport`` (``k=1``).
            2. Enrich context with graph traversals and chunk search inside that community.
            3. Synthesize an answer via ``qa_prompt_with_subgraph``.

        ``community_subgraph`` (``KnowledgeGraph.community_subgraph``):
            Cypher-filters the graph to non-``Chunk`` nodes that share the report's
            ``community_{community_type}`` id, returning every intra-community edge as
            ``{node_1, relationship, node_2}`` dicts (community metadata stripped).
            Gives the LLM the **entity-relationship skeleton** of the community—e.g.
            who is linked to whom or which concepts co-occur—without raw chunk text.

        ``mentioned_entities`` (``KnowledgeGraph.mentioned_entities``):
            For each chunk retrieved by similarity search within the community, follows
            ``(Chunk)-[:MENTIONS]->(entity)`` and returns the mentioned entity nodes
            (as property dicts; here only ``name`` is written into context).
            Grounds each chunk's **text evidence** in the specific entities that passage
            refers to, bridging narrative content and the community's entity graph.

        Together, the report summarizes the theme, ``community_subgraph`` shows how
        entities relate, and chunk + ``mentioned_entities`` pairs tie facts to sources.

        Example (query: "父亲在车站为作者做了什么？", community_id=3):

            CommunityReport::
                "该社区围绕《背影》中浦口车站送别场景，涉及朱自清与父亲在月台的互动。"

            community_subgraph → COMMUNITY GRAPH in context::
                {"node_1": {"name": "父亲"}, "relationship": {"type": "RELATED_TO"},
                 "node_2": {"name": "浦口车站"}}
                {"node_1": {"name": "朱自清"}, "relationship": {"type": "RELATED_TO"},
                 "node_2": {"name": "父亲"}}

            search_chunks (filtered to community 3) + mentioned_entities per chunk::
                CHUNK CONTENT: "...父亲蹒跚地走到铁道边，慢慢探身下去..."
                MENTIONED ENTITIES: 父亲 \\n 月台 \\n 铁道

            The LLM sees the report (theme), the subgraph (who connects to whom),
            and each passage with its entity anchors, then answers in one pass.
        """
        context = ""

        try:
            reports = self.graph.search_reports(
                query=query,
                k=1,
                filter={"community_type": community_type},
            )
            for report in reports:
                logger.info(f"Retrieved Community Reports of type {community_type} with community id: {report.metadata['community_id']}")

        except Exception as e:
            logger.warning(f"Failed to retrieve Community Reports with exception: {e}")
            reports = []


        for report in reports:

            context += f"SUMMARY OF COMMUNITY CHUNKS: \n {report.page_content} \n"

            community_subgraph = self.graph.community_subgraph(
                community_ids=[report.metadata['community_id']],
                community_type=community_type,
            )

            context += f"COMMUNITY GRAPH: {community_subgraph} \n --------------------------------------- \n "

            context += f"COMMUNITY CHUNKS: "

            try:
                # fetch only similar chunks in the community
                community_chunks = self.graph.search_chunks(
                    query=query,
                    filter={f"community_{community_type}": report.metadata['community_id']}
                )
                logger.info(f"Retrieved {len(community_chunks)} Chunks for community: {report.metadata['community_id']}")

                for chunk in community_chunks:

                    context += f" \n --------------------------------------- \n CHUNK CONTENT: \n {chunk.page_content} \n "
                    context += f"MENTIONED ENTITIES: \n"

                    mentioned_entities = self.graph.mentioned_entities(
                        self._doc_to_chunk(chunk), use_elementId=False
                    )

                    for ent_dict in mentioned_entities:
                        context += f"{ent_dict['name']} \n"

            except Exception as e:
                logger.warning(f"Failed to enrich context with chunks from community: {report.metadata['community_id']}")

        answer: BaseMessage = self.qa_llm.invoke(
            input=self.qa_prompt_with_subgraph.format(
                question=query,
                context=context,
                history=history
            )
        )

        return answer.content


    def answer(
        self,
        query: str,
        use_adjacent_chunks: bool=False,
        filter:Optional[Dict[str, Any]]=None,
        history: Optional[str] = None
    ) -> str:
        """
        Answers the user query performing text generation after having retrieved
        context both via Vector Search and Cypher Queries.
        Results from both this methods are synthetized in a comprehensive answer.

        If a configuration is provided for the rephrasing LLM, it will be used
        to rephrase the user's query according to the `KnowledgeGraph` schema.
        """
        try:
            context_docs = self.graph.search_chunks(query=query, filter=filter)
        except Exception as e:
            logger.warning(f"Failed to retrieve context with exception: {e}")
            context_docs = []

        context = self._chunks_context(context_docs, use_adjacent_chunks)

        cypher_steps = None
        cypher_result = self.answer_with_cypher(query=query, intermediate_steps=True)
        if cypher_result is not None:
            _, cypher_steps = cypher_result
        else:
            logger.warning("Unable to run Cypher chain for this question")

        final_answer: BaseMessage = self.qa_llm.invoke(
            input=self.summarize_prompt.format(
                history=history,
                question=query,
                retrieved_context=context,
                query_result=cypher_steps
            )
        )

        return final_answer.content
