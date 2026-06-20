from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from langchain_openai.chat_models import ChatOpenAI, AzureChatOpenAI
from graphrag.utils.logger import get_logger

from graphrag.config import LLMConf

logger = get_logger(__name__)


def fetch_llm(conf: LLMConf) -> BaseChatModel | None:
    """
    Fetches the LLM model.
    """
    logger.info(f"Fetching LLM model '{conf.model}'..")

    if conf.type == "openai":
        llm = ChatOpenAI(
            model=conf.model,
            api_key=conf.api_key,
            deployment=conf.deployment,
            temperature=conf.temperature,
        )
    elif conf.type == "azure-openai":
        llm = AzureChatOpenAI(
            model=conf.model,
            azure_endpoint=conf.endpoint,
            azure_deployment=conf.deployment,
            api_key=conf.api_key,
            temperature=conf.temperature,
            api_version=conf.api_version
        )
    elif conf.type == "groq":
        llm = ChatGroq(
            model=conf.model, 
            api_key=conf.api_key,
            temperature=conf.temperature,
            max_retries=3
        )
    elif conf.type == "google":
        llm = ChatGoogleGenerativeAI(
            model=conf.model,
            api_key=conf.api_key,
            temperature=conf.temperature,
        )
    elif conf.type == "zhipu":
        # Zhipu AI (智谱 GLM) exposes an OpenAI-compatible Chat Completions API.
        # `endpoint` carries the base_url, e.g. https://open.bigmodel.cn/api/coding/paas/v4/
        llm = ChatOpenAI(
            model=conf.model,
            api_key=conf.api_key,
            base_url=conf.endpoint,
            temperature=conf.temperature,
        )
    else:
        logger.warning(f"LLM type '{conf.type}' not supported.")
        llm = None
    
    logger.info(f"Initialized LLM of type: '{conf.type}'")
    return llm 