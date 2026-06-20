from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from typing import Union

from graphrag.config import EmbedderConf
from graphrag.utils.logger import get_logger


logger = get_logger(__name__)


def get_embeddings(conf: EmbedderConf) -> Union[
    OpenAIEmbeddings,
    AzureOpenAIEmbeddings,
    None
    ]:

        if conf.type == "openai":
            embeddings = OpenAIEmbeddings(
                model=conf.model,
                api_key=conf.api_key,
                deployment=conf.deployment,
            )
        elif conf.type == "siliconflow":
            # SiliconFlow (硅基流动) exposes an OpenAI-compatible embeddings API.
            # `endpoint` carries the base_url, e.g. https://api.siliconflow.cn/v1
            embeddings = OpenAIEmbeddings(
                model=conf.model,
                api_key=conf.api_key,
                base_url=conf.endpoint,
            )
        elif conf.type == "azure-openai":
            embeddings = AzureOpenAIEmbeddings(
                model=conf.model, 
                azure_endpoint=conf.endpoint,
                azure_deployment=conf.deployment,
                dimensions=1536,
                api_key=conf.api_key,
                api_version=conf.api_version
                
            )
        else:
            logger.warning(f"Embedder type '{conf.type}' not supported.")
            embeddings = None

        return embeddings