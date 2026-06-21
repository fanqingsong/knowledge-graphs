from __future__ import annotations

from typing import Union

from graphrag.core.config import EmbedderConf
from .logger import get_logger


logger = get_logger(__name__)


def get_embeddings(conf: EmbedderConf) -> Union[OpenAIEmbeddings, AzureOpenAIEmbeddings, None]:
    """
    Provider SDKs (`langchain_openai`) are imported lazily inside the branch
    that uses them, so importing this function does not require the provider
    package unless an OpenAI/Azure embedder is actually requested.
    """
    if conf.type == "openai":
        from langchain_openai.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model=conf.model,
            api_key=conf.api_key,
            deployment=conf.deployment,
        )
    elif conf.type == "siliconflow":
        # SiliconFlow (硅基流动) exposes an OpenAI-compatible embeddings API.
        # `endpoint` carries the base_url, e.g. https://api.siliconflow.cn/v1
        from langchain_openai.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model=conf.model,
            api_key=conf.api_key,
            base_url=conf.endpoint,
        )
    elif conf.type == "azure-openai":
        from langchain_openai.embeddings import AzureOpenAIEmbeddings
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
