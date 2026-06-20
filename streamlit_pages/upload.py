import os
import shutil
import streamlit as st

from graphrag.config import Configuration
from graphrag.graph.knowledge_graph import KnowledgeGraph
from graphrag.ingestion.local_ingestor import LocalIngestor
from graphrag.ingestion.chunker import Chunker
from graphrag.ingestion.cleaner import Cleaner
from graphrag.ingestion.embedder import ChunkEmbedder
from graphrag.ingestion.graph_miner import GraphMiner
from graphrag.agents.community_summarizer import CommunitiesSummarizer

from streamlit_pages.utils import get_configuration_from_env

st.set_page_config(
    page_title="Upload",
    page_icon="🗳️",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    ## 🗳️ Ingestion of Files in the Graph 🕸️

    Use the box below to upload Files in `.pdf`, `.docx`, `.txt` or `.html` format.
    They will be uploaded inside this App's root directory Source Folder and will then be available
    for the ingestion process into your Knowledge Graph.

    Each uploaded file will be passed into a pipeline that will;
    * load it into a json format;
    * cleaning its text;
    * divide its text into smaller pieces, called chunks;
    * embed each chunk into its vector representation;
    * use a LLM model to extract a graph of concepts from each chunk;
    * upload the obtained vectors and entities into the Knowledge Graph;
    * update the centralities measures and the division of the Graph into communities.
    """
)

# 源文件存放目录：上传的文件会保存到这里的 source_docs 子目录下
SOURCE_FOLDER = f"{os.getcwd()}/source_docs"
# 应用配置文件路径：优先从这里读取 Neo4j、模型、切分等参数
CONF_PATH = f"{os.getcwd()}/configuration.json"

# env 标记是否改用环境变量提供配置（这里固定 False，保留以便后续扩展）
env = False
conf = None
uploaded_files = None
# 在 session_state 中初始化状态标记，用于控制按钮点击与流程进度
st.session_state['ingest_clicked'] = False     # 是否点击了"开始摄取"按钮
st.session_state["cleanup_clicked"] = False    # 是否点击了"清理文件夹"按钮

# 优先从配置文件读取；若读取失败（文件缺失或格式错误）则回退到环境变量配置
try:
    conf = Configuration.from_file(CONF_PATH)
except Exception as e:
    conf = get_configuration_from_env()


# 只要有可用配置（文件或环境变量），就显示文件上传组件
if conf or env:
    uploaded_files = st.file_uploader(
        label="Upload Files to ingest",
        accept_multiple_files=True
    )

# 当用户确实上传了文件时，才进入下面的保存与摄取流程
if len(uploaded_files) > 0:

    # 确保源文件目录存在，避免写入时报错
    os.makedirs(SOURCE_FOLDER, exist_ok=True)

    # 将用户上传的每个文件原样落盘到 source_docs 目录，供后续摄取流水线读取
    for file in uploaded_files:
        file_name = file.name

        file_path = SOURCE_FOLDER + f"/{file_name}"

        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

    # 显示"开始摄取"按钮，点击结果记录到 session_state
    st.session_state["ingest_clicked"] = st.button(
        label="Ingest into Knowledge Graph",
        icon="🗳️"
    )

    # 用户点击摄取按钮后，启动整条摄取流水线
    if st.session_state["ingest_clicked"]:
        with st.status(
            f"Ingesting {len(uploaded_files)} Documents...",
            expanded=True
            ) as status:

            st.write("Setting up the Ingestion Pipeline ..")

            # === 第 0 步：实例化整条流水线的各个组件 ===
            ingestor = LocalIngestor(source=conf.source_conf)   # 文件加载器：把原始文件读成统一文档结构
            cleaner = Cleaner()                                  # 文本清洗器：去除噪声、规范化文本
            chunker = Chunker(conf=conf.chunker_conf)           # 分块器：按配置切分长文本为 chunk
            embedder = ChunkEmbedder(conf=conf.embedder_conf)   # 向量化器：把 chunk 转成嵌入向量
            graph_miner = GraphMiner(                           # 图谱抽取器：用 LLM 从文本中抽取实体与关系
                conf=conf.re_model_conf,
                ontology=conf.database.ontology
            )
            knowledge_graph = KnowledgeGraph(                   # 知识图谱连接：封装 Neo4j 读写与向量索引
                conf=conf.database,
                embeddings_model=embedder.embeddings,
            )

            # 先校验 Neo4j 连接凭证，认证失败则中断流程并提示用户检查配置
            if not knowledge_graph._driver.verify_authentication():
                st.error("Check your Neo4j Configuration!")

            else:
                # === 第 1 步：加载 ===
                # 从 source_docs 读取所有文件，转换为统一的文档对象列表
                st.write("Loading..")
                docs = ingestor.batch_ingest()

                # === 第 2 步：清洗 ===
                # 清理文本内容（去空白、去乱码、统一格式等）
                st.write("Cleaning..")
                docs = cleaner.clean_documents(docs)

                # === 第 3 步：分块 ===
                # 将长文档切成较短的 chunk，便于向量化和逐块抽取
                st.write("Chunking..")
                docs = chunker.chunk_documents(docs)

                # === 第 4 步：向量化 ===
                # 为每个 chunk 计算嵌入向量，用于后续相似度检索
                st.write("Embedding..")
                docs = embedder.embed_documents_chunks(docs)

                # === 第 5 步：图谱抽取 ===
                # 用 LLM 从每个 chunk 抽取实体和关系，构建局部知识子图
                st.write("Extracting a Knowledge Graph from each file..")
                docs = graph_miner.mine_graph_from_docs(docs=docs)

                # === 第 6 步：写入知识图谱 ===
                # 把文档、向量、实体、关系全部写入 Neo4j
                st.write("Uploading Data to Knowledge Graph..")
                knowledge_graph.add_documents(docs)

                # === 第 7 步：更新图结构指标 ===
                # 计算节点中心性指标，并执行社区发现（Leiden / Louvain 两种算法）
                st.write("Updating Communities and computing Centralities in the Graph..")
                knowledge_graph.update_centralities_and_communities()

                # === 第 8 步：社区摘要 ===
                st.write("Summarizing Communities into Reports..")
                # 社区/子图问答模式会从 CommunityReport 向量库检索内容，
                # 因此必须为每个检测到的社区生成一份摘要报告（两种算法各生成一份）。
                summarizer = CommunitiesSummarizer(
                    llm_conf=conf.qa_model,
                    embeddings_conf=conf.embedder_conf
                )
                # 分别对 leiden 和 louvain 两种社区划分结果生成报告并入库
                for comm_type in ["leiden", "louvain"]:
                    communities = knowledge_graph.get_communities(comm_type=comm_type)
                    reports = summarizer.get_reports(communities)
                    knowledge_graph.store_community_reports(reports)
                    st.write(f"  · {comm_type}: {len(reports)} community reports stored")

                # 全部步骤完成，更新状态条为完成态
                status.update(
                    label="Done with the Ingestion",
                    state="complete",
                    expanded=False
                )

        # 流水线成功完成后，显示成功提示并提供"清理文件夹"按钮
        if status._current_state == "complete":
            st.success(body=f"Done with the Ingestion of {len(docs)} Files")

            st.session_state["cleanup_clicked"] = st.button(
                label="Cleanup Folder",
                help=f"Clicking this will delete files in folder {SOURCE_FOLDER}",
                icon="🗑️"
            )

        # 用户点击清理按钮后，删除 source_docs 下的所有文件（保留目录本身）
        if st.session_state["cleanup_clicked"]:
            for filename in os.listdir(SOURCE_FOLDER):
                file_path = os.path.join(SOURCE_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            st.info("Cleanup Completed!")
            