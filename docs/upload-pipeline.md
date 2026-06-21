# Upload（文档摄入）流程说明

> 本文档说明在 Streamlit 的 **Upload** 页面上传文件后，文档如何一步步变成知识图谱与社区摘要报告。入口页面：`streamlit_pages/upload.py`。
> *(English version: [`upload-pipeline.en.md`](upload-pipeline.en.md))*

## 1. 总览

Upload 页面把文件接收与"摄入流水线"分开：

- **UI 层**：接收上传文件并落盘到本地 `source_docs/`。
- **流水线**：从 `source_docs/` 读盘，对 `List[ProcessedDocument]` 做 8 个阶段的处理，逐级富集数据（先有文本 → 加 chunk → 加 embedding → 加 graph → 入库 → 算社区 → 生成摘要）。

每个阶段都接收并返回 `List[ProcessedDocument]`：

```
source_docs/*.pdf,docx,txt,html
   │  ① load    (MIME→loader)         → doc.source
   │  ② clean   (正则归一)             → source
   │  ③ chunk   (RecursiveSplitter)    → doc.chunks[].text
   │  ④ embed   (向量模型)             → chunk.embedding
   │  ⑤ mine    (LLM 抽实体/关系)      → chunk.nodes / chunk.relationships
   │  ⑥ store   → Neo4j (Chunk/Entity/Relationship + 向量)
   │  ⑦ graph   → DiGraph → Leiden/Louvain/中心性 → 写回节点属性
   └─ ⑧ report  → 每社区 LLM 摘要 + 向量 → CommunityReport 向量库
```

## 2. UI 层：文件接收（`streamlit_pages/upload.py`）

1. **加载配置**：先尝试 `Configuration.from_file(configuration.json)`，失败则从环境变量读取（`get_configuration_from_env()`）。
2. **上传文件**：`st.file_uploader` 接收 `.pdf / .docx / .txt / .html`，逐个**写入本地 `source_docs/` 目录**。
   > ⚠ 流水线是从 `source_docs/` **读盘**运行的，不是直接处理内存里的上传对象。
3. 点击 **"Ingest into Knowledge Graph"** 触发流水线；期间用 `st.status` 展示各阶段进度。

## 3. 流水线八阶段

### ① Loading（`LocalIngestor` → `Ingestor`）
`graphrag/ingestion/ingestor.py`、`graphrag/ingestion/local_ingestor.py`

- `list_files()` 用 `os.walk` 扫描 `source_docs/`，拿到全部文件路径。
- `file_preparation()` 取父目录名作为 `folder` 元数据。
- `load_file()` 用 **python-magic 探测 MIME 类型**，按下表选 loader；不支持的类型跳过并告警：

  | MIME | Loader |
  | --- | --- |
  | `application/pdf` | `PDFPlumberLoader` |
  | `text/plain` | `TextLoader` |
  | `text/html` | `BSHTMLLoader`（UTF-8） |
  | `.docx` | `Docx2txtLoader` |

- 各页内容用 `\n\n` 拼接，封装为 `ProcessedDocument(filename, source, metadata)`。

### ② Cleaning（`Cleaner`）
`graphrag/ingestion/cleaner.py` —— `_clean_text()` 是一组正则，规范化文本：

- 去星号、修正被误识别为 `l` 的项目符号；
- 统一破折号/连字符、把各种花式撇号归一为 `'`；
- 去掉 BOM (`﻿`) 与控制字符；
- 在小写字母/数字后接大写处补空格（拆开粘连的 camelCase）；
- 折叠多余空行/空格、去掉意大利语页脚 `"Pagina X di Y"`。

目的：提升后续切块与抽取质量。

### ③ Chunking（`Chunker`）
`graphrag/ingestion/chunker.py`

- 用 `RecursiveCharacterTextSplitter`，参数取自 `CHUNKER_*`（默认 `chunk_size=1000`、`chunk_overlap=100`）。
- 给每个 chunk 分配自增 `chunk_id`，封装为 `Chunk(text, chunk_id, chunk_size, chunk_overlap)` 挂到 `doc.chunks`。

### ④ Embedding（`ChunkEmbedder`）
`graphrag/ingestion/embedder.py`

- 对每个 `chunk.text` 调 embedding 模型生成向量，写入 `chunk.embedding`，并记录 `chunk.embeddings_model`。
- 模型由 `EMBEDDINGS_TYPE` 决定（本项目默认硅基流动 `bge-m3`，1024 维）。

### ⑤ Graph Extraction（`GraphMiner` → `GraphExtractor`）
`graphrag/ingestion/graph_miner.py`、`graphrag/agents/graph_extractor.py`

- **逐 chunk** 用 LLM 抽取实体与关系，输出 `_Graph(nodes, relationships)`。
- 经 `map_to_lc_graph()`（`graphrag/graph/graph_converters.py`）转成 langchain 的 `GraphDocument`，把 `nodes` / `relationships` 挂回对应 `chunk`。
- 可选传入 `Ontology`（`conf.database.ontology`）约束允许的标签/关系类型。
- **这是最贵的一步**：每个 chunk 都是一次 LLM 调用，也是大语料下的主要瓶颈。

### ⑥ Uploading to Graph（`KnowledgeGraph.add_documents`）
门面 `graphrag/graph/knowledge_graph.py`；实现见 `graphrag/graph/ingestion.py`（`IngestionMixin`），事务回调见 `graphrag/graph/cypher.py`

- 遍历 docs → `store_chunks_for_doc()`：把 Chunk（带向量）及其抽取出的 nodes/relationships 写入 Neo4j，并建立 `Chunk -[:MENTIONS]-> Entity` 等关系。
- 至此图谱实体、关系、向量都已落库。

### ⑦ Centralities & Communities（`update_centralities_and_communities`）
编排见 `graphrag/graph/analysis.py`（`AnalysisMixin`）；检测算法见 `graphrag/graph/graph_data_structure.py`

- `get_digraph()` 把整张 Neo4j 图读成 `networkx.DiGraph`。
- 跑 **Louvain** 与 **Leiden** 两种社区检测，各自算出模块度（modularity）。
- `compute_centralities()` 算 PageRank / betweenness / closeness。
- `update_properties()` 把 `community_leiden`、`community_louvain`、三种中心性、以及两种模块度写回节点属性。
- 每个子步骤都 try/except 包裹，单步失败只记 warning，不中断整体。
- 关于"社区"本身的含义见 [`community.md`](community.md)。

### ⑧ Community Reports（`CommunitiesSummarizer`）
`graphrag/agents/community_summarizer.py`

对 `leiden` 和 `louvain` **各跑一遍**：

- `get_communities(comm_type)` 从图里取每个社区的实体/关系/所含 chunks。
- `get_reports()` 对每个社区：拼接 chunks 文本为 `context` → LLM 生成摘要 → 对摘要生成向量 → 封装 `CommunityReport`。
- `store_community_reports()` 把摘要文本 + 向量 + 元数据写入 `CommunityReport` 向量库（`cr_store`）。
- 这一步为 **Communities / Subgraph** 两种问答模式准备检索源。

## 4. 结束与清理

- `status.update(state="complete")` 后显示成功，并出现 **"Cleanup Folder"** 按钮。
- 点击会删除 `source_docs/` 下所有文件（**只删源文件，不动数据库里的图**）。

## 5. 注意事项

- **顺序处理、无并发**：流水线逐文档、逐 chunk 串行；第⑤步（LLM 抽取）是主要耗时点。
- **向量索引创建被注释**：`upload.py` 中 `index_exists()/create_index()` 段为 `# TODO`，依赖部署侧已存在的索引（`INDEX_NAME`）。
- **重复摄入会产生重复节点**：当前未做去重/实体消歧；如需清空重来，停服务后 `./bin/stop.sh -v` 删除数据卷再重启。
