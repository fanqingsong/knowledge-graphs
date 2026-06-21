# 部署与使用指南（Docker Compose · 智谱 GLM · 硅基流动）

> 本文档介绍如何用 Docker Compose 一键启动本项目，并使用**智谱 GLM**（LLM）+ **硅基流动**（Embedding）的远程 API 运行知识图谱 / GraphRAG 应用，支持**热加载**开发。

---

## 一、架构概览

```
┌───────────────────────────────────────────────────────────┐
│  docker compose (graphnet)                                │
│                                                           │
│  ┌───────────────┐    ┌─────────────────────────────────┐ │
│  │  kg-app       │    │  kg-neo4j (2025.03.0)           │ │
│  │  Streamlit    │◄──►│  图数据库 + 向量索引            │ │
│  │  :8080        │    │  Browser :7474  Bolt :7687      │ │
│  └──────┬────────┘    └─────────────────────────────────┘ │
│         │                                                 │
└─────────┼─────────────────────────────────────────────────┘
          │ 远程 API（OpenAI 兼容）
          ▼
   ┌──────────────────┐        ┌──────────────────────┐
   │ 智谱 GLM (LLM)   │        │ 硅基流动 (Embedding) │
   │ glm-4.7          │        │ BAAI/bge-m3 (1024维) │
   │ open.bigmodel.cn │        │ api.siliconflow.cn   │
   └──────────────────┘        └──────────────────────┘
```

- **kg-app**：Streamlit 应用容器，挂载项目根目录实现热加载。
- **kg-neo4j**：同时作为图数据库和向量库；实体/关系存为图节点/边，文档 Chunk 带向量。
- **LLM / Embedding 走远程 API**：无需本地 GPU，无需 ollama / torch。

---

## 二、前置准备

1. 安装 Docker 与 Docker Compose v2（`docker compose` 子命令）。
2. 获取两把 API Key：
   - 智谱：https://open.bigmodel.cn
   - 硅基流动：https://cloud.siliconflow.cn

---

## 三、配置 `.env`

`.env` 为本地未纳入版本控制的配置文件（已在 `.gitignore`）。`bin/start.sh` 首次运行时会从 `config_example.env` 自动生成；你也可以手动创建。

关键项（**把占位符换成真实 Key**）：

```env
# Neo4j（与 docker-compose 中 NEO4J_AUTH 一致）
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
INDEX_NAME=vector

# Embedding —— 硅基流动（OpenAI 兼容）
EMBEDDINGS_TYPE=siliconflow
EMBEDDINGS_MODEL_NAME=BAAI/bge-m3          # 1024 维
EMBEDDINGS_API_KEY=sk-xxxxxxxx
EMBEDDINGS_ENDPOINT=https://api.siliconflow.cn/v1

# 关系抽取 LLM —— 智谱 GLM
RE_MODEL_TYPE=zhipu
RE_MODEL_NAME=glm-4.7
RE_API_KEY=xxxxxxxx
RE_MODEL_ENDPOINT=https://open.bigmodel.cn/api/coding/paas/v4/

# 问答 LLM —— 智谱 GLM
QA_MODEL_TYPE=zhipu
QA_MODEL_NAME=glm-4.7
QA_API_KEY=xxxxxxxx
QA_MODEL_ENDPOINT=https://open.bigmodel.cn/api/coding/paas/v4/
```

> 维度说明：`BAAI/bge-m3` 输出 **1024 维**，与原默认 `mxbai-embed-large` 一致；切换其它 embedding 模型时需注意维度，必要时重建 Neo4j 向量索引。

> 社区检测范围（可选）：`COMMUNITY_SOURCE`
> - `full`（默认）：在整张图上跑 Louvain/Leiden（历史行为）。
> - `entities`：只在实体子图上聚类，再把社区号沿 `MENTIONS` 传播给 Chunk——社区更贴合主题，**大语料推荐**。详见 [`docs/community.md`](docs/community.md) §3.1。
> 切换后需**重新摄入或重跑社区检测**才会生效（社区属性写回节点后才有意义）。

---

## 四、启动 / 停止（`bin/` 脚本）

| 命令 | 作用 |
|---|---|
| `./bin/start.sh` | 构建（首次）并后台启动 app + neo4j；首次运行自动从 `config_example.env` 生成 `.env` |
| `./bin/stop.sh` | 停止服务，**保留**数据卷（neo4j_data） |
| `./bin/start.sh -v`（即 `./bin/stop.sh -v`） | 停止并**删除**数据卷（清空图与向量数据） |

启动后访问：

| 服务 | 地址 |
|---|---|
| Streamlit 应用 | http://localhost:8080 |
| Neo4j Browser | http://localhost:7474 （`neo4j` / `password123`） |

---

## 五、热加载（Hot Reload）

`docker-compose.yml` 把项目根目录挂载到容器 `WORKDIR`（`/knowledge-graphs`），Dockerfile 中 Streamlit 启用：

```
--server.fileWatcherType=poll
--server.runOnSave=true
```

因此**编辑 `graphrag/`、`streamlit_pages/`、`app.py` 后无需重启**，浏览器会自动刷新。

> 仅当修改 `.env` 时需要重建 app 容器：
> ```bash
> docker compose up -d --force-recreate app
> ```

---

## 六、模型类型与扩展

`graphrag/config.py` 的 `ModelType` 支持以下类型：

| type | 用途 | 说明 |
|---|---|---|
| `zhipu` | LLM | 智谱 GLM，OpenAI 兼容，走 `ChatOpenAI` + `base_url` |
| `siliconflow` | Embedding | 硅基流动，OpenAI 兼容，走 `OpenAIEmbeddings` + `base_url` |
| `openai` / `azure-openai` | LLM / Embedding | OpenAI / Azure |
| `groq` / `google` | LLM | Groq / Gemini |

工厂代码：`graphrag/utils/llm.py`、`graphrag/utils/embeddings.py`。

新增本地推理类型（HuggingFace / Ollama）的依赖已移除，以避免拉取 torch/CUDA 等数 GB 的包；如需恢复，再加回 `langchain-huggingface` / `langchain-ollama` 及对应分支即可。

---

## 七、镜像加速（华为云源）

Dockerfile 内已将 **apt** 与 **pip** 源指向华为云，显著加速构建（尤其国内网络）：

- apt：`https://repo.huaweicloud.com/debian`
- pip：`https://repo.huaweicloud.com/repository/pypi/simple`

Docker 守护进程的 registry 镜像源在 `/etc/docker/daemon.json` 配置。

---

## 八、聊天页面的「回答方法」

进入 **Chat with Graph**（http://localhost:8080），左侧栏 **Select Answering Method** 提供 5 种从知识图谱取上下文的策略：

| 方法 | 原理 | 前置条件 | 适合场景 |
|---|---|---|---|
| **Similarity Search** | 向量相似检索 Chunk（纯 RAG），不查图关系 | 文档已摄入、向量索引已建 | 单篇文档事实查找；快速验证检索 |
| **Cypher** | LLM 依据图谱 schema 自动生成并执行 Cypher | 已抽取实体/关系 | 关系型问题（"谁连了谁""共有多少 X"） |
| **Communities** | 基于预先生成的社区摘要报告回答 | 已跑社区检测+摘要；选定 leiden/louvain | 全局/主题性问题 |
| **Subgraph** | 取出社区子图结构交给 LLM 推理 | 已跑社区检测；选定 leiden/louvain | 查看社区内部连接拓扑 |
| **Combine** | 同时跑向量检索 + Cypher，综合两路结果作答 | 索引 + 关系抽取均完成 | **默认首选**，答案最全面 |

侧边栏开关：

- **Use neighbouring Chunks**：相似检索时，把命中 Chunk 的前后相邻 Chunk（`NEXT` 关系）一并拼入上下文。
- **Select the reference Community**：Communities / Subgraph 模式必选 `leiden` 或 `louvain`，对应页面顶部 Graph Metrics 中的社区统计。

> ⚠️ 若 Graph Metrics 里 **Leiden / Louvain Communities = 0**，说明尚未摄入文档或社区聚类，Communities 与 Subgraph 暂不可用。可先用 **Similarity Search** 或 **Combine**。

### 8.0 摄入流水线说明

> 📖 关于 Upload 页面提交文档后完整的 8 步摄入流水线（加载 → 清洗 → 切块 → 向量化 → 抽图 → 入库 → 社区/中心性 → 社区摘要），见独立文档：**[docs/upload-pipeline.md](docs/upload-pipeline.md)**。

### 8.1 启用 Communities / Subgraph 模式（社区摘要报告）

> 📖 关于"社区"的概念、Leiden/Louvain 检测原理以及社区摘要如何支撑全局问答，见独立文档：**[docs/community.md](docs/community.md)**。

这两种模式检索的是 **CommunityReport**（社区摘要报告）向量库，而非社区分区本身。摄入流程会自动做**社区检测**（Leiden/Louvain 分区），并由 `IngestionPipeline` 在摄入末尾自动调用 `CommunitiesSummarizer`（智谱生成摘要 + 硅基流动向量化）为 leiden / louvain 两种分区各生成一份社区摘要报告——无需任何额外操作。

> 若旧数据是在启用自动摘要前摄入的、因而缺失 `CommunityReport`，需重新摄入这些文档（目前没有独立的补报告工具）。

验证报告是否生成：

```bash
docker exec kg-neo4j cypher-shell -u neo4j -p password123 \
  "MATCH (c:CommunityReport) RETURN c.community_type AS type, count(*) AS n;"
```

报告数 > 0 后，即可在 Chat 页选择 Communities / Subgraph + 指定 leiden/louvain 使用。

> 日志中出现 `Neo.ClientNotification.Statement.AggregationSkippedNull` 是 Neo4j 聚合跳过 null 的**良性提示**，不影响结果。

---

## 九、常用排障

| 现象 | 处理 |
|---|---|
| 构建慢 / apt 卡 12 kB/s | 确认 Dockerfile 已用华为云源（默认已配置） |
| 拉取 torch/CUDA 等大包 | 不应出现；已移除 `langchain-huggingface`。若仍出现，检查 `requirements.txt` |
| 改了代码不生效 | 确认是热加载范围内的文件；或刷新浏览器；`.env` 改动需 `up -d --force-recreate app` |
| 向量检索报维度错误 | embedding 模型维度需与 Neo4j 索引一致；换模型后清空 `neo4j_data` 重建 |
| 智谱 / 硅基流动 401/403 | 检查 `.env` 中 API Key 与 Endpoint 是否正确，并 `--force-recreate app` |
| 停服务后想彻底重来 | `./bin/stop.sh -v` 后 `./bin/start.sh` |

---

## 十、目录与关键文件

```
.
├── bin/
│   ├── start.sh            # 构建并启动（自动生成 .env）
│   └── stop.sh             # 停止（-v 清数据）
├── docker-compose.yml      # app + neo4j，热加载挂载
├── Dockerfile              # python:3.12-slim，华为云 apt/pip 源，Streamlit poll 热加载
├── config_example.env      # 配置模板（智谱 + 硅基流动）
├── .env                    # 本地实际配置（gitignored，由 start.sh 生成）
├── requirements.txt        # 已移除 torch/ollama 相关本地推理依赖
├── app.py                  # Streamlit 多页入口
├── streamlit_pages/        # 页面：home / upload / chat
└── graphrag/
    ├── config.py           # ModelType: zhipu / siliconflow / openai / ...
    ├── factory/            # llm.py / embeddings.py 模型工厂
    ├── agents/             # graph_qa.py（5 种回答方法）、抽取/摘要 agents
    ├── graph/              # knowledge_graph.py（门面）+ metadata/ingestion/analysis mixin
    │                       #   + cypher.py / graph_data_structure.py / graph_model.py / graph_converters.py
    └── ingestion/          # chunker / embedder / graph_miner / ingestor
```
