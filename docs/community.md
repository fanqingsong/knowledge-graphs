# 知识图谱中的「社区（Community）」

> 本文档解释本项目（GraphRAG 风格的知识图谱问答系统）中"社区"的概念、它在流程中的作用，以及对应的代码实现位置。
> 🔬 想深入了解背后的数学与算法原理（模块度、Louvain/Leiden 差异、社区摘要为何能解决全局问答）？见 **[community-theory.md](community-theory.md)**。

## 1. 什么是社区

**社区（community）** 是知识图谱中一组联系紧密的节点（及其边）所形成的"簇"：

- **社区内部**：节点之间连接很多、关系很密。
- **社区之间**：连接相对稀少。

可以把它类比为社交网络里的"朋友圈"或"兴趣小组"——组内成员彼此很熟，但跟别的组来往较少。本质上，社区是图的一种**聚类 / 分组结构**，把零散的实体按主题组织成有意义的簇。

## 2. 为什么需要社区

传统 RAG 基于向量近邻召回文本片段，擅长回答**具体、局部**的问题，但很难回答需要"通览全局"的问题，例如：

- 这批资料的主要主题是什么？
- 数据里有哪些关键实体、它们之间的关系脉络如何？

社区的价值正在于此——它把图谱预先组织成主题簇，并生成**社区摘要（Community Report）**，让检索既能"见树木"（具体实体/关系），也能"见森林"（社区级摘要），从而支撑宏观、概括性的问答。这正是微软 **GraphRAG** 方法的核心思路。

## 3. 常见的社区发现算法

本项目使用 **模块度（modularity）** 作为衡量分组质量的指标，并提供两种检测算法：

| 算法 | 特点 | 本项目节点属性 |
| --- | --- | --- |
| **Leiden** | Louvain 的改进版，保证社区连通性、质量更高，本项目主推 | `community_leiden` |
| **Louvain** | 基于模块度的贪心算法，速度快、最常用 | `community_louvain` |

模块度越高，说明社区划分越"内紧外松"、质量越好。

其他常见（本项目未实现）的算法：Girvan-Newman（逐步移除"桥"边）、Label Propagation（标签传播，适合大规模图）、层次化聚类（大社区套小社区，多层结构）。

### 3.1 社区计算的图范围（`community_source` 开关）

社区检测**以图为对象**——但"用整张图还是只用实体子图"会影响社区质量。本项目通过 `community_source` 配置切换（`config.env` 的 `COMMUNITY_SOURCE`）：

| 取值 | 检测对象 | Chunk 如何获得社区号 | 适用 |
| --- | --- | --- | --- |
| `full`（默认） | **整张图**（实体 + Chunk + Document + 所有边） | 算法直接给每个 Chunk 分配（多按 `NEXT` 链聚团） | 行为与历史一致 |
| `entities` | **仅实体子图**（`__Entity__` 节点 + 实体间边） | 检测后**传播**：每个 Chunk 取其 `MENTIONS` 实体的多数社区 | 主题更内聚，**大语料推荐** |

> 为什么 `full` 模式社区偏散？因为 Chunk 数量通常远多于实体，且 Chunk 之间有大量 `NEXT` 链，会稀释实体间的真实主题结构——大量 Chunk 只是按"文档内顺序"聚团。`entities` 模式只在实体上聚类，再把社区号沿 `MENTIONS` 传给 Chunk，社区更贴合"主题簇"。注意：不提及任何实体的 Chunk 在 `entities` 模式下不会被分配社区（社区号为 `null`），因而不参与社区检索——这是预期行为（它们没有主题信息）。

实现：`analysis.py` 的 `get_entity_digraph()`（实体子图）、`_propagate_communities_to_chunks()`（传播）；传播 Cypher 见 `cypher.py` 的 `propagate_communities_to_chunks`。

## 4. 本项目中的社区流程

整体流程分为四个阶段：

```
文档 → 实体/关系抽取 → 构建知识图谱 → 社区发现 → 社区摘要 → 检索问答
```

1. **建图**：从文档中抽取实体与关系，写入 Neo4j 知识图谱。
2. **社区发现**：对图跑 Leiden / Louvain，给每个节点打上社区编号。
3. **社区摘要**：对每个社区，用 LLM 把该社区包含的文本块（chunks）总结成一段摘要，并生成向量嵌入以便检索。
4. **问答**：把社区摘要作为宏观上下文喂给 LLM，回答全局性问题。

## 5. 对应的代码实现

> `KnowledgeGraph` 是门面类，方法按职责拆分到 mixin；下表标注实际实现的文件。

| 关注点 | 位置 |
| --- | --- |
| 门面类（构造、连接、`vector_store`/`cr_store`） | `graphrag/graph/knowledge_graph.py` |
| Leiden / Louvain 检测 + 模块度计算 | `graphrag/graph/graph_data_structure.py`<br>`detect_leiden_communities()`、`detect_louvain_communities()`、`update_modularity()` |
| 社区与摘要的数据模型 | `graphrag/graph/graph_model.py`<br>`Community`、`CommunityReport` |
| 摘要生成（LLM 总结 + 嵌入） | `graphrag/agents/community_summarizer.py`<br>`CommunitiesSummarizer.get_reports()` / `get_community_report()` |
| 社区读取/写入（`get_communities`、`store_community_reports`、`update_centralities_and_communities`、`get_digraph`） | `graphrag/graph/analysis.py`（`AnalysisMixin`） |
| 统计属性（`number_of_*_communities`、`*_modularity` 等） | `graphrag/graph/metadata.py`（`MetadataMixin`） |
| 社区相关 Cypher 与事务回调 | `graphrag/graph/cypher.py`（`fetch_communities`、`fetch_chunk`） |

### 关键数据结构

**`Community`**（`graphrag/graph/graph_model.py`）描述图谱中的一个社区：

- `community_type`：社区类型，如 `leiden` 或 `louvain`。
- `community_id`：该社区在节点属性中的编号。
- `community_size`：社区内节点数量。
- `entity_ids` / `entity_names`：社区相关的实体。
- `relationship_ids` / `relationship_types`：社区相关的关系。
- `chunks`：落到该社区内的文本块，摘要即基于它们生成。

**`CommunityReport`**（同文件）描述由社区生成的摘要报告：

- `community_type` / `community_id`：所属社区。
- `summary`：LLM 生成的摘要文本。
- `rank`：排序权重，越高越靠前。
- `summary_embeddings`：摘要的向量嵌入，用于检索。

### 摘要生成逻辑（`community_summarizer.py`）

`get_community_report()` 的处理步骤：

1. 若该社区没有任何 chunk，记录警告并跳过。
2. 把社区内所有 chunk 的文本拼接为 `context`。
3. 用摘要 prompt 调用 LLM 生成 `summary`。
4. 对 `summary` 生成向量嵌入 `summary_embeddings`。
5. 封装为 `CommunityReport` 返回。

## 6. 参考

- GraphRAG（Microsoft）：基于社区摘要回答全局性问题的检索增强方法。
- Traag, V. A. et al. *From Louvain to Leiden: guaranteeing well-connected communities.*（Leiden 算法）
- Blondel, V. et al. *Fast unfolding of communities in large networks.*（Louvain 算法）
