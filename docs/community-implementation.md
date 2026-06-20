# 社区链路：从计算到存储到社区报告（代码实现详解）

> 本文从**代码层面**逐行追踪社区在本项目中如何被「计算 → 存储 → 打到 Chunk 上 → 关联到社区报告」。概念入门见 [`community.md`](community.md)、数学与算法原理见 [`community-theory.md`](community-theory.md)。
>
> 本文回答四个具体问题：
> 1. 社区是怎么算出来的？
> 2. 算完存到哪里？
> 3. Chunk 的社区标签是怎么打上去的？
> 4. 社区和社区报告靠什么关联？

---

## 0. 一条链路总览

```
Neo4j 图 ─快照─▶ networkx ─Louvain/Leiden─▶ 节点打 community_xx 属性
                                                       │ SET 写回
Neo4j 节点 (community_leiden / community_louvain) ◀─────┘
        │ GROUP BY community_id
        ▼
Community (含 chunks) ─LLM 汇总 chunk 文本─▶ CommunityReport ─embedding─▶ (:CommunityReport) + 向量索引
        ◀──── 关联靠 (community_type, community_id) 元数据，而非图边 ────▶
```

核心结论先记住：**社区不是独立节点，而是贴在节点上的一个整数属性；社区报告是独立节点 + 独立向量索引；两者靠 `community_id` 这个值关联，没有显式的图关系边把社区和报告连起来。**

---

## 1. 社区如何计算

入口：`AnalysisMixin.update_centralities_and_communities()`（`graphrag/graph/analysis.py:182`）。

### 1.1 第一步：把图快照进 networkx

根据 `community_source` 配置决定快照哪张图：

| 取值 | 方法 | 图内容 |
| --- | --- | --- |
| `entities`（默认推荐） | `get_entity_digraph()` | 仅 `__Entity__` 节点 + 实体间边（去掉 Chunk 及 NEXT/PART_OF/MENTIONS 稠密链） |
| 其它（`full`） | `get_digraph()` | 整图（Entity + Chunk + 所有边） |

> 为什么有这个开关？见 [`community.md`](community.md) §3.1。简言之：full 模式下 Chunk 数量远多于实体且彼此有大量 `NEXT` 链，会稀释实体间的真实主题结构；entities 模式只在实体上聚类，社区更贴合"主题簇"。

### 1.2 第二步：跑两种社区检测算法

纯算法都在 `graphrag/graph/graph_data_structure.py`。两种都跑、都存，可对比。

**Louvain**（`detect_louvain_communities`）：
- `G_undirected = G.to_undirected()`：Louvain 只认无向图。
- `partition = community.best_partition(G_undirected)`：返回 `{node: group_id}` 映射。
- `nx.set_node_attributes(G, partition, "community_louvain")`：给每个节点打 `community_louvain` 属性。
- `modularity = community.modularity(partition, G_undirected)`：算全局质量分。

**Leiden**（`detect_leiden_communities`）：
- `leidenalg` 只认 igraph，所以先建 `mapping = {node: 整数id}`。
- 在 igraph 有向图上 `find_partition(ig_G, ModularityVertexPartition)`。
- 用 `reverse_mapping` 把整数社区号翻回节点名，写到 `G.nodes[...]["community_leiden"]`。
- `modularity = partition.modularity`：Leiden 直接给出质量分。

> 两者的判定本质都是**最大化模块度（modularity）**——找一个分组让组内连边稠密、组间稀疏。Louvain 用贪心移动 + 层次凝聚近似求解；Leiden 在此基础上加 refinement 步，保证每个社区内部连通、更稳定。数学细节见 [`community-theory.md`](community-theory.md) §1–2。

### 1.3 第三步：顺手算中心性

`compute_centralities(G)`（同文件）给每个节点打 `pagerank` / `betweenness` / `closeness` 三个属性。与社区正交——社区回答"属于哪一团"，中心性回答"在团里/团间多关键"。

### 1.4 第四步（仅 entities 模式）：社区号下传到 Chunk

entities 模式下检测只跑在实体上，**只有实体带了社区号，Chunk 没有**。但 QA 是按社区取 Chunk 的，所以 `_propagate_communities_to_chunks()`（`analysis.py:244`）把社区号沿 `:MENTIONS` 边投票下传。详见 §3。

---

## 2. 社区如何存储

社区**不是独立节点**，而是作为**节点属性**贴在 Neo4j 的实体/Chunk 节点上。

### 2.1 写回节点属性

`update_properties()`（`analysis.py:124`）遍历每个 networkx 节点，用 `build_update_query()`（`graph_data_structure.py:236`）拼出参数化 Cypher：

```cypher
MATCH (n) WHERE elementId(n) = $node_id
SET n.community_leiden = $community_leiden,
    n.community_louvain = $community_louvain,
    n.pagerank = $pagerank,
    n.betweenness = $betweenness,
    n.closeness = $closeness
```

- 没分配到的社区号默认 `-1`，中心性默认 `0.0`。
- 每个字段按需加入 SET 子句（不每次都全写）。
- 值通过 `$placeholder` 传参，防 Cypher 注入。

### 2.2 存全局质量分

整张图的 modularity 单独存到 `(:GraphMetric)` 节点（`update_modularity`，`graph_data_structure.py:192`）：

```cypher
MERGE (m:GraphMetric {name: 'leiden_modularity'}) SET m.value = $modularity
```

> 注意：`GraphMetric` 节点在后续 `get_communities` 里会"伪装成社区"，需要专门跳过（见 §2.3）。

### 2.3 读取社区

`get_communities(comm_type)`（`analysis.py:269`）按 `community_<type>` 属性 GROUP BY，把同一社区里的实体、关系、以及提到这些实体的 Chunk 聚成一个 `Community` 对象返回。核心 Cypher（`cypher.py:154` 的 `fetch_communities`）：

```cypher
OPTIONAL MATCH (chunk:Chunk) WHERE chunk.community_leiden = n.community_leiden
```

两点注意：
- **直接靠 Chunk 上的属性值匹配**，不绕实体边。
- 跳过 `GraphMetric` 行——它们在原始结果里伪装成社区（`analysis.py:302`）。

---

## 3. Chunk 的社区标签是怎么打上去的

这是整条链路里最微妙的一环。"打标签"就是给 Chunk 节点写一个 `community_leiden` / `community_louvain` 整数属性。具体怎么打，分两条路径：

### 3.1 路径 A：`community_source="entities"`（默认）—— 投票下传

此路径下 Chunk **不参与**社区检测，标签是事后从实体"投"下来的。一条 Cypher 搞定（`cypher.py:201` 的 `propagate_communities_to_chunks`）：

```cypher
MATCH (c:Chunk)-[:MENTIONS]->(e:__Entity__)
WHERE e.community_leiden IS NOT NULL
WITH c, e.community_leiden AS cid, count(*) AS freq
ORDER BY freq DESC, cid
WITH c, collect(cid)[0] AS top_cid
SET c.community_leiden = top_cid
```

逐步拆开：

1. **`(c:Chunk)-[:MENTIONS]->(e:__Entity__)`**：抓出每条"Chunk 提到实体"的边，把 Chunk 跟带社区号的实体挂上。
2. **`e.community_leiden IS NOT NULL`**：只认本身已被打了社区号的实体（entities 模式下检测跑在实体上，实体都有号）。
3. **`WITH c, ..., count(*) AS freq`**：对每个 Chunk 按社区号 group，统计每个社区号被提到几次。
4. **`ORDER BY freq DESC, cid`**：出现次数多的社区号排前；次数相同（平票）取社区号最小的。
5. **`collect(cid)[0]`**：每个 Chunk 取排第一的社区号 = **多数票胜出的社区**。
6. **`SET c.community_leiden = top_cid`**：把号写到 Chunk 节点。标签完成。

**边界**：一个 Chunk 如果一个实体都没提到（`MENTIONS` 不命中），就**拿不到标签**，属性保持 `null`，不参与社区检索——这是预期行为（它没有主题信息）。

此逻辑在 `_propagate_communities_to_chunks`（`analysis.py:244`）里对 `leiden` 和 `louvain` 各跑一次，两个标签分别打上。

### 3.2 路径 B：`community_source != "entities"`（full 模式）—— 检测时直接算上

此路径下 Chunk 节点**本来就是 networkx 图里的节点**（`get_digraph` 把 `:Chunk` 节点和 NEXT/PART_OF 边都拉进来）。所以：

1. Louvain/Leiden 在整图上跑，Chunk 节点在算法内部**直接被分到一个社区**，拿到 `community_leiden` 属性。
2. 然后 `update_properties` + `build_update_query` 对**每个节点（含 Chunk）**执行 §2.1 的 SET，把社区号写回 Neo4j。

即：full 模式**根本不需要投票下传那步**——`update_centralities_and_communities` 末尾的 `if source == "entities"` 判断为假，直接跳过 `_propagate_communities_to_chunks`。

### 3.3 两条路径对照

| 模式 | 标签来源 | 关键依赖 |
| --- | --- | --- |
| `entities`（默认） | 沿 `:MENTIONS` 边，从"被提到的实体的社区"做**多数票**下传 | `propagate_communities_to_chunks` 的一条 `SET` Cypher |
| 其它（full） | Chunk 作为图节点**直接被算法分到社区**，随 `update_properties` 写回 | 算法本身 + `build_update_query` 的 SET |

两种方式**最终都落成 Chunk 节点上同一个 `community_leiden`/`community_louvain` 整数属性**。下游 QA 检索时读这个值，不关心它是哪条路径打上去的。

### 3.4 怎么查某个 Chunk 属于哪个社区

直接读属性，一条查询即可：

```cypher
MATCH (c:Chunk) WHERE elementId(c) = $elementId
RETURN c.community_leiden, c.community_louvain
```

---

## 4. 社区与社区报告如何关联

关联是通过 **`(community_type, community_id, community_size)` 这组元数据**挂的，**不是靠图关系连边**。

### 4.1 生成报告

`CommunitiesSummarizer`（`graphrag/agents/community_summarizer.py`）：

1. `get_reports()`（`:29`）拿到 `get_communities()` 返回的 `Community` 列表。
2. 对每个社区调 `get_community_report()`（`:43`）：
   - **LLM 汇总的对象 = 该社区名下的所有 Chunk 文本**，不是实体/关系：

     ```python
     chunks_content = "\n\n".join(
         chunk.text.replace("\n\n", "\n") for chunk in community.chunks
     )
     summary = self.llm.invoke(
         input=self.summarize_community_prompt.format(context=chunks_content)
     ).content
     ```

   - 这些 `chunks` 是 `get_communities()` 沿 `:MENTIONS` 抓回来的（`analysis.py:316-322`）。
   - 边界：`if not community.chunks`（`:48`）——社区没抓到任何 Chunk 就跳过，只 warning。这也是 entities 模式必须做 `_propagate_communities_to_chunks` 的原因：保证每个社区都能回捞到 Chunk。
3. 对 summary 做 embedding（`embed_documents`）。
4. 打包成 `CommunityReport`（带 `community_type / community_id / community_size`）。

### 4.2 存储报告

`store_community_reports()`（`analysis.py:328`）：

- 每个 report 调 `self.cr_store.add_embeddings(...)`，变成一个 **`(:CommunityReport)` 节点**，属性含 `summary`、`summary_embeddings`、以及上面那三个社区元数据。
- 全部写完后 `create_new_index()` 重建名为 `reports` 的**向量索引**。

### 4.3 QA 如何检索报告

**不是去社区节点上找报告**，而是对 query embedding 在 `reports` 向量索引里做相似度搜索，返回的 `CommunityReport` 自带 `community_id` 等元数据，QA agent 再用这些元数据去过滤/定位到原图里的实体和 Chunk。

> 两个索引服务两种检索粒度：原始 `Chunk` 向量库服务"局部细节"，`CommunityReport` 向量库服务"全局主题"。原理见 [`community-theory.md`](community-theory.md) §3.3。

---

## 5. 涉及的代码位置速查

| 关注点 | 位置 |
| --- | --- |
| 分析编排入口 | `graphrag/graph/analysis.py`：`AnalysisMixin.update_centralities_and_communities()` |
| 整图 / 实体子图快照 | `analysis.py`：`get_digraph()` / `get_entity_digraph()` |
| 节点属性写回 | `analysis.py`：`update_properties()` + `graph_data_structure.py`：`build_update_query()` |
| Louvain / Leiden 算法 + 模块度 | `graphrag/graph/graph_data_structure.py`：`detect_louvain_communities()` / `detect_leiden_communities()` / `update_modularity()` |
| Chunk 社区号下传（entities 模式） | `analysis.py`：`_propagate_communities_to_chunks()` |
| 读取社区 | `analysis.py`：`get_communities()` |
| 报告持久化 | `analysis.py`：`store_community_reports()` |
| 数据模型 | `graphrag/graph/graph_model.py`：`Community` / `CommunityReport` |
| 报告生成（LLM + embedding） | `graphrag/agents/community_summarizer.py`：`CommunitiesSummarizer` |
| Cypher 与事务回调 | `graphrag/graph/cypher.py`：`fetch_communities` / `fetch_chunk` / `propagate_communities_to_chunks` |

---

## 6. 一句话总结

社区 = 算法（Louvain/Leiden）在图上按模块度最大化分出来、贴在每个节点上的一个整数属性；Chunk 的这个属性要么是 full 模式下算法直接给的，要么是 entities 模式下从它提到的实体社区"多数票"投出来的；社区报告是独立的节点 + 独立向量索引，与社区靠 `community_id` 元数据关联而非图边；报告内容由 LLM 汇总该社区 Chunk 原文生成。
