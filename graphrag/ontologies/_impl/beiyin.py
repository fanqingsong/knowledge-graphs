"""
beiyin.py — Ontology for Zhu Ziqing's prose 《背影》("The Sight of Father's Back")
====================================================================================

A hand-crafted `Ontology` for modern Chinese reminiscence / lyrical prose
centered on 《背影》. It targets the genre's recurring building blocks:
people (narrator, father, grandmother), places (Xuzhou, Nanjing, Pukou
railway station, Beijing), symbolically charged objects (oranges, the
father's clothing), the events that drive the narrative, the work itself,
and the emotions the narrator attaches to them.

Because 《背影》is short (~1500 chars) and yields only a few chunks,
sampling-based auto-discovery would under-cover its content. Using this
fixed ontology instead gives a clean, reproducible graph schema and keeps
Cypher queries against the result predictable.

Wire it in via the knowledge-graph config::

    from graphrag.ontologies.beiyin import beiyin_ontology
    from graphrag.core.config import KnowledgeGraphConfig

    kg_conf = KnowledgeGraphConfig(..., ontology=beiyin_ontology)
"""

from graphrag.core.models import Ontology


beiyin_ontology = Ontology(
    allowed_labels=[
        "Person",      # 人物：叙述者、父亲、祖母等个体
        "Location",    # 地点：徐州、南京、浦口火车站、北京
        "Object",      # 物件：橘子、黑布小帽、深青布棉袍等承载叙事/情感的物品
        "Event",       # 事件：祖母去世、车站送别、买橘子
        "Time",        # 时间节点：那年冬天、近几年来（时点/时段，非数值日期）
        "Work",        # 作品：《背影》及其文学属性（标题、作者、体裁）
        "Emotion",     # 情感：悔恨、感激、伤怀，用于刻画主题
    ],
    labels_descriptions={
        "Person":   "文本中出现的人物，包括叙述者、亲属及其他个体。",
        "Location": "故事发生的地理位置或场所，如城市、车站、家中。",
        "Object":   "具有叙事意义的具体物件，尤其是承载情感象征的物品（如橘子、父亲的衣着）。",
        "Event":    "推动叙事或承载情感的事件、行为、场景。",
        "Time":     "时间标记，指事件发生的时点或时段（非具体日期数值）。",
        "Work":     "散文作品本身及其作为文学文本的属性（标题、作者、体裁）。",
        "Emotion":  "人物内心或叙述者抒发的情感状态，用于刻画主题。",
    },
    allowed_relations=[
        "FATHER_OF",       # 父子关系（父亲 → 朱自清）
        "RELATIVE_OF",     # 泛亲属关系（祖母 → 父亲/朱自清）
        "TRAVELS_TO",      # 前往某地（朱自清 → 浦口）
        "DEPARTS_FROM",    # 离开某地
        "OCCURS_AT",       # 事件发生于某地（送别 → 浦口火车站）
        "OCCURS_DURING",   # 事件发生于某时点（买橘子 → 那年冬天）
        "PARTICIPATES_IN", # 人物参与某事件
        "PERFORMS",        # 人物执行某行为/购买某物（父亲 → 买橘子）
        "SYMBOLIZES",      # 物件象征某情感（橘子/背影 → 感激）
        "AUTHORED_BY",     # 作品的作者（Work → Person）
        "SET_IN",          # 作品设定的地点/背景
        "FEELS",           # 人物怀有某情感
    ],
)
