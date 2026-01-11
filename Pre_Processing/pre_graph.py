from typing import Dict, List, Tuple, Optional, Any
import os, json, re
import torch

RELATION_ORDER = [
    "has_category", "has_anatomy",
    "located_in", "negated_in", "uncertain_in",
    "normal_of"
]
ALLOWED_RELATIONS = set(RELATION_ORDER)  # RadGraph→KG 映射仅保留这些（可按需扩）

# ----------------- 工具函数 -----------------
def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', (text or "").lower()).strip('_')

def label_to_polarity(label: str) -> str:
    lab = (label or "").lower()
    if "absent" in lab or "negated" in lab:  return "negative"
    if "uncertain" in lab or "maybe" in lab or "suspected" in lab: return "uncertain"
    if "present" in lab:  return "positive"
    return "unknown"

def compose_phrase(head_rid: str, entities: Dict[str, Dict]) -> str:
    """
    把 modify 修饰词和 head 拼成短语，如 'focal airspace disease'
    head_rid: 字符串 ID（RadGraph 的 entity key）
    """
    head = entities[str(head_rid)]
    parts = [(head.get("start_ix", 0), head.get("tokens", ""))]
    for mid, ent in entities.items():
        for rel, tgt in ent.get("relations", []):
            if rel.lower() == "modify" and str(tgt) == str(head_rid):
                parts.append((ent.get("start_ix", 0), ent.get("tokens", "")))
    parts.sort(key=lambda x: x[0])
    return " ".join(tok for _, tok in parts).strip()

def _pretty_name_from_id(nid: str) -> str:
    s = nid.replace("find_", "").replace("anat_", "").replace("_", " ").strip()
    return s.title() if s else nid

def _infer_type_from_id(nid: str) -> str:
    if nid == "root": return "root"
    if nid == "other": return "category"
    if nid.startswith("anat_"): return "anatomy"
    if nid.startswith("find_"): return "finding"
    return "finding"


# =========================================================
# 主类：图谱预处理（加载基础图谱→初始化词表；支持 dynamic_kg / RadGraph）
# =========================================================
class KGPreprocessor:
    """
    统一输出：
      - encode_graph 输入：
          node_ids:  (1, Nmax) int64，节点索引；pad = -1
          rel_mat:   (1, Nmax, Nmax) int64，无边 = -1；有向边：正向=r，反向=r+R_base
          node_mask: (1, Nmax) bool
      - encode_triplets 输入：
          heads, rels, tails: (1, M) int64（无方向；反向在 encode_graph 内处理）

    词表：
      - entity_vocab: str -> int，包含 "[UNK]"=0，基础图谱中的节点将预先收录
      - relation_vocab: str -> int，顺序=RELATION_ORDER ∪ (基础图谱中出现的关系 ∪ 额外关系，按字母序追加)
    """

    # ---------- 构造 & 词表 ----------
    def __init__(self,
                 base_kg: Optional[Dict[str, Any]] = None,
                 entity_vocab: Optional[Dict[str, int]] = None,
                 relation_vocab: Optional[Dict[str, int]] = None,
                 max_nodes: int = 64):
        self.base_kg = base_kg or {}
        self.max_nodes = max_nodes

        # relation_vocab
        if relation_vocab is not None:
            self.relation_vocab = dict(relation_vocab)
        else:
            self.relation_vocab = {r: i for i, r in enumerate(RELATION_ORDER)}

        # entity_vocab
        if entity_vocab is not None:
            self.entity_vocab = dict(entity_vocab)
        else:
            self.entity_vocab = {"[UNK]": 0}
            for n in self.base_kg.get("nodes", []):
                self._ensure_entity(n["id"])

    @staticmethod
    def _load_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def from_base_kg(cls,
                     base_kg: Dict[str, Any],
                     *,
                     max_nodes: int = 128,
                     extra_relations: Optional[List[str]] = None):
        """
        用内存中的基础图谱初始化预处理器。
        """
        # 1) entity vocab
        entity_vocab = {"[UNK]": 0}
        for n in base_kg.get("nodes", []):
            nid = n["id"]
            if nid not in entity_vocab:
                entity_vocab[nid] = len(entity_vocab)

        # 2) relation vocab（核心顺序 + base_kg 出现的关系 + 额外关系）
        rels = set(e.get("relation") for e in base_kg.get("edges", []) if e.get("relation"))
        if extra_relations:
            rels.update(extra_relations)
        rels.update(RELATION_ORDER)
        head = [r for r in RELATION_ORDER if r in rels]
        tail = sorted([r for r in rels if r not in RELATION_ORDER])
        ordered = head + tail
        relation_vocab = {r: i for i, r in enumerate(ordered)}

        return cls(base_kg=base_kg,
                   entity_vocab=entity_vocab,
                   relation_vocab=relation_vocab,
                   max_nodes=max_nodes)

    @classmethod
    def from_basekg_file(cls,
                         base_kg_path: str=r"E:\CKRRG\Knowledge graph\cxkg.json",
                         *,
                         max_nodes: int = 128,
                         extra_relations = ("negated_in", "uncertain_in")):
        """
        从文件加载基础图谱并初始化预处理器。
        """
        assert os.path.exists(base_kg_path), f"Base KG file not found: {base_kg_path}"
        base_kg = cls._load_json(base_kg_path)
        extra_relations = list(extra_relations) if extra_relations is not None else None
        return cls.from_base_kg(base_kg=base_kg,
                                max_nodes=max_nodes,
                                extra_relations=extra_relations)

    def _ensure_entity(self, nid: str) -> int:
        if nid not in self.entity_vocab:
            self.entity_vocab[nid] = len(self.entity_vocab)
        return self.entity_vocab[nid]

    # =====================================================
    # dynamic_kg → 稠密图/三元组
    # =====================================================
    def _sanitize_dynamic_kg(self, dkg: Dict) -> Dict:
        nodes: List[Dict] = list(dkg.get("nodes", []))
        edges: List[Dict] = list(dkg.get("edges", []))
        id2node = {n["id"]: n for n in nodes}

        # 只保留词表中存在的关系（通常为核心集合 + 基础图谱中的扩展）
        edges = [e for e in edges
                 if e.get("relation") in self.relation_vocab and e.get("source") and e.get("target")]

        # 边里引用但 nodes 缺失 → 自动补节点
        for e in edges:
            for key in ("source", "target"):
                nid = e[key]
                if nid not in id2node:
                    node = {
                        "id": nid,
                        "name": _pretty_name_from_id(nid),
                        "type": _infer_type_from_id(nid),
                        "parent": "other" if nid not in ("root", "other") else ("root" if nid == "other" else None)
                    }
                    nodes.append(node); id2node[nid] = node

        # 保证 root / other 存在以及 root->other 边
        if "root" not in id2node:
            nodes.insert(0, {"id": "root", "name": "CXR Findings", "type": "root"}); id2node["root"] = nodes[0]
        if "other" not in id2node:
            node_other = {"id": "other", "name": "Other Finding", "type": "category", "parent": "root"}
            nodes.insert(1, node_other); id2node["other"] = node_other
            edges.append({"source": "root", "target": "other", "relation": "has_category"})
        else:
            if not any(e.get("source") == "root" and e.get("target") == "other" for e in edges):
                edges.append({"source": "root", "target": "other", "relation": "has_category"})

        # 去重边
        seen = set()
        dedup = []
        for e in edges:
            key = (e["source"], e["target"], e["relation"])
            if key not in seen:
                seen.add(key); dedup.append(e)

        return {"nodes": nodes, "edges": dedup}

    def dynamic_kg_to_dense(self, dkg: Dict) -> Dict[str, torch.Tensor]:
        dkg = self._sanitize_dynamic_kg(dkg)
        nodes, edges = dkg["nodes"], dkg["edges"]
        R_base = len(self.relation_vocab)

        # 节点顺序：root → other → 其余按出现顺序（最多 max_nodes）
        id2idx_global = {n["id"]: i for i, n in enumerate(nodes)}
        order = list(range(len(nodes)))
        front = [id2idx_global.get("root"), id2idx_global.get("other")]
        front = [i for i in front if i is not None]
        rest = [i for i in order if i not in front]
        keep = (front + rest)[:self.max_nodes]
        kept_set = set(keep)
        local_pos = {gi: pos for pos, gi in enumerate(keep)}

        # node_ids / node_mask
        node_ids = torch.full((1, self.max_nodes), -1, dtype=torch.long)
        node_mask = torch.zeros((1, self.max_nodes), dtype=torch.bool)
        for pos, gi in enumerate(keep):
            nid = nodes[gi]["id"]
            node_ids[0, pos] = self._ensure_entity(nid)
            node_mask[0, pos] = True

        # rel_mat
        rel_mat = torch.full((1, self.max_nodes, self.max_nodes), -1, dtype=torch.long)
        for e in edges:
            rname = e.get("relation")
            si, ti = id2idx_global.get(e["source"]), id2idx_global.get(e["target"])
            if si is None or ti is None or si not in kept_set or ti not in kept_set:
                continue
            i, j = local_pos[si], local_pos[ti]
            r = self.relation_vocab[rname]
            rel_mat[0, i, j] = r
            rel_mat[0, j, i] = r + R_base  # 反向关系编号
        return {"node_ids": node_ids, "rel_mat": rel_mat, "node_mask": node_mask}

    def dynamic_kg_to_triplets(self,
                               dkg: Dict,
                               max_triplets: Optional[int] = None
                               ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        dkg = self._sanitize_dynamic_kg(dkg)
        edges: List[Dict] = dkg["edges"]
        triplets: List[Tuple[int, int, int]] = []
        for e in edges:
            rname = e.get("relation")
            h = self._ensure_entity(e["source"])
            t = self._ensure_entity(e["target"])
            r = self.relation_vocab[rname]
            triplets.append((h, r, t))
        if not triplets:
            # 兜底：至少返回 1 个占位，避免下游空张量报错
            return (torch.zeros(1, 1, dtype=torch.long),
                    torch.zeros(1, 1, dtype=torch.long),
                    torch.zeros(1, 1, dtype=torch.long))
        if max_triplets is not None:
            triplets = triplets[:max_triplets]
        heads = torch.tensor([[h for h, _, _ in triplets]], dtype=torch.long)
        rels  = torch.tensor([[r for _, r, _ in triplets]], dtype=torch.long)
        tails = torch.tensor([[t for _, _, t in triplets]], dtype=torch.long)
        return heads, rels, tails

    # =====================================================
    # RadGraph → 最小动态图谱（仅使用 located_at 建边，modify 只用于短语合成）
    # =====================================================
    def radgraph_to_dynamic(self, entry: Dict) -> Dict:
        ents: Dict[str, Dict] = entry.get("radgraph_entities", {})
        if not ents:
            return {
                "nodes": [
                    {"id": "root", "name": "CXR Findings", "type": "root"},
                    {"id": "other", "name": "Other Finding", "type": "category", "parent": "root"},
                ],
                "edges": [{"source": "root", "target": "other", "relation": "has_category"}]
            }

        nodes: Dict[str, Dict] = {
            "root": {"id": "root", "name": "CXR Findings", "type": "root"},
            "other": {"id": "other", "name": "Other Finding", "type": "category", "parent": "root"}
        }
        edges: List[Dict] = [{"source": "root", "target": "other", "relation": "has_category"}]

        temp: Dict[str, Dict] = {}
        for rid, ent in ents.items():
            phrase = compose_phrase(rid, ents) or ent.get("tokens", "")
            pol = label_to_polarity(ent.get("label", ""))
            lab = (ent.get("label", "") or "").lower()

            if lab.startswith("anatomy"):
                nid = f"anat_{slugify(phrase)}"
                if nid not in nodes:
                    nodes[nid] = {"id": nid, "name": _pretty_name_from_id(nid), "type": "anatomy", "parent": "other"}
                temp[rid] = {"type": "anatomy", "nid": nid, "polarity": pol}

            elif lab.startswith("observation"):
                phrase_l = phrase.lower()
                is_normal = ("normal" in phrase_l) or ("clear" in phrase_l) or ("intact" in phrase_l) or ("within normal limits" in phrase_l)
                nid = f"find_{slugify(phrase)}"
                if nid not in nodes:
                    nodes[nid] = {"id": nid, "name": _pretty_name_from_id(nid), "type": "finding", "parent": "other"}
                temp[rid] = {"type": "finding", "nid": nid, "polarity": pol, "is_normal": is_normal}

            else:
                nid = f"find_{slugify(phrase)}"
                if nid not in nodes:
                    nodes[nid] = {"id": nid, "name": _pretty_name_from_id(nid), "type": "finding", "parent": "other"}
                temp[rid] = {"type": "finding", "nid": nid, "polarity": pol, "is_normal": False}

        # 根据 located_at 连接 observation 与 anatomy
        for rid, ent in ents.items():
            for rel, tgt in ent.get("relations", []):
                if rel.lower() != "located_at":
                    continue
                src, dst = temp.get(str(rid)), temp.get(str(tgt))
                if not src or not dst:
                    continue
                if src["type"] == "anatomy" and dst["type"] == "finding":
                    apart, fpart = src, dst
                elif src["type"] == "finding" and dst["type"] == "anatomy":
                    apart, fpart = dst, src
                else:
                    continue

                a_id, f_id = apart["nid"], fpart["nid"]
                if fpart.get("is_normal"):
                    norm_id = f"find_{slugify(nodes[a_id]['name'] + ' normal')}"
                    if norm_id not in nodes:
                        nodes[norm_id] = {"id": norm_id, "name": _pretty_name_from_id(norm_id),
                                          "type": "normal", "parent": a_id}
                    edges.append({"source": a_id, "target": norm_id, "relation": "normal_of"})
                else:
                    pol = fpart.get("polarity", "unknown")
                    if pol == "negative":
                        edges.append({"source": a_id, "target": f_id, "relation": "negated_in"})
                    elif pol == "uncertain":
                        edges.append({"source": a_id, "target": f_id, "relation": "uncertain_in"})
                    else:
                        edges.append({"source": a_id, "target": f_id, "relation": "located_in"})

        # 去重边
        seen = set(); dedup = []
        for e in edges:
            key = (e["source"], e["target"], e["relation"])
            if key not in seen:
                seen.add(key); dedup.append(e)

        return {"nodes": list(nodes.values()), "edges": dedup}

    # ---------- 统一对外接口 ----------
    def from_dynamic_kg(self, dkg: Dict):
        """
        输入：dynamic_kg（包含 nodes+edges）
        输出：graph_inputs(dict) 与三元组 (heads, rels, tails)
        """
        graph_inputs = self.dynamic_kg_to_dense(dkg)
        triplets = self.dynamic_kg_to_triplets(dkg)
        return graph_inputs, triplets

    def from_radgraph(self, entry: Dict):
        """
        输入：含 radgraph_entities 的样本
        输出：graph_inputs、triplets，以及由 RadGraph 生成的最小 dkg
        """
        dkg = self.radgraph_to_dynamic(entry)
        graph_inputs = self.dynamic_kg_to_dense(dkg)
        triplets = self.dynamic_kg_to_triplets(dkg)
        return graph_inputs, triplets, dkg

