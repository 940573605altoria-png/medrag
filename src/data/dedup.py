"""库内去重（T039）—— 精确(SHA1) + 结构化键 + 近重复(MinHash/LSH)，来源 ID 并入溯源。

RAG 库去重的两条铁律（plan.md）：
1. **保守**：近重复阈值取高（Jaccard ≥ 0.85），宁可漏去重也不误删——覆盖率优先。
2. **溯源不丢**：删掉的副本，其 `source_id`（及节点 id）**并入存活节点的 `source_ids`**，
   下游引用任一来源都能回指（constitution I）。

三级（**库内**做，不跨库；文档级先去重再分块）：
- 精确：归一化文本 SHA1，删完全相同。
- 结构化键：a 药品按 (药名+章节) 等业务键精确去重。
- 近重复：word/char n-gram shingle → Jaccard ≥ τ。datasketch MinHash+LSH 跑规模（守卫导入）；
  纯 jaccard 暴力法作本地测/小集参照。

纯逻辑（哈希/集合/并查集）本地全测；datasketch 路守卫。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Hashable, Sequence

from src.contracts.schemas import KnowledgeNode
from src.data.normalize import NormalizeConfig, normalize_text


@dataclass
class DedupResult:
    kept: list[KnowledgeNode]
    removed_ids: list[str] = field(default_factory=list)
    merges: dict[str, list[str]] = field(default_factory=dict)  # survivor_id -> [removed_id]


def sha1_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def shingles(text: str, *, n: int = 3, char: bool = False) -> set[str]:
    """n-gram shingle 集合（默认词级 3-gram）。"""
    units = list(text) if char else text.split()
    if len(units) < n:
        return {" ".join(units)} if units else set()
    sep = "" if char else " "
    return {sep.join(units[i : i + n]) for i in range(len(units) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _merge_into(survivor: KnowledgeNode, dup: KnowledgeNode) -> None:
    """把 dup 的来源并入 survivor（去重保序），dup 的 node_id 也并入以保溯源。"""
    seen = list(survivor.source_ids)
    for sid in [*dup.source_ids, dup.node_id]:
        if sid not in seen:
            seen.append(sid)
    survivor.source_ids = seen
    if "dedup" not in survivor.flags:
        survivor.flags = [*survivor.flags, "dedup"]


def _dedup_by_key(
    nodes: Sequence[KnowledgeNode], key_fn: Callable[[KnowledgeNode], Hashable]
) -> DedupResult:
    """通用按键去重：同键保第一个，其余并入溯源。"""
    survivors: dict[Hashable, KnowledgeNode] = {}
    kept: list[KnowledgeNode] = []
    removed: list[str] = []
    merges: dict[str, list[str]] = {}
    for node in nodes:
        key = key_fn(node)
        if key in survivors:
            surv = survivors[key]
            _merge_into(surv, node)
            removed.append(node.node_id)
            merges.setdefault(surv.node_id, []).append(node.node_id)
        else:
            survivors[key] = node
            kept.append(node)
    return DedupResult(kept=kept, removed_ids=removed, merges=merges)


def exact_dedup(
    nodes: Sequence[KnowledgeNode], *, norm_config: NormalizeConfig | None = None
) -> DedupResult:
    """精确去重：归一化文本 SHA1 相同 → 合并。"""
    return _dedup_by_key(nodes, lambda n: sha1_key(normalize_text(n.text, norm_config)))


def structured_dedup(
    nodes: Sequence[KnowledgeNode], key_fields: Sequence[str]
) -> DedupResult:
    """结构化键去重（a 药品）：按 metadata 中 `key_fields`（如药名+章节）精确去重。"""
    def key_fn(n: KnowledgeNode) -> Hashable:
        return tuple(str(n.metadata.get(f, "")) for f in key_fields)

    return _dedup_by_key(nodes, key_fn)


def near_dedup(
    nodes: Sequence[KnowledgeNode],
    *,
    threshold: float = 0.85,
    n: int = 3,
    char: bool = False,
    norm_config: NormalizeConfig | None = None,
    backend: str = "auto",
) -> DedupResult:
    """近重复去重：Jaccard ≥ threshold 视为重复，聚类后保第一个、其余并入溯源。

    `backend`：'minhash'(datasketch,规模) / 'bruteforce'(纯 jaccard,小集/测试) / 'auto'(有则用)。
    """
    shingle_sets = [
        shingles(normalize_text(nd.text, norm_config), n=n, char=char) for nd in nodes
    ]
    if backend == "auto":
        backend = "minhash" if _has_datasketch() else "bruteforce"

    if backend == "minhash":
        pairs = _minhash_pairs(shingle_sets, threshold)
    else:
        pairs = _bruteforce_pairs(shingle_sets, threshold)

    return _merge_clusters(nodes, pairs)


def _bruteforce_pairs(sets: list[set], threshold: float) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if jaccard(sets[i], sets[j]) >= threshold:
                pairs.append((i, j))
    return pairs


def _has_datasketch() -> bool:
    try:
        import datasketch  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _minhash_pairs(sets: list[set], threshold: float, num_perm: int = 128) -> list[tuple[int, int]]:
    """datasketch MinHash+LSH 找候选近重复对（守卫导入）。"""
    try:
        from datasketch import MinHash, MinHashLSH  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError("MinHash 去重需要 datasketch：pip install datasketch") from exc
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mh_list = []
    for idx, sh in enumerate(sets):
        mh = MinHash(num_perm=num_perm)
        for tok in sh:
            mh.update(tok.encode("utf-8"))
        lsh.insert(str(idx), mh)
        mh_list.append(mh)
    pairs = []
    for i, mh in enumerate(mh_list):
        for cand in lsh.query(mh):
            j = int(cand)
            if j > i:
                pairs.append((i, j))
    return pairs


def _merge_clusters(
    nodes: Sequence[KnowledgeNode], pairs: list[tuple[int, int]]
) -> DedupResult:
    """并查集聚类近重复对，每簇保最早节点、其余并入溯源。"""
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)  # 早的当根（保最早为存活）

    kept: list[KnowledgeNode] = []
    removed: list[str] = []
    merges: dict[str, list[str]] = {}
    root_node: dict[int, KnowledgeNode] = {}
    for idx, node in enumerate(nodes):
        root = find(idx)
        if root not in root_node:
            root_node[root] = node
            kept.append(node)
        else:
            surv = root_node[root]
            _merge_into(surv, node)
            removed.append(node.node_id)
            merges.setdefault(surv.node_id, []).append(node.node_id)
    return DedupResult(kept=kept, removed_ids=removed, merges=merges)


__all__ = [
    "DedupResult",
    "sha1_key",
    "shingles",
    "jaccard",
    "exact_dedup",
    "structured_dedup",
    "near_dedup",
]
