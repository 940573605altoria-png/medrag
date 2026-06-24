"""hybrid 文本检索（T045）—— BM25 + dense → RRF 融合 → AutoMerging 上浮 → RetrievalResult。

**为什么 hybrid（plan.md 已定）**：药名/编码/剂量这类**精确词**靠 BM25，语义近义靠 **dense**
（Qwen3-Embedding）；两路用 **RRF（reciprocal rank fusion）** 按**排名**融合——跨通道鲁棒，
不受 BM25 原始分无界/不可比之累（阈值只用 top-N / 归一化分，拒答阈值留到 rerank 之后 T046）。

**AutoMerging 上浮**：dense/BM25 命中的是子/叶块；若同一父块的子块**命中比例够高**，就**上浮成父块**
（大段连续原文），既精准又完整、且溯源无损（constitution I）。

架构对齐：沿用本仓自建的 [store.VectorStore]/[index_text.DocStore]，**RRF 与上浮直接实现**（短小、
可控、纯逻辑本地全测），不引入 LlamaIndex 的 Node 抽象。BM25 用 `rank_bm25`（守卫导入）；dense 路
chromadb 本地已装可功能测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from src.contracts.schemas import (
    AbstainReason,
    EvidenceItem,
    Modality,
    RetrievalResult,
)
from src.rag.embed_text import TextEmbedder
from src.rag.index_text import DocStore
from src.rag.store import VectorStore

_TOKEN = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """多语分词：拉丁词整体 + CJK 单字（喂 BM25）。小写归一。"""
    return [t.lower() for t in _TOKEN.findall(text)]


def rrf_fuse(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """RRF：对多个有序 id 列表，score(id)=Σ 1/(k+rank)，按分降序返回 (id, score)。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def automerge(
    scored: Sequence[tuple[str, float]],
    docstore: DocStore,
    *,
    merge_ratio: float = 0.5,
) -> list[tuple[str, float]]:
    """AutoMerging 上浮：同一父块被命中的子块比例 ≥ merge_ratio → 合并成父块（取子块最高分）。

    未达阈值的子块、无父块的节点原样保留。返回按分降序的 (id, score)。
    """
    score_map = dict(scored)
    by_parent: dict[str, list[str]] = {}
    for nid, _ in scored:
        rec = docstore.get(nid)
        if rec is not None and rec.parent_id:
            by_parent.setdefault(rec.parent_id, []).append(nid)

    merged: dict[str, float] = {}
    consumed: set[str] = set()
    for parent_id, kids in by_parent.items():
        total = len(docstore.children_of(parent_id)) or len(kids)
        if total > 0 and len(kids) / total >= merge_ratio:
            merged[parent_id] = max(score_map[k] for k in kids)
            consumed.update(kids)

    result: list[tuple[str, float]] = []
    added: set[str] = set()
    for nid, sc in scored:
        if nid in consumed:
            pid = docstore.get(nid).parent_id
            if pid in merged and pid not in added:
                result.append((pid, merged[pid]))
                added.add(pid)
            continue
        result.append((nid, sc))
    result.sort(key=lambda kv: kv[1], reverse=True)
    return result


class BM25Index:
    """BM25 稀疏检索（精确词通道）。守卫 `rank_bm25`。"""

    def __init__(self, ids: Sequence[str], texts: Sequence[str]):
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RuntimeError("BM25 检索需要 rank_bm25：pip install rank-bm25") from exc
        self._ids = list(ids)
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    @classmethod
    def from_docstore(cls, docstore: DocStore) -> "BM25Index":
        """从 docstore 的**叶块**建 BM25（与向量库索引对象一致）。"""
        leaves = [n for n in docstore._nodes.values() if n.is_leaf]  # noqa: SLF001
        return cls([n.node_id for n in leaves], [n.text for n in leaves])

    def search(self, query: str, top_n: int) -> list[str]:
        if not self._ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self._ids)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in order[:top_n]]


@dataclass
class RetrieveConfig:
    top_n: int = 10
    rrf_k: int = 60
    dense_top: int = 50
    bm25_top: int = 50
    merge_ratio: float = 0.5
    normalize_scores: bool = True   # RRF 分按最大值归一到 (0,1]


def _citation(rec, node_id: str) -> str:
    if rec is None:
        return node_id
    q = rec.metadata.get("question")
    doc_id = rec.metadata.get("doc_id")
    return str(q or doc_id or node_id)


@dataclass
class HybridRetriever:
    """组合 dense(向量库) + BM25 → RRF → AutoMerge → RetrievalResult。

    `bm25=None` 时退化为纯 dense（本地无 rank_bm25 也能跑）。拒答阈值不在此卡（留 T046），
    仅当无任何候选时按 NO_EVIDENCE 拒答。
    """

    store: VectorStore
    embedder: TextEmbedder
    collection: str
    docstore: DocStore
    bm25: BM25Index | None = None
    config: RetrieveConfig = field(default_factory=RetrieveConfig)

    def retrieve(self, query: str) -> RetrievalResult:
        cfg = self.config
        qv = self.embedder.encode_queries([query])
        dres = self.store.query(self.collection, qv.tolist(), n_results=cfg.dense_top)
        dense_ids = dres.get("ids", [[]])[0] if dres.get("ids") else []

        rankings: list[Sequence[str]] = [dense_ids]
        if self.bm25 is not None:
            rankings.append(self.bm25.search(query, cfg.bm25_top))

        fused = rrf_fuse(rankings, k=cfg.rrf_k)
        merged = automerge(fused, self.docstore, merge_ratio=cfg.merge_ratio)
        top = merged[: cfg.top_n]

        if not top:
            return RetrievalResult(query=query, evidence=[], abstain=True,
                                   abstain_reason=AbstainReason.NO_EVIDENCE)

        max_s = max(s for _, s in top) or 1.0
        evidence = []
        for nid, sc in top:
            rec = self.docstore.get(nid)
            evidence.append(EvidenceItem(
                source_id=nid,
                citation=_citation(rec, nid),
                score=(sc / max_s if cfg.normalize_scores else sc),
                modality=Modality.TEXT,
                text=(rec.text if rec is not None else None),
            ))
        return RetrievalResult(query=query, evidence=evidence, abstain=False)


__all__ = [
    "tokenize",
    "rrf_fuse",
    "automerge",
    "BM25Index",
    "RetrieveConfig",
    "HybridRetriever",
]
