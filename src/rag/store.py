"""ChromaDB 向量库 + collection 管理（T013）—— RAG 全链的存储地基。

按数据集类型分 collection（plan.md），互不串味、便于按通道消融（名字带 a/b/c 前缀对应
plan 的 a 药品 / b 医学QA / c CT；chromadb 要求集合名≥3 字符，故 a→a_drug、b→b_medqa）：
- `a_drug`       药品信息（文本）
- `b_medqa`      医学 QA（文本）
- `c_text`       CT 报告文本
- `c_img_whole`  CT 全图视觉向量
- `c_img_roi`    CT ROI 视觉向量

统一 cosine 空间；支持 metadata `where` 过滤（如按面积带/语言/case_id 筛）。
`path=None` 用内存 EphemeralClient（单测/消融），给定则 PersistentClient 落数据盘
（AutoDL `/root/autodl-tmp/chroma`）。

守卫导入 chromadb：无该包也能 import 本模块（拿到 COLLECTIONS/校验），建库才需 chromadb。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# 五个固定 collection（按数据集类型分，禁止混入未知名；名字≥3 字符满足 chromadb 约束）
COLLECTIONS = ("a_drug", "b_medqa", "c_text", "c_img_whole", "c_img_roi")


def validate_collection(name: str) -> str:
    """校验 collection 名属于固定集合，否则报清晰错误。纯函数，无需 chromadb。"""
    if name not in COLLECTIONS:
        raise ValueError(f"未知 collection {name!r}，应属 {COLLECTIONS}")
    return name


def _require_chromadb():
    try:
        import chromadb  # noqa: PLC0415

        return chromadb
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError("向量库需要 chromadb：pip install chromadb") from exc


@dataclass
class StoreConfig:
    path: str | None = None     # None=内存(测试)；否则持久化目录
    space: str = "cosine"


class VectorStore:
    """ChromaDB 薄封装：固定 collection、cosine、metadata where。"""

    def __init__(self, config: StoreConfig | None = None):
        chromadb = _require_chromadb()
        cfg = config or StoreConfig()
        self.space = cfg.space
        self._client = (
            chromadb.PersistentClient(path=cfg.path) if cfg.path
            else chromadb.EphemeralClient()
        )

    def collection(self, name: str):
        """取/建 collection（cosine 空间）。"""
        validate_collection(name)
        return self._client.get_or_create_collection(
            name, metadata={"hnsw:space": self.space}
        )

    def add(self, name: str, ids: Sequence[str], embeddings: Sequence[Sequence[float]],
            *, metadatas: Sequence[dict] | None = None,
            documents: Sequence[str] | None = None) -> None:
        """写入向量（带 id/可选 metadata/可选原文）。"""
        self.collection(name).add(
            ids=list(ids), embeddings=[list(e) for e in embeddings],
            metadatas=list(metadatas) if metadatas else None,
            documents=list(documents) if documents else None,
        )

    def query(self, name: str, query_embeddings: Sequence[Sequence[float]],
              *, n_results: int = 5, where: dict | None = None,
              include: Sequence[str] | None = None) -> dict[str, Any]:
        """按向量近邻检索，支持 metadata `where` 过滤。返回 chromadb 原始结果 dict。"""
        kwargs: dict[str, Any] = {
            "query_embeddings": [list(q) for q in query_embeddings],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        if include:
            kwargs["include"] = list(include)
        return self.collection(name).query(**kwargs)

    def count(self, name: str) -> int:
        return self.collection(name).count()

    def reset(self, name: str) -> None:
        """删除并重建某 collection（消融/重建索引用）。"""
        validate_collection(name)
        try:
            self._client.delete_collection(name)
        except Exception:  # noqa: BLE001 - 不存在则忽略
            pass


__all__ = ["COLLECTIONS", "validate_collection", "StoreConfig", "VectorStore"]
