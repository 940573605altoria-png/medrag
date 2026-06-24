"""a/b 叶块入库 + docstore 承载 AutoMerging（T044）。

把 [chunk] 产出的层级节点落地成可检索结构：
- **叶/子块 → 向量库**（[store.VectorStore] 的 `a_drug`/`b_medqa` collection）：用 [embed_text] 嵌入
  每个叶块的 `embed_text`（含 contextual 前置）、cosine 入库；展示文本存原文（溯源无损）。
- **全节点（含父块）→ docstore**：AutoMerging 上浮要靠父块，父块**不进向量库**但必须可按 id 取回，
  且要记 child→parent 关系。docstore 是纯 Python 结构（可 JSON 落盘跨会话），T045 检索时据此上浮。

向量库经 [store.VectorStore] 守卫 chromadb；docstore 纯逻辑。嵌入器可注入 `encode_fn`，故本模块在装了
chromadb 的环境（本地已装）可端到端功能测；真实嵌入模型仍在 AutoDL。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.rag import chunk as ck
from src.rag.embed_text import TextEmbedder
from src.rag.store import VectorStore, validate_collection


@dataclass
class NodeRecord:
    """docstore 里的一个节点（叶或父）。"""

    node_id: str
    text: str                       # 展示原文（无损）
    parent_id: str | None
    level: int
    is_leaf: bool
    metadata: dict = field(default_factory=dict)


class DocStore:
    """承载 AutoMerging 的节点仓：按 id 取节点、查父/查子（纯逻辑，可 JSON 持久化）。"""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeRecord] = {}

    def add_chunks(self, chunks: Sequence[ck.Chunk]) -> None:
        for c in chunks:
            self._nodes[c.chunk_id] = NodeRecord(
                node_id=c.chunk_id, text=c.text, parent_id=c.parent_id,
                level=c.level, is_leaf=c.is_leaf, metadata=dict(c.metadata),
            )

    def get(self, node_id: str) -> NodeRecord | None:
        return self._nodes.get(node_id)

    def parent_of(self, node_id: str) -> NodeRecord | None:
        rec = self._nodes.get(node_id)
        if rec is None or rec.parent_id is None:
            return None
        return self._nodes.get(rec.parent_id)

    def children_of(self, parent_id: str) -> list[NodeRecord]:
        return [n for n in self._nodes.values() if n.parent_id == parent_id]

    def __len__(self) -> int:
        return len(self._nodes)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {nid: vars(rec) for nid, rec in self._nodes.items()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DocStore":
        store = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store._nodes = {nid: NodeRecord(**rec) for nid, rec in data.items()}
        return store


@dataclass
class IndexResult:
    collection: str
    n_leaves_indexed: int
    n_nodes_stored: int


def _leaf_metadata(
    leaf: ck.Chunk, *, lang: str | None, source_ids: Sequence[str] | None
) -> dict:
    """构造 chroma 安全的 metadata（仅 str/int/float/bool，丢 None、列表拼成串）。"""
    meta: dict = {"kind": str(leaf.metadata.get("kind", "leaf"))}
    if leaf.parent_id:
        meta["parent_id"] = leaf.parent_id
    if lang:
        meta["lang"] = lang
    q = leaf.metadata.get("question")
    if q:
        meta["question"] = str(q)
    doc_id = leaf.metadata.get("doc_id")
    if doc_id:
        meta["doc_id"] = str(doc_id)
    if source_ids:
        meta["source_ids"] = "|".join(source_ids)  # 溯源：保来源 id
    return meta


def index_chunks(
    chunks: Sequence[ck.Chunk],
    *,
    store: VectorStore,
    embedder: TextEmbedder,
    collection: str,
    docstore: DocStore,
    lang: str | None = None,
    source_ids: Sequence[str] | None = None,
) -> IndexResult:
    """叶块嵌入入向量库 + 全节点入 docstore。`collection` 须是 a_drug/b_medqa。"""
    validate_collection(collection)
    leaves = ck.leaves(chunks)
    if leaves:
        embeddings = embedder.encode_documents([lf.embed_text for lf in leaves])
        store.add(
            collection,
            ids=[lf.chunk_id for lf in leaves],
            embeddings=embeddings.tolist(),
            metadatas=[_leaf_metadata(lf, lang=lang, source_ids=source_ids) for lf in leaves],
            documents=[lf.text for lf in leaves],
        )
    docstore.add_chunks(chunks)
    return IndexResult(collection=collection, n_leaves_indexed=len(leaves),
                       n_nodes_stored=len(chunks))


def index_qa(
    question: str,
    answer: str,
    *,
    store: VectorStore,
    embedder: TextEmbedder,
    collection: str,
    docstore: DocStore,
    cfg: ck.ChunkConfig | None = None,
    lang: str | None = None,
    source_ids: Sequence[str] | None = None,
) -> IndexResult:
    """便捷：QA 对 → 分块 → 入库（b_medqa 主用）。"""
    chunks = ck.chunk_qa(question, answer, doc_id=None, cfg=cfg)
    return index_chunks(chunks, store=store, embedder=embedder, collection=collection,
                        docstore=docstore, lang=lang, source_ids=source_ids)


__all__ = [
    "NodeRecord",
    "DocStore",
    "IndexResult",
    "index_chunks",
    "index_qa",
]
