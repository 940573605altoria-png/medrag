"""父子层级分块（T043）—— AutoMerging：只索引子/叶块，命中多则上浮父块。

**范式（plan.md 已定）**：结构层级 `AutoMergingRetriever`——子块进向量库、**父块=大段连续原文
（非 LLM 摘要，无损可溯源）**进 docstore；检索命中同一父块的子块够多时上浮返回父块。这样既有子块
的精确召回，又有父块的完整上下文，且证据始终能回指原文（constitution I）。

本模块两条路：
1. **QA 分块（b 数据集，纯逻辑、本地全测，本模块核心）**：短答案→单叶节点；长答案→父（整段
   答案）+ 子（按尺寸切）。**子块 embed 文本前置 question**（contextual，进 embed 不进展示文本），
   提升短查询对长答案子块的召回。
2. **文档分块（a 数据集，结构感知+尺寸）**：复用 LlamaIndex `HierarchicalNodeParser`
   (`chunk_sizes=[1024,256]`)——守卫导入，功能跑在装了 llama-index 的环境（AutoDL）。

**可复现/可消融（constitution III）**：`chunk_sizes`、层数、是否 contextual、短答案阈值均为
`ChunkConfig` 开关，单独消融看检索命中率/faithfulness。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Sequence

from src.contracts.schemas import KnowledgeNode

# 句子切分：中英标点 + 换行，保留尾标点。
_SENT = re.compile(r"[^。！？.!?\n]+[。！？.!?]?")
_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z0-9]+")


def estimate_tokens(text: str) -> int:
    """多语粗估 token 数：CJK 字符各计 1 + 拉丁词各计 1。够用于尺寸阈值判定。"""
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    return cjk + latin


def split_sentences(text: str) -> list[str]:
    """按句切分（去空白空句）。"""
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


@dataclass
class ChunkConfig:
    """分块开关（全部可消融）。"""

    chunk_sizes: tuple[int, ...] = (1024, 256)  # 父/子层目标 token（LlamaIndex 两层起步）
    short_qa_tokens: int = 256                  # 答案 ≤ 此值 → 单叶节点（不建父子）
    child_tokens: int = 256                     # 长答案子块目标尺寸
    contextual: bool = True                     # 子块 embed 文本前置 question/标题
    min_chunk_tokens: int = 16                  # 过短子块并入前一块（护栏）


@dataclass
class Chunk:
    """一个层级节点。`text`=展示原文（无损溯源）；`embed_text`=送嵌入的文本（可含 contextual 前置）。"""

    chunk_id: str
    text: str
    embed_text: str
    level: int                       # 0=叶/子，越大越靠父
    is_leaf: bool
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def contextualize(text: str, context: str | None) -> str:
    """contextual 前置：`{context}\n{text}`（仅用于 embed 文本，不改展示文本）。"""
    if not context:
        return text
    return f"{context}\n{text}"


def _split_by_size(sentences: Sequence[str], target: int, min_tokens: int) -> list[str]:
    """按句累积到 ~target token 成块；过短的尾块并入前一块。"""
    chunks: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for s in sentences:
        st = estimate_tokens(s)
        if buf and buf_tok + st > target:
            chunks.append(" ".join(buf))
            buf, buf_tok = [s], st
        else:
            buf.append(s)
            buf_tok += st
    if buf:
        tail = " ".join(buf)
        if chunks and estimate_tokens(tail) < min_tokens:
            chunks[-1] = chunks[-1] + " " + tail
        else:
            chunks.append(tail)
    return chunks


def chunk_qa(
    question: str,
    answer: str,
    *,
    doc_id: str | None = None,
    cfg: ChunkConfig | None = None,
) -> list[Chunk]:
    """QA 对分块：短答案→单叶；长答案→父（整段）+ 子（按尺寸，embed 前置 question）。

    返回的节点里**叶/子块是要入向量库的**（`is_leaf=True`）；父块进 docstore 供 AutoMerging 上浮。
    """
    cfg = cfg or ChunkConfig()
    context = question if cfg.contextual else None

    # 短答案：单叶节点，无父。
    if estimate_tokens(answer) <= cfg.short_qa_tokens:
        leaf = Chunk(
            chunk_id=_new_id("leaf"),
            text=answer,
            embed_text=contextualize(answer, context),
            level=0, is_leaf=True,
            metadata={"doc_id": doc_id, "question": question, "kind": "qa_short"},
        )
        return [leaf]

    # 长答案：父块（整段答案）+ 子块。
    parent = Chunk(
        chunk_id=_new_id("parent"),
        text=answer,
        embed_text=answer,                # 父块不入向量库，embed_text 仅占位
        level=1, is_leaf=False,
        metadata={"doc_id": doc_id, "question": question, "kind": "qa_parent"},
    )
    sub_texts = _split_by_size(split_sentences(answer), cfg.child_tokens, cfg.min_chunk_tokens)
    nodes: list[Chunk] = [parent]
    for sub in sub_texts:
        nodes.append(Chunk(
            chunk_id=_new_id("leaf"),
            text=sub,
            embed_text=contextualize(sub, context),
            level=0, is_leaf=True, parent_id=parent.chunk_id,
            metadata={"doc_id": doc_id, "question": question, "kind": "qa_child"},
        ))
    return nodes


def leaves(chunks: Sequence[Chunk]) -> list[Chunk]:
    """只取叶/子块（入向量库的对象）。"""
    return [c for c in chunks if c.is_leaf]


def to_knowledge_nodes(
    chunks: Sequence[Chunk],
    *,
    collection: str,
    lang: str | None = None,
    source_ids: Sequence[str] | None = None,
) -> list[KnowledgeNode]:
    """把叶/子块映射成 [schemas.KnowledgeNode]（供 T044 入库）。

    `text`=展示原文；contextual 的 embed 文本放进 `metadata['embed_text']`，索引端据此嵌入，
    展示端仍用无损原文（训推一致 + 不污染溯源）。
    """
    out: list[KnowledgeNode] = []
    for c in leaves(chunks):
        meta = dict(c.metadata)
        meta["embed_text"] = c.embed_text
        if c.parent_id:
            meta["parent_id"] = c.parent_id
        out.append(KnowledgeNode(
            node_id=c.chunk_id,
            text=c.text,
            collection=collection,
            lang=lang,
            source_ids=list(source_ids or []),
            metadata=meta,
        ))
    return out


def hierarchical_chunk_documents(documents: Sequence[str], cfg: ChunkConfig | None = None):
    """文档分块（a 数据集）：复用 LlamaIndex `HierarchicalNodeParser`。守卫导入。

    返回 LlamaIndex nodes（父子+关系）；子/叶入 ChromaVectorStore、全节点入 docstore 由 T044 接。
    功能跑在装了 llama-index 的环境（AutoDL）。
    """
    cfg = cfg or ChunkConfig()
    try:
        from llama_index.core.node_parser import HierarchicalNodeParser  # noqa: PLC0415
        from llama_index.core.schema import Document  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "文档分块需要 llama-index；请 `pip install llama-index-core`（QA 分块走 chunk_qa 无需它）。"
        ) from exc
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=list(cfg.chunk_sizes))
    docs = [Document(text=t) for t in documents]
    return parser.get_nodes_from_documents(docs)


__all__ = [
    "estimate_tokens",
    "split_sentences",
    "ChunkConfig",
    "Chunk",
    "contextualize",
    "chunk_qa",
    "leaves",
    "to_knowledge_nodes",
    "hierarchical_chunk_documents",
]
