"""文本嵌入服务（T014）—— Qwen3-Embedding 4B，非对称查询编码 + 归一化 + 训推一致。

**为什么非对称**（plan.md 已定）：Qwen3-Embedding 这类指令型嵌入器，**query 端加任务指令前缀、
document 端不加**，能显著提升检索相关性。格式为 `Instruct: {task}\nQuery: {q}`。

**训推一致（铁律 II 推论）**：索引端与查询端**同一模型 + 同一预处理**（contextual 前置、归一化）、
固定模型版本。距离用 **cosine**——向量 L2 归一化后入 ChromaDB（cosine space），点积即余弦。

模型加载守卫导入（sentence-transformers/transformers 缺了也能 import 本模块）；**格式化与归一化是
纯逻辑、本地全测**，真实 encode 用注入的 `encode_fn` 测、或在 AutoDL 装模型跑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
# 检索任务的默认指令（可按医学任务覆盖）。
DEFAULT_QUERY_INSTRUCTION = (
    "Given a medical query, retrieve relevant passages that answer the query"
)

# 后端编码器签名：一批文本 → (n, d) 数组（未归一化）。
EncodeFn = Callable[[Sequence[str]], np.ndarray]


def format_query(query: str, instruction: str | None = None) -> str:
    """非对称查询编码：query 端加指令前缀。"""
    instruction = instruction or DEFAULT_QUERY_INSTRUCTION
    return f"Instruct: {instruction}\nQuery: {query}"


def format_document(document: str) -> str:
    """document 端不加指令（非对称）。"""
    return document


def l2_normalize(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """按行 L2 归一化（零向量安全）。归一化后点积 = 余弦相似度。"""
    v = np.asarray(vectors, dtype=float)
    if v.ndim == 1:
        v = v[None, :]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, eps)


@dataclass
class EmbedConfig:
    """嵌入配置——模型 id 可被环境变量覆盖，固定版本可复现。"""

    model_id: str = field(
        default_factory=lambda: os.environ.get("MEDRAG_EMBED_MODEL", DEFAULT_MODEL_ID)
    )
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    normalize: bool = True


@dataclass
class TextEmbedder:
    """a/b 文本嵌入器。索引端与查询端共用一个实例 → 训推一致。

    `encode_fn` 可注入（测试/自定义后端）；为空则惰性加载 Qwen3-Embedding（守卫导入）。
    """

    config: EmbedConfig = field(default_factory=EmbedConfig)
    encode_fn: EncodeFn | None = None
    _backend: EncodeFn | None = field(default=None, init=False, repr=False)

    def _encoder(self) -> EncodeFn:
        if self.encode_fn is not None:
            return self.encode_fn
        if self._backend is None:
            self._backend = self._load_backend()
        return self._backend

    def _load_backend(self) -> EncodeFn:
        """守卫加载 sentence-transformers 后端。"""
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "文本嵌入需要 sentence-transformers + 模型权重（Qwen3-Embedding）；"
                "本地可注入 encode_fn 测，功能跑在 AutoDL。"
            ) from exc
        model = SentenceTransformer(self.config.model_id, trust_remote_code=True)
        return lambda texts: np.asarray(model.encode(list(texts)))

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """编码查询（加指令前缀 + 归一化）。"""
        formatted = [format_query(q, self.config.query_instruction) for q in queries]
        return self._encode(formatted)

    def encode_documents(self, documents: Sequence[str]) -> np.ndarray:
        """编码文档（不加指令 + 归一化）。索引对象 = 分块的子/叶块 embed 文本。"""
        formatted = [format_document(d) for d in documents]
        return self._encode(formatted)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.asarray(self._encoder()(texts), dtype=float)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        return l2_normalize(vecs) if self.config.normalize else vecs


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_QUERY_INSTRUCTION",
    "EncodeFn",
    "format_query",
    "format_document",
    "l2_normalize",
    "EmbedConfig",
    "TextEmbedder",
]
