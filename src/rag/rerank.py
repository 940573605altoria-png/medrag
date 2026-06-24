"""Qwen3 Reranker 精排 + 拒答门（T046）—— 召回（已 AutoMerge）→ cross-encoder Top-K → 拒答。

**为什么要 cross-encoder 重排**：T045 的 BM25/dense 是 bi-encoder（query 与 doc 各自编码、点积），
快但粗。reranker 是 **cross-encoder**——把 (query, passage) 拼起来一起过模型，精度高得多，但贵，
所以只对召回的少量候选跑、取 Top-K（默认 5）。

**拒答门放这里（plan.md 已定）**：BM25 原始分无界不可比，故最终拒答阈值卡在**重排后的统一分**上——
top 候选的重排分低于阈值 → 整体拒答（`LOW_CONFIDENCE`），绝不硬答（constitution I）。统一分用
sigmoid 把 cross-encoder logit 映射到 (0,1)，阈值好解释（默认 0.5）。

reranker 模型守卫导入；**排序与拒答是纯逻辑、本地全测**，真实打分用注入 `score_fn` 测、或 AutoDL
装模型跑。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

from src.contracts.schemas import AbstainReason, RetrievalResult

DEFAULT_RERANKER_ID = "Qwen/Qwen3-Reranker-4B"

# 打分后端：(query, passages) → 每条 passage 的相关性原始分（logit，越大越相关）。
ScoreFn = Callable[[str, Sequence[str]], Sequence[float]]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class RerankConfig:
    """精排配置——模型 id 可环境覆盖，固定版本可复现。"""

    model_id: str = field(
        default_factory=lambda: os.environ.get("MEDRAG_RERANKER_MODEL", DEFAULT_RERANKER_ID)
    )
    top_k: int = 5
    min_score: float = 0.5         # 拒答门：重排后 top 统一分 < 此值 → 拒答
    normalize: bool = True         # True=sigmoid 统一分到 (0,1)；False=用原始 logit


@dataclass
class Reranker:
    """cross-encoder 精排器：重排候选、卡拒答门。`score_fn` 可注入（测试/自定义后端）。"""

    config: RerankConfig = field(default_factory=RerankConfig)
    score_fn: ScoreFn | None = None
    _backend: ScoreFn | None = field(default=None, init=False, repr=False)

    def _scorer(self) -> ScoreFn:
        if self.score_fn is not None:
            return self.score_fn
        if self._backend is None:
            self._backend = self._load_backend()
        return self._backend

    def _load_backend(self) -> ScoreFn:
        """守卫加载 cross-encoder 后端（sentence-transformers CrossEncoder）。"""
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "精排需要 sentence-transformers + reranker 权重（Qwen3-Reranker）；"
                "本地可注入 score_fn 测，功能跑在 AutoDL。"
            ) from exc
        model = CrossEncoder(self.config.model_id, trust_remote_code=True)
        return lambda query, passages: list(
            model.predict([(query, p) for p in passages])
        )

    def rerank(self, result: RetrievalResult) -> RetrievalResult:
        """重排 `result.evidence` → Top-K → 拒答门。上游已拒答/空候选则原样透传。"""
        cfg = self.config
        if result.abstain or not result.evidence:
            return result

        passages = [e.text or e.citation for e in result.evidence]
        raw = list(self._scorer()(result.query, passages))
        if len(raw) != len(result.evidence):
            raise ValueError("score_fn 返回分数个数与候选数不一致")

        scored = []
        for ev, r in zip(result.evidence, raw):
            unified = _sigmoid(float(r)) if cfg.normalize else float(r)
            scored.append((ev, unified))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: cfg.top_k]

        reranked = [ev.model_copy(update={"score": s}) for ev, s in top]
        top_score = top[0][1] if top else float("-inf")
        if top_score < cfg.min_score:
            return RetrievalResult(
                query=result.query, evidence=reranked,
                abstain=True, abstain_reason=AbstainReason.LOW_CONFIDENCE,
            )
        return RetrievalResult(query=result.query, evidence=reranked, abstain=False)


__all__ = [
    "DEFAULT_RERANKER_ID",
    "ScoreFn",
    "RerankConfig",
    "Reranker",
]
