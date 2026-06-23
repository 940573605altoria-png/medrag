"""RAG 评估指标（T051）—— 检索 recall@k/nDCG/MRR（纯逻辑）+ ragas faithfulness/context（LLM 裁判）。

分两层,与本项目"重依赖隔离"方法论一致:
1. **检索排序指标(纯 numpy,本地全测)**:recall@k、nDCG@k、MRR——回答"相关证据有没有被召回、
   排得够不够靠前"。输入是排好序的来源 id 列表 + 相关 id 集合(金标准)。
2. **ragas 生成质量指标(LLM 裁判,功能跑)**:faithfulness(答案有没有超出上下文 → 直接量化"禁
   编造")、context_precision/recall。裁判走 **DashScope/qwen-max,temperature=0**(对齐 T051
   "固定 judge temp0",可复现)。

**裁判模型与 key 全部从配置/环境读,绝不写死**:
- `DASHSCOPE_API_KEY`:百炼 key(必填,功能跑时)。**不进代码、不进仓库。**
- `MEDRAG_RAGAS_JUDGE_MODEL`:默认 `qwen-max`(文本旗舰);多模态判图证据时可换 `qwen3-vl-plus`。
- `MEDRAG_RAGAS_EMBED_MODEL`:默认 `text-embedding-v3`。
ragas/langchain 守卫导入:未安装也能 import 本模块,只有调用 `ragas_scores` 才需要。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# DashScope OpenAI 兼容端点（国内直连，无需代理）。
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_JUDGE_MODEL = "qwen-max"
DEFAULT_EMBED_MODEL = "text-embedding-v3"


# ════════════════════════════════════════════════════════════════════
# 1) 检索排序指标（纯逻辑）
# ════════════════════════════════════════════════════════════════════

def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """前 k 个命中的相关项 / 全部相关项。relevant 为空 → 1.0（无可召回，不罚）。"""
    if not relevant:
        return 1.0
    hit = sum(1 for r in retrieved[:k] if r in relevant)
    return hit / len(relevant)


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """首个相关项的倒数排名；前面都不相关 → 0。"""
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """二值相关性 nDCG@k：DCG / 理想 DCG。relevant 为空 → 1.0。"""
    if not relevant:
        return 1.0
    dcg = sum(
        1.0 / math.log2(i + 2) for i, r in enumerate(retrieved[:k]) if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(
    per_query: Sequence[tuple[Sequence[str], set[str]]],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> dict:
    """对多条查询的 (检索 id 列表, 相关 id 集) 求平均 recall@k / nDCG@k / MRR。"""
    if not per_query:
        raise ValueError("retrieval_metrics 需要至少一条查询")
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = float(
            np.mean([recall_at_k(r, rel, k) for r, rel in per_query])
        )
        metrics[f"ndcg@{k}"] = float(
            np.mean([ndcg_at_k(r, rel, k) for r, rel in per_query])
        )
    metrics["mrr"] = float(np.mean([mrr(r, rel) for r, rel in per_query]))
    return {"metrics": metrics}


# ════════════════════════════════════════════════════════════════════
# 2) ragas 生成质量（LLM 裁判，功能跑；守卫导入）
# ════════════════════════════════════════════════════════════════════

@dataclass
class RagasSample:
    """一条 ragas 评测样本（字段名对齐 ragas 的 SingleTurnSample）。"""

    user_input: str                       # 问题
    response: str                         # 系统答案
    retrieved_contexts: list[str]         # 检索到的上下文片段
    reference: str | None = None          # 参考答案（context_recall 需要）


@dataclass
class RagasConfig:
    """裁判配置——全部可被环境变量覆盖，key 永不入参默认值。"""

    judge_model: str = field(
        default_factory=lambda: os.environ.get("MEDRAG_RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    )
    embed_model: str = field(
        default_factory=lambda: os.environ.get("MEDRAG_RAGAS_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    )
    base_url: str = DASHSCOPE_BASE_URL
    temperature: float = 0.0  # T051：固定 judge temp0，可复现

    def api_key(self) -> str:
        key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            raise RuntimeError(
                "缺少 DASHSCOPE_API_KEY 环境变量——ragas 裁判需要百炼 key。"
                "请 `setx DASHSCOPE_API_KEY <key>`（勿写进代码/仓库）。"
            )
        return key


def build_judge(cfg: RagasConfig | None = None):
    """构造 ragas 用的 LLM 裁判 + embedding（DashScope，OpenAI 兼容）。守卫导入。"""
    cfg = cfg or RagasConfig()
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: PLC0415
        from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
        from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "ragas 裁判需要 ragas + langchain-openai；请 `pip install ragas langchain-openai`。"
        ) from exc

    key = cfg.api_key()
    llm = ChatOpenAI(
        model=cfg.judge_model, temperature=cfg.temperature,
        base_url=cfg.base_url, api_key=key,
    )
    emb = OpenAIEmbeddings(model=cfg.embed_model, base_url=cfg.base_url, api_key=key)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def ragas_scores(
    samples: Sequence[RagasSample],
    *,
    cfg: RagasConfig | None = None,
    with_reference: bool = False,
) -> dict:
    """跑 ragas：faithfulness + context_precision（+ 有 reference 时 context_recall）。

    功能性（联网调 DashScope）。返回 EvalRecord.metrics 形态。
    """
    try:
        from ragas import EvaluationDataset, evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
        )
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError("ragas 未安装；请 `pip install ragas`。") from exc

    llm, emb = build_judge(cfg)
    rows = [
        {
            "user_input": s.user_input,
            "response": s.response,
            "retrieved_contexts": s.retrieved_contexts,
            **({"reference": s.reference} if s.reference is not None else {}),
        }
        for s in samples
    ]
    dataset = EvaluationDataset.from_list(rows)
    metrics = [Faithfulness(), LLMContextPrecisionWithoutReference()]
    if with_reference:
        metrics.append(LLMContextRecall())
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb)
    return {"metrics": {k: float(v) for k, v in result._repr_dict.items()}}


__all__ = [
    "DASHSCOPE_BASE_URL",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_EMBED_MODEL",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "retrieval_metrics",
    "RagasSample",
    "RagasConfig",
    "build_judge",
    "ragas_scores",
]
