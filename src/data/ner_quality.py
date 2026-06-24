"""NER 覆盖率 + 多信号质量筛选（T042）—— 零实体硬丢、低覆盖软降权 + 覆盖护栏。

RAG 库不是越大越好——满是泛泛叙述、零医学实体的片段会稀释检索。用 NER 抽出的实体当**质量信号**筛：
- **密度**：实体数 / token 数（信息浓度）。
- **可链接率**：能挂到 UMLS（有 `cui`）的实体比例（越规范越可信）。
- **稀有命中**：命中低频实体的比例（稀有≈有判别力；df 基准**须排除评估集**，防泄露）。

策略（plan.md）：**零实体硬丢**；**低覆盖软降权**（不删、降 `quality_score` + 标 `low_coverage`）。

**覆盖护栏（FR-012）**：去重/筛选**不得删掉任何独有实体**——筛完比对前后实体全集，丢失的独有实体
要能被 `coverage_guard` 抓出来（断言级）。纯逻辑本地全测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.contracts.schemas import KnowledgeNode
from src.data.ner import Entity
from src.rag.chunk import estimate_tokens

# 一篇文档的标注：节点 + 它的实体。
Annotated = tuple[KnowledgeNode, Sequence[Entity]]


@dataclass
class QualityConfig:
    min_entities: int = 1            # < 此值（即 0 实体）→ 硬丢
    low_density: float = 0.02        # 密度低于此 → 低覆盖软降权
    low_coverage_penalty: float = 0.5
    rare_df_ratio: float = 0.05      # df 占比 ≤ 此 → 视为稀有实体
    w_density: float = 0.4
    w_linkable: float = 0.3
    w_rare: float = 0.3


@dataclass
class QualitySignals:
    n_entities: int
    density: float
    linkable_rate: float
    rare_rate: float
    score: float
    low_coverage: bool


@dataclass
class FilterResult:
    kept: list[KnowledgeNode]
    dropped_ids: list[str] = field(default_factory=list)
    signals: dict[str, QualitySignals] = field(default_factory=dict)  # node_id -> signals


def document_frequency(
    annotated: Sequence[Annotated], *, eval_mask: Sequence[bool] | None = None
) -> dict[str, int]:
    """实体文档频率（df）。`eval_mask[i]=True` 的文档**排除**在外（基准不含评估集，防泄露）。"""
    df: dict[str, int] = {}
    for i, (_, ents) in enumerate(annotated):
        if eval_mask is not None and i < len(eval_mask) and eval_mask[i]:
            continue
        for norm in {e.norm for e in ents}:
            df[norm] = df.get(norm, 0) + 1
    return df


def compute_signals(
    text: str,
    entities: Sequence[Entity],
    *,
    df: dict[str, int] | None = None,
    n_docs: int = 1,
    config: QualityConfig | None = None,
) -> QualitySignals:
    """算单文档的多信号质量分。"""
    cfg = config or QualityConfig()
    n = len(entities)
    toks = max(estimate_tokens(text), 1)
    density = n / toks
    linkable = (sum(1 for e in entities if e.cui) / n) if n else 0.0
    if df and n:
        rare = sum(
            1 for e in entities if df.get(e.norm, 0) / max(n_docs, 1) <= cfg.rare_df_ratio
        ) / n
    else:
        rare = 0.0

    base = (
        cfg.w_density * min(density / cfg.low_density, 1.0)
        + cfg.w_linkable * linkable
        + cfg.w_rare * rare
    )
    low = density < cfg.low_density
    score = base * (cfg.low_coverage_penalty if low else 1.0)
    return QualitySignals(n_entities=n, density=density, linkable_rate=linkable,
                          rare_rate=rare, score=score, low_coverage=low)


def filter_nodes(
    annotated: Sequence[Annotated],
    *,
    df: dict[str, int] | None = None,
    n_docs: int = 1,
    config: QualityConfig | None = None,
) -> FilterResult:
    """零实体硬丢、其余打质量分；低覆盖软降权（标 `low_coverage`、不删）。

    会把 `quality_score` 与实体写回 KnowledgeNode（供入库/溯源）。
    """
    cfg = config or QualityConfig()
    kept: list[KnowledgeNode] = []
    dropped: list[str] = []
    signals: dict[str, QualitySignals] = {}
    for node, ents in annotated:
        if len(ents) < cfg.min_entities:        # 零实体硬丢
            dropped.append(node.node_id)
            continue
        sig = compute_signals(node.text, ents, df=df, n_docs=n_docs, config=cfg)
        node.quality_score = sig.score
        node.entities = [e.norm for e in ents]
        if sig.low_coverage and "low_coverage" not in node.flags:
            node.flags = [*node.flags, "low_coverage"]
        signals[node.node_id] = sig
        kept.append(node)
    return FilterResult(kept=kept, dropped_ids=dropped, signals=signals)


def collect_entities(annotated: Sequence[Annotated]) -> set[str]:
    return {e.norm for _, ents in annotated for e in ents}


def coverage_guard(
    before: Sequence[Annotated], after_nodes: Sequence[KnowledgeNode]
) -> set[str]:
    """去重/筛选前后实体覆盖比对，返回**丢失的独有实体**（FR-012：理想为空集）。"""
    before_set = collect_entities(before)
    after_set: set[str] = set()
    for n in after_nodes:
        after_set.update(n.entities)
    return before_set - after_set


def assert_coverage(before: Sequence[Annotated], after_nodes: Sequence[KnowledgeNode]) -> None:
    """断言无独有实体丢失，否则抛错（质量门可挂此护栏）。"""
    lost = coverage_guard(before, after_nodes)
    if lost:
        raise ValueError(f"覆盖护栏失败：去重/筛选丢失独有实体 {sorted(lost)}（违反 FR-012）")


__all__ = [
    "Annotated",
    "QualityConfig",
    "QualitySignals",
    "FilterResult",
    "document_frequency",
    "compute_signals",
    "filter_nodes",
    "collect_entities",
    "coverage_guard",
    "assert_coverage",
]
