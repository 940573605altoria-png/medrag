"""a/b/c 原始数据 ingestion 加载器 + 文本清洗管线（T008）。

把分散的数据预处理步骤（[normalize] → [deid] → [dedup] →（b）[qa_conflict] → [ner]+[ner_quality]）
**串成一条可配置、可消融的管线**（plan.md 顺序：ingestion → 去标识 → 去重 → NER 覆盖率筛 → 切块 →
embedding → 入库；本模块负责到"NER 覆盖率筛"为止，产出干净 `KnowledgeNode` 交给 [chunk]/[index_text]）。

设计要点：
- **每步开关**（`IngestConfig`，constitution III 可消融）；NER/judge/merge 后端**可注入**，故管线本地可测。
- **隐私前置**（FR-007）：清洗即 deidentify，PHI 不入后续任何环节。
- **覆盖护栏**（FR-012）：全程比对清洗后实体全集 vs 最终留存，`coverage_lost` 抓被去重/筛选误删的独有实体。

raw 加载器：jsonl 通用读 + a/b/c 适配（c → `CTSample`，视觉侧由 coreset 链处理，文本报告可走本管线）。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.contracts.schemas import CTSample, KnowledgeNode
from src.data.deid import DeidConfig, deidentify
from src.data.dedup import exact_dedup, near_dedup, structured_dedup
from src.data.ner_quality import QualityConfig, coverage_guard, filter_nodes
from src.data.normalize import NormalizeConfig, normalize_text
from src.data.qa_conflict import (
    JudgeFn,
    MergeFn,
    QAConflictConfig,
    QAItem,
    ResolvedQA,
    resolve_conflicts,
)


# ════════════════════════════════════════════════════════════════════
# raw 加载器
# ════════════════════════════════════════════════════════════════════

@dataclass
class RawRecord:
    record_id: str
    text: str = ""
    question: str | None = None
    answer: str | None = None
    lang: str | None = None
    metadata: dict = field(default_factory=dict)


def load_jsonl(
    path: str | Path,
    *,
    id_field: str = "id",
    text_field: str = "text",
    question_field: str = "question",
    answer_field: str = "answer",
) -> list[RawRecord]:
    """读 jsonl → RawRecord 列表（缺字段宽容处理；其余字段并入 metadata）。"""
    records: list[RawRecord] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        known = {id_field, text_field, question_field, answer_field, "lang"}
        records.append(RawRecord(
            record_id=str(obj.get(id_field, i)),
            text=obj.get(text_field, "") or "",
            question=obj.get(question_field),
            answer=obj.get(answer_field),
            lang=obj.get("lang"),
            metadata={k: v for k, v in obj.items() if k not in known},
        ))
    return records


def drug_records_to_nodes(
    records: Sequence[RawRecord], *, collection: str = "a_drug"
) -> list[KnowledgeNode]:
    """a 药品：RawRecord → KnowledgeNode（结构化键 drug/section 入 metadata 供结构化去重）。"""
    return [
        KnowledgeNode(node_id=r.record_id, text=r.text, collection=collection,
                      lang=r.lang, source_ids=[r.record_id], metadata=dict(r.metadata))
        for r in records
    ]


def qa_records_to_items(records: Sequence[RawRecord]) -> list[QAItem]:
    """b 医学QA：RawRecord → QAItem（缺 question/answer 的跳过）。"""
    items: list[QAItem] = []
    for r in records:
        if r.question and r.answer:
            items.append(QAItem(question=r.question, answer=r.answer, source_id=r.record_id))
    return items


def ct_records_to_samples(records: Sequence[RawRecord]) -> list[CTSample]:
    """c CT-QA：RawRecord → CTSample（图像路径/报告文本在 metadata；ROI 由视觉链解析）。"""
    return [
        CTSample(case_id=r.record_id,
                 image_path=str(r.metadata.get("image_path", "")),
                 report_text=r.answer or r.text or None,
                 metadata=dict(r.metadata))
        for r in records
    ]


# ════════════════════════════════════════════════════════════════════
# 文本清洗管线
# ════════════════════════════════════════════════════════════════════

@dataclass
class IngestConfig:
    normalize: bool = True
    deidentify: bool = True
    exact_dedup: bool = True
    near_dedup: bool = False
    near_threshold: float = 0.85
    structured_key_fields: tuple[str, ...] = ()     # a 用 ("drug","section")
    qa_conflict: bool = False                       # b 用
    ner_filter: bool = True
    norm_config: NormalizeConfig | None = None
    deid_config: DeidConfig | None = None
    quality_config: QualityConfig | None = None


@dataclass
class IngestStats:
    n_in: int
    n_after_dedup: int
    n_dropped_quality: int
    n_out: int
    phi_found: int


@dataclass
class IngestResult:
    nodes: list[KnowledgeNode]
    dropped_ids: list[str] = field(default_factory=list)
    stats: IngestStats | None = None
    coverage_lost: set[str] = field(default_factory=set)   # FR-012：应为空


def _clean_text(text: str, cfg: IngestConfig) -> tuple[str, int]:
    """归一化 + 去标识（PHI 前置）。返回 (clean, phi_count)。"""
    phi = 0
    if cfg.normalize:
        text = normalize_text(text, cfg.norm_config)
    if cfg.deidentify:
        res = deidentify(text, cfg.deid_config)
        text, phi = res.text, len(res.found)
    return text, phi


def _dedup_and_filter(
    nodes: Sequence[KnowledgeNode], cfg: IngestConfig, ner: Any
) -> tuple[list[KnowledgeNode], list[str], int, set[str]]:
    """去重 → NER 覆盖率/质量筛；带全程覆盖护栏。"""
    deduped = list(nodes)
    if cfg.structured_key_fields:
        deduped = structured_dedup(deduped, cfg.structured_key_fields).kept
    if cfg.exact_dedup:
        deduped = exact_dedup(deduped, norm_config=cfg.norm_config).kept
    if cfg.near_dedup:
        deduped = near_dedup(deduped, threshold=cfg.near_threshold,
                             norm_config=cfg.norm_config).kept

    kept, dropped, lost = deduped, [], set()
    if cfg.ner_filter and ner is not None:
        before = [(n, ner.extract(n.text)) for n in nodes]        # 清洗后全集实体（护栏基准）
        annotated_kept = [(n, ner.extract(n.text)) for n in deduped]
        fres = filter_nodes(annotated_kept, config=cfg.quality_config)
        kept, dropped = fres.kept, fres.dropped_ids
        lost = coverage_guard(before, kept)
    return kept, dropped, len(deduped), lost


def ingest_documents(
    nodes: Sequence[KnowledgeNode],
    *,
    config: IngestConfig | None = None,
    ner: Any = None,
) -> IngestResult:
    """a/通用文本管线：清洗 → 去重 → NER 质量筛。原地清洗节点 text。"""
    cfg = config or IngestConfig()
    phi_total = 0
    for n in nodes:
        n.text, p = _clean_text(n.text, cfg)
        phi_total += p
    kept, dropped, n_dedup, lost = _dedup_and_filter(nodes, cfg, ner)
    return IngestResult(
        nodes=kept, dropped_ids=dropped, coverage_lost=lost,
        stats=IngestStats(len(nodes), n_dedup, len(dropped), len(kept), phi_total),
    )


def ingest_qa(
    items: Sequence[QAItem],
    *,
    config: IngestConfig | None = None,
    judge_fn: JudgeFn | None = None,
    merge_fn: MergeFn | None = None,
    ner: Any = None,
    collection: str = "b_medqa",
) -> IngestResult:
    """b 医学QA 管线：清洗 q/a → QA 冲突分档 → 建 node → 去重 → NER 质量筛。"""
    cfg = config or IngestConfig()
    phi_total = 0
    cleaned: list[QAItem] = []
    for it in items:
        q, p1 = _clean_text(it.question, cfg)
        a, p2 = _clean_text(it.answer, cfg)
        phi_total += p1 + p2
        cleaned.append(QAItem(q, a, it.source_id))

    if cfg.qa_conflict and judge_fn is not None:
        resolved = resolve_conflicts(
            cleaned, judge_fn=judge_fn, merge_fn=merge_fn,
            config=QAConflictConfig(enabled=True, deid=False),  # 已 deid，避免重复
            deid_config=cfg.deid_config,
        )
    else:
        resolved = [ResolvedQA(it.question, it.answer, [it.source_id]) for it in cleaned]

    nodes = [
        KnowledgeNode(
            node_id=f"qa-{uuid.uuid4().hex[:12]}", text=r.answer, collection=collection,
            source_ids=list(r.source_ids), flags=list(r.flags),
            metadata={"question": r.question},
        )
        for r in resolved
    ]
    kept, dropped, n_dedup, lost = _dedup_and_filter(nodes, cfg, ner)
    return IngestResult(
        nodes=kept, dropped_ids=dropped, coverage_lost=lost,
        stats=IngestStats(len(items), n_dedup, len(dropped), len(kept), phi_total),
    )


__all__ = [
    "RawRecord",
    "load_jsonl",
    "drug_records_to_nodes",
    "qa_records_to_items",
    "ct_records_to_samples",
    "IngestConfig",
    "IngestStats",
    "IngestResult",
    "ingest_documents",
    "ingest_qa",
]
