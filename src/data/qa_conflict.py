"""QA 同问不同答冲突判别 + 合并（T040）—— LLM-judge 四档 + LLM-merge，送审前强制去标识。

b 医学QA 库里常见"同一问题、多个来源给了不同答案"。粗暴去重会丢信息、保留全部又自相矛盾。策略
（plan.md）：对同问的不同答用 **LLM-judge 判四档**，再分档处置：
- **EQUIVALENT（等价/复述）** → 留一份，来源并入（溯源不丢）。
- **COMPLEMENTARY（互补）** → **LLM-merge** 成一条（标 `llm_merged`、保双来源、**不加新事实**）。
- **CONFLICTING（矛盾）** → 各留、标 `conflict`，**不替模型编造"正确答案"**（constitution I）。
- **UNRELATED（实为不同问题）** → 各留。

**隐私铁律（FR-007 / [deid]）**：送外部 LLM-judge/merge 前 **MUST 去标识**——本模块对 question/answers
先 `deidentify` + `assert_no_phi` 才外发，PHI 绝不出库。整套冲突处置是**开关**（可消融）。

judge/merge 后端可注入（DashScope LLM 在生产，像 ragas 那样）；编排/分档/去标识纯逻辑本地全测。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from src.data.deid import DeidConfig, assert_no_phi, deidentify
from src.data.normalize import normalize_text


class Verdict(str, Enum):
    EQUIVALENT = "equivalent"
    COMPLEMENTARY = "complementary"
    CONFLICTING = "conflicting"
    UNRELATED = "unrelated"


@dataclass
class QAItem:
    question: str
    answer: str
    source_id: str


@dataclass
class ResolvedQA:
    question: str
    answer: str
    source_ids: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


# 判别/合并后端：输入已**去标识**的 question + answers。
JudgeFn = Callable[[str, list[str]], Verdict]
MergeFn = Callable[[str, list[str]], str]


@dataclass
class QAConflictConfig:
    enabled: bool = True             # 可消融：关掉则原样透传
    deid: bool = True                # 送审前去标识（FR-007）


def group_by_question(items: Sequence[QAItem]) -> "OrderedDict[str, list[QAItem]]":
    """按归一化问题分组（保插入序，可复现）。"""
    groups: "OrderedDict[str, list[QAItem]]" = OrderedDict()
    for it in items:
        key = normalize_text(it.question)
        groups.setdefault(key, []).append(it)
    return groups


def _deid_for_judge(texts: Sequence[str], cfg: QAConflictConfig,
                    deid_config: DeidConfig | None) -> list[str]:
    """去标识 + 外发前硬护栏。"""
    if not cfg.deid:
        return list(texts)
    out = []
    for t in texts:
        clean = deidentify(t, deid_config).text
        assert_no_phi(clean, deid_config)     # 残留 PHI → 抛错，绝不外发
        out.append(clean)
    return out


def resolve_conflicts(
    items: Sequence[QAItem],
    *,
    judge_fn: JudgeFn,
    merge_fn: MergeFn | None = None,
    config: QAConflictConfig | None = None,
    deid_config: DeidConfig | None = None,
) -> list[ResolvedQA]:
    """对同问不同答分档处置，返回处理后的 QA 列表。"""
    cfg = config or QAConflictConfig()
    if not cfg.enabled:                       # 消融关：原样透传
        return [ResolvedQA(it.question, it.answer, [it.source_id]) for it in items]

    resolved: list[ResolvedQA] = []
    for _, members in group_by_question(items).items():
        question = members[0].question
        # 同问下去重答案（归一化相同视为同答），保留代表与其全部来源。
        by_answer: "OrderedDict[str, list[QAItem]]" = OrderedDict()
        for m in members:
            by_answer.setdefault(normalize_text(m.answer), []).append(m)

        if len(by_answer) == 1:               # 唯一答案：留一份，来源并入
            reps = next(iter(by_answer.values()))
            resolved.append(ResolvedQA(
                question, reps[0].answer,
                source_ids=[m.source_id for m in reps],
            ))
            continue

        answers = [reps[0].answer for reps in by_answer.values()]
        all_sources = [m.source_id for reps in by_answer.values() for m in reps]

        deid_q, *deid_answers = _deid_for_judge([question, *answers], cfg, deid_config)
        verdict = judge_fn(deid_q, deid_answers)

        if verdict is Verdict.EQUIVALENT:
            resolved.append(ResolvedQA(question, answers[0], all_sources, flags=["equivalent"]))
        elif verdict is Verdict.COMPLEMENTARY:
            if merge_fn is None:
                raise RuntimeError("COMPLEMENTARY 需要 merge_fn 才能 LLM-merge")
            merged = merge_fn(deid_q, deid_answers)
            resolved.append(ResolvedQA(question, merged, all_sources, flags=["llm_merged"]))
        elif verdict is Verdict.CONFLICTING:
            for reps in by_answer.values():    # 各留、标 conflict，不编造裁决
                resolved.append(ResolvedQA(
                    question, reps[0].answer,
                    source_ids=[m.source_id for m in reps], flags=["conflict"],
                ))
        else:                                  # UNRELATED：各留
            for reps in by_answer.values():
                resolved.append(ResolvedQA(
                    question, reps[0].answer,
                    source_ids=[m.source_id for m in reps],
                ))
    return resolved


__all__ = [
    "Verdict",
    "QAItem",
    "ResolvedQA",
    "JudgeFn",
    "MergeFn",
    "QAConflictConfig",
    "group_by_question",
    "resolve_conflicts",
]
