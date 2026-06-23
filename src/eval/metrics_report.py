"""报告评估指标（T050）—— 域无关 实体 F1 + 关系 F1。

**为什么"域无关"**：医学报告没有现成的统一标签体系，逐病种建模式不可持续。所以走 RadGraph
式思路——用一个 NER/关系抽取器把"生成报告"和"参考报告"各自解析成 (实体, 关系) 图，再比图。
指标只关心**集合重叠**，与具体疾病无关，故可跨数据集复用（constitution III 一致口径）。

**抽取器解耦**：实体/关系抽取依赖 T041 双语医学 NER（`src/data/ner.py`，尚未实现）。本模块
**不内置 NER**，而是吃一个可注入的 `ner_fn` / `rel_fn`，纯 PRF 计算逻辑本地可全量单测；真实
NER 到位后直接注入即可，无需改本模块。缺省未注入时调用 → 抛清晰错（守卫）。

匹配口径：实体 = 字符串或 (type, text) 元组；关系 = (subj, rel, obj) 三元组。默认对文本做
**轻量域无关归一**（小写 + 去首尾空白），可关；按多重集合做 micro-PRF（语料级 TP/FP/FN 累加）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Sequence

# 抽取器签名：文本 → 可哈希项（实体串/元组、关系三元组）的可迭代。
Extractor = Callable[[str], Iterable[Hashable]]


@dataclass
class PRF:
    """精确率/召回率/F1 + 计数（便于 micro 聚合与调试）。"""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1}


def _normalize(item: Hashable, *, lower: bool) -> Hashable:
    """对字符串/元组做域无关归一；非字符串原样返回。"""
    if isinstance(item, str):
        s = item.strip()
        return s.lower() if lower else s
    if isinstance(item, tuple):
        return tuple(_normalize(x, lower=lower) for x in item)
    return item


def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    """由 TP/FP/FN 计 PRF；分母为 0 时按惯例取 0（空预测/空参考边界）。"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def set_prf(
    pred: Iterable[Hashable], ref: Iterable[Hashable], *, lower: bool = True
) -> PRF:
    """多重集合 PRF：TP = 预测与参考的交集计数，FP/FN 为各自多出的计数。

    用 multiset（Counter）而非 set：报告里同一实体出现多次是有意义的。
    """
    pc = Counter(_normalize(x, lower=lower) for x in pred)
    rc = Counter(_normalize(x, lower=lower) for x in ref)
    tp = sum((pc & rc).values())
    fp = sum(pc.values()) - tp
    fn = sum(rc.values()) - tp
    return prf_from_counts(tp, fp, fn)


def _require(fn: Extractor | None, name: str) -> Extractor:
    if fn is None:
        raise RuntimeError(
            f"{name} 未注入：报告实体/关系指标依赖 T041 NER（src/data/ner.py）。"
            f"请传入 {name}=可调用(text)->实体/关系 的抽取器。"
        )
    return fn


def _corpus_prf(
    pairs: Sequence[tuple[str, str]], extract: Extractor, *, lower: bool
) -> PRF:
    """语料级 micro-PRF：逐对 (pred_text, ref_text) 抽取后累加 TP/FP/FN。"""
    tp = fp = fn = 0
    for pred_text, ref_text in pairs:
        r = set_prf(extract(pred_text), extract(ref_text), lower=lower)
        tp, fp, fn = tp + r.tp, fp + r.fp, fn + r.fn
    return prf_from_counts(tp, fp, fn)


def report_entity_f1(
    pairs: Sequence[tuple[str, str]],
    *,
    ner_fn: Extractor | None = None,
    lower: bool = True,
) -> PRF:
    """域无关实体 F1（micro）。`pairs` 为 (生成报告, 参考报告) 文本对。"""
    return _corpus_prf(pairs, _require(ner_fn, "ner_fn"), lower=lower)


def report_relation_f1(
    pairs: Sequence[tuple[str, str]],
    *,
    rel_fn: Extractor | None = None,
    lower: bool = True,
) -> PRF:
    """域无关关系 F1（micro）。`rel_fn(text)` 返回 (subj, rel, obj) 三元组的可迭代。"""
    return _corpus_prf(pairs, _require(rel_fn, "rel_fn"), lower=lower)


def report_metrics(
    pairs: Sequence[tuple[str, str]],
    *,
    ner_fn: Extractor | None = None,
    rel_fn: Extractor | None = None,
    lower: bool = True,
) -> dict:
    """一站式：实体 F1 +（可选）关系 F1，整理成 [schemas.EvalRecord].metrics 形态。"""
    out: dict[str, float] = {}
    ent = report_entity_f1(pairs, ner_fn=ner_fn, lower=lower)
    out.update({f"entity_{k}": v for k, v in ent.as_dict().items()})
    if rel_fn is not None:
        rel = report_relation_f1(pairs, rel_fn=rel_fn, lower=lower)
        out.update({f"relation_{k}": v for k, v in rel.as_dict().items()})
    return {"metrics": out}


__all__ = [
    "Extractor",
    "PRF",
    "prf_from_counts",
    "set_prf",
    "report_entity_f1",
    "report_relation_f1",
    "report_metrics",
]
