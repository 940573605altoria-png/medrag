"""端到端评估指标（T052）—— 证据可溯源率 + 拒答正确性。

**为什么这两个是端到端命题的总分**：本项目卖点是"不编造、可溯源、不会就拒答"。前面 detection
/report/rag 各管一段,这里量两件全局事:
- **证据可溯源率**:每条非"不确定"结论是否真挂了证据或 ROI(constitution I)。理想 = 1.0;
  低于 1 说明系统在无据时仍下结论(编造泄漏)。
- **拒答正确性**:该拒答时拒答、不该拒答时别乱拒(FR-003)。用二值混淆给 abstention P/R/F1,
  既抓"该拒不拒"(漏报,危险),也抓"过度拒答"(召回好但没用)。

纯逻辑,无模型;复用 [metrics_report.prf_from_counts] 保持 PRF 口径一致。
"""

from __future__ import annotations

from typing import Sequence

from src.contracts.schemas import Finding, ReportResult
from src.eval.metrics_report import PRF, prf_from_counts


def _is_grounded(f: Finding) -> bool:
    """结论是否锚到证据或 ROI（schema 已禁止非 uncertain 结论无据，这里也计 uncertain 的实际锚定）。"""
    return bool(f.evidence) or f.roi is not None


def traceability_rate(reports: Sequence[ReportResult]) -> dict[str, float]:
    """证据可溯源率 = 已锚定结论数 / 全部结论数（含 uncertain）。

    另给 `grounded_excl_uncertain`(只看断言性结论,理论上应=1.0,<1 即编造泄漏)与平均证据条数。
    """
    total = grounded = 0
    total_assert = grounded_assert = 0
    ev_count = 0
    for r in reports:
        for f in r.findings:
            total += 1
            g = _is_grounded(f)
            grounded += int(g)
            ev_count += len(f.evidence)
            if not f.uncertain:
                total_assert += 1
                grounded_assert += int(g)
    return {
        "traceability_rate": grounded / total if total else 1.0,
        "grounded_excl_uncertain": grounded_assert / total_assert if total_assert else 1.0,
        "avg_evidence_per_finding": ev_count / total if total else 0.0,
        "n_findings": float(total),
    }


def abstention_metrics(
    pred_abstain: Sequence[bool], gold_abstain: Sequence[bool]
) -> dict[str, float]:
    """拒答正确性：以"该拒答"为正类的二值 P/R/F1 + 准确率。

    TP=该拒且拒, FP=不该拒却拒(过度拒答), FN=该拒却答(危险漏拒)。
    """
    p_arr, g_arr = list(pred_abstain), list(gold_abstain)
    if len(p_arr) != len(g_arr):
        raise ValueError(f"pred/gold 数量不一致：{len(p_arr)} vs {len(g_arr)}")
    if not p_arr:
        raise ValueError("abstention_metrics 需要非空样本")
    tp = sum(1 for p, g in zip(p_arr, g_arr) if p and g)
    fp = sum(1 for p, g in zip(p_arr, g_arr) if p and not g)
    fn = sum(1 for p, g in zip(p_arr, g_arr) if not p and g)
    tn = sum(1 for p, g in zip(p_arr, g_arr) if not p and not g)
    prf: PRF = prf_from_counts(tp, fp, fn)
    out = {f"abstain_{k}": v for k, v in prf.as_dict().items()}
    out["abstain_accuracy"] = (tp + tn) / len(p_arr)
    return out


def e2e_metrics(
    reports: Sequence[ReportResult],
    *,
    gold_abstain: Sequence[bool] | None = None,
) -> dict:
    """一站式：溯源率 +（给了金标准则）拒答正确性，整理成 EvalRecord.metrics 形态。"""
    metrics = dict(traceability_rate(reports))
    if gold_abstain is not None:
        pred_abstain = [r.abstain for r in reports]
        metrics.update(abstention_metrics(pred_abstain, gold_abstain))
    return {"metrics": metrics}


__all__ = [
    "traceability_rate",
    "abstention_metrics",
    "e2e_metrics",
]
