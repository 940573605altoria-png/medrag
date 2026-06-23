"""评估指标单测（T049/T050）—— 纯逻辑、CPU、无 torch。

用合成的、TP/FP 已知的 DetectionResult 与文本对手算对拍 FROC/sensitivity@FP/mAP 与实体F1，
守住"创新增益"那把尺子的口径不漂（constitution III）。
"""

from __future__ import annotations

import math

import pytest

from src.contracts.schemas import (
    AbstainReason,
    AreaBand,
    BBox,
    DetectionResult,
    EvidenceItem,
    Finding,
    ReportResult,
    ROI,
)
from src.eval import metrics_detection as md
from src.eval import metrics_e2e as me
from src.eval import metrics_rag as mrag
from src.eval import metrics_report as mr
from src.eval import stats as st


def _roi(x1, y1, x2, y2, *, frac=0.01, conf=1.0):
    return ROI(bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), area_fraction=frac, confidence=conf)


# ── 检测：IoU 与匹配 ────────────────────────────────────────────────

def test_bbox_iou_identical_disjoint_and_half():
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    assert md.bbox_iou(a, a) == pytest.approx(1.0)
    assert md.bbox_iou(a, BBox(x1=100, y1=100, x2=110, y2=110)) == 0.0
    # 交集 5x10=50，并集 100+100-50=150 → 1/3
    half = BBox(x1=5, y1=0, x2=15, y2=10)
    assert md.bbox_iou(a, half) == pytest.approx(50 / 150)


def test_match_image_tp_and_fp():
    gt = [_roi(0, 0, 10, 10)]
    preds = [_roi(0, 0, 10, 10, conf=0.9), _roi(100, 100, 110, 110, conf=0.8)]
    flags = md.match_image(preds, gt, iou_thr=0.3)
    assert [f.is_tp for f in flags] == [True, False]  # 降序：先 TP 再 FP


def test_match_image_each_gt_taken_once():
    gt = [_roi(0, 0, 10, 10)]
    # 两条预测都命中同一 GT → 只有最高置信那条算 TP
    preds = [_roi(0, 0, 10, 10, conf=0.9), _roi(1, 1, 11, 11, conf=0.7)]
    flags = md.match_image(preds, gt, iou_thr=0.3)
    assert [f.is_tp for f in flags] == [True, False]


# ── 检测：FROC / sensitivity@FP / mAP（手算对拍）────────────────────
# 场景：2 张图、2 个 GT。排序后预测为 TP(0.9), FP(0.7), TP(0.5)。
#   A 图 GT 小病灶(<2%)，B 图 GT 大病灶(>5%)。

def _froc_scenario():
    img_a = DetectionResult(rois=[_roi(0, 0, 10, 10, conf=0.9),
                                  _roi(100, 100, 110, 110, conf=0.7)])
    img_b = DetectionResult(rois=[_roi(1, 1, 11, 11, conf=0.5)])
    gt_a = [_roi(0, 0, 10, 10, frac=0.01)]   # SMALL
    gt_b = [_roi(0, 0, 10, 10, frac=0.10)]   # LARGE
    return [img_a, img_b], [gt_a, gt_b]


def test_froc_curve_points():
    preds, gts = _froc_scenario()
    curve = md.froc_curve(preds, gts, iou_thr=0.3)
    assert curve.n_images == 2
    assert curve.total_gt == 2
    # 累积点：起点 + 每条预测 → fp/img 与 sensitivity
    assert list(curve.fp_per_image) == pytest.approx([0.0, 0.0, 0.5, 0.5])
    assert list(curve.sensitivity) == pytest.approx([0.0, 0.5, 0.5, 1.0])


def test_sensitivity_at_fp():
    preds, gts = _froc_scenario()
    curve = md.froc_curve(preds, gts, iou_thr=0.3)
    s = md.sensitivity_at_fp(curve, fp_points=[0.0, 0.5, 2.0])
    assert s[0.0] == pytest.approx(0.5)   # 只含首条 TP
    assert s[0.5] == pytest.approx(1.0)   # 含全部
    assert s[2.0] == pytest.approx(1.0)


def test_sensitivity_at_fp_stratified():
    preds, gts = _froc_scenario()
    curve = md.froc_curve(preds, gts, iou_thr=0.3)
    by = md.sensitivity_at_fp_by_band(curve, fp_points=[0.0, 0.5])
    # fp=0 仅含 A 图那条 TP（命中 SMALL）：SMALL=1, LARGE=0
    assert by[AreaBand.SMALL][0.0] == pytest.approx(1.0)
    assert by[AreaBand.LARGE][0.0] == pytest.approx(0.0)
    # fp=0.5 含全部：两带都召回
    assert by[AreaBand.SMALL][0.5] == pytest.approx(1.0)
    assert by[AreaBand.LARGE][0.5] == pytest.approx(1.0)
    # MEDIUM 无 GT → 恒 0
    assert by[AreaBand.MEDIUM][0.5] == pytest.approx(0.0)


def test_average_precision_known_value():
    preds, gts = _froc_scenario()
    # VOC all-point AP：rec=[.5,.5,1] prec=[1,.5,.667] → 0.5*1 + 0.5*0.667 ≈ 0.8333
    ap = md.average_precision(preds, gts, iou_thr=0.3)
    assert ap == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3), abs=1e-6)


def test_average_precision_no_gt_is_zero():
    preds = [DetectionResult(rois=[_roi(0, 0, 10, 10, conf=0.9)])]
    assert md.average_precision(preds, [[]], iou_thr=0.3) == 0.0


def test_detection_metrics_shape():
    preds, gts = _froc_scenario()
    out = md.detection_metrics(preds, gts, iou_thr=0.3, fp_points=[0.5])
    assert "mAP" in out["metrics"]
    assert "sens@0.5fp" in out["metrics"]
    assert set(out["stratified"]) == {b.value for b in AreaBand}


def test_froc_length_mismatch_raises():
    with pytest.raises(ValueError):
        md.froc_curve([DetectionResult()], [[], []])


# ── 报告：实体 F1 / 关系 F1 ─────────────────────────────────────────

def _words(text: str):
    return text.split()


def test_set_prf_exact_partial_empty():
    # 完全匹配
    r = mr.set_prf(["liver", "mass"], ["liver", "mass"])
    assert (r.precision, r.recall, r.f1) == pytest.approx((1.0, 1.0, 1.0))
    # 部分匹配：liver 命中，lung 误报，mass 漏报 → P=R=F1=0.5
    r = mr.set_prf(["liver", "lung"], ["liver", "mass"])
    assert (r.precision, r.recall, r.f1) == pytest.approx((0.5, 0.5, 0.5))
    # 空预测：tp=fp=0 → P=0；fn=1 → R=0
    r = mr.set_prf([], ["liver"])
    assert (r.precision, r.recall, r.f1) == (0.0, 0.0, 0.0)


def test_set_prf_multiset_and_normalization():
    # 多重集合：pred 出现两次、ref 一次 → tp=1, fp=1
    r = mr.set_prf(["liver", "liver"], ["liver"])
    assert r.precision == pytest.approx(0.5)
    assert r.recall == pytest.approx(1.0)
    # 域无关归一：大小写不敏感
    r = mr.set_prf(["Liver"], ["liver"])
    assert r.f1 == pytest.approx(1.0)


def test_report_entity_f1_corpus_micro():
    pairs = [("liver mass", "liver mass"), ("lung", "liver")]
    r = mr.report_entity_f1(pairs, ner_fn=_words)
    # 累积：tp=2(liver,mass), fp=1(lung), fn=1(liver in 2nd) → P=R=2/3
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == pytest.approx(2 / 3)


def test_report_relation_f1():
    rel_fn = lambda t: [tuple(line.split("-")) for line in t.split(";") if line]
    pairs = [("liver-has-mass", "liver-has-mass;lung-has-nodule")]
    r = mr.report_relation_f1(pairs, rel_fn=rel_fn)
    # tp=1, fp=0, fn=1 → P=1, R=0.5
    assert r.precision == pytest.approx(1.0)
    assert r.recall == pytest.approx(0.5)


def test_extractor_not_injected_raises():
    with pytest.raises(RuntimeError):
        mr.report_entity_f1([("a", "b")])
    with pytest.raises(RuntimeError):
        mr.report_relation_f1([("a", "b")])


def test_report_metrics_optional_relation():
    pairs = [("liver", "liver")]
    out = mr.report_metrics(pairs, ner_fn=_words)
    assert "entity_f1" in out["metrics"]
    assert not any(k.startswith("relation_") for k in out["metrics"])
    out2 = mr.report_metrics(pairs, ner_fn=_words, rel_fn=lambda t: [])
    assert "relation_f1" in out2["metrics"]


# ── RAG 检索指标（T051，手算对拍）──────────────────────────────────

def test_recall_mrr_ndcg_known():
    retrieved = ["d1", "d2", "d3", "d4"]
    rel = {"d2", "d4"}
    assert mrag.recall_at_k(retrieved, rel, 2) == pytest.approx(0.5)
    assert mrag.recall_at_k(retrieved, rel, 4) == pytest.approx(1.0)
    assert mrag.mrr(retrieved, rel) == pytest.approx(0.5)  # 首个相关在 rank2
    # nDCG@4: dcg=1/log2(3)+1/log2(5), idcg=1/log2(2)+1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert mrag.ndcg_at_k(retrieved, rel, 4) == pytest.approx(dcg / idcg)


def test_retrieval_metrics_empty_relevant_is_one():
    assert mrag.recall_at_k(["a"], set(), 1) == 1.0
    assert mrag.ndcg_at_k(["a"], set(), 1) == 1.0


def test_retrieval_metrics_aggregate():
    per_query = [(["d1", "d2"], {"d2"}), (["d3", "d4"], {"d3"})]
    out = mrag.retrieval_metrics(per_query, ks=[1, 2])
    # recall@1: q1 d1 miss→0, q2 d3 hit→1 → 平均 0.5
    assert out["metrics"]["recall@1"] == pytest.approx(0.5)
    assert out["metrics"]["recall@2"] == pytest.approx(1.0)


def test_ragas_config_reads_env_and_requires_key(monkeypatch):
    monkeypatch.delenv("MEDRAG_RAGAS_JUDGE_MODEL", raising=False)
    assert mrag.RagasConfig().judge_model == mrag.DEFAULT_JUDGE_MODEL  # qwen-max
    monkeypatch.setenv("MEDRAG_RAGAS_JUDGE_MODEL", "qwen3-max")
    assert mrag.RagasConfig().judge_model == "qwen3-max"
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        mrag.RagasConfig().api_key()


# ── 端到端指标（T052）──────────────────────────────────────────────

def _ev():
    return EvidenceItem(source_id="s1", citation="c", score=0.9)


def test_traceability_rate():
    grounded = Finding(text="肝占位", evidence=[_ev()])
    uncertain_ungrounded = Finding(text="待定", uncertain=True)
    reports = [ReportResult(findings=[grounded, uncertain_ungrounded])]
    out = me.traceability_rate(reports)
    assert out["traceability_rate"] == pytest.approx(0.5)         # 2 条中 1 条锚定
    assert out["grounded_excl_uncertain"] == pytest.approx(1.0)   # 断言性结论全锚定


def test_abstention_metrics_confusion():
    # 该拒[T,T,F,F] vs 实拒[T,F,F,T]: TP=1,FP=1,FN=1,TN=1
    out = me.abstention_metrics([True, False, False, True], [True, True, False, False])
    assert out["abstain_precision"] == pytest.approx(0.5)
    assert out["abstain_recall"] == pytest.approx(0.5)
    assert out["abstain_accuracy"] == pytest.approx(0.5)


def test_e2e_metrics_with_gold_abstain():
    r = ReportResult(findings=[Finding(text="x", evidence=[_ev()])],
                     abstain=True, abstain_reason=AbstainReason.NO_EVIDENCE)
    out = me.e2e_metrics([r], gold_abstain=[True])
    assert out["metrics"]["abstain_recall"] == pytest.approx(1.0)
    assert "traceability_rate" in out["metrics"]


# ── 显著性 / CI（T053）─────────────────────────────────────────────

def test_bootstrap_ci_constant():
    ci = st.bootstrap_ci([5.0, 5.0, 5.0, 5.0], n_resamples=200, seed=0)
    assert ci.point == ci.low == ci.high == pytest.approx(5.0)


def test_paired_permutation_identical_vs_separated():
    ident = st.paired_permutation_test([1, 2, 3, 4], [1, 2, 3, 4], n_resamples=500, seed=0)
    assert ident == pytest.approx(1.0)
    a = [1.0] * 8
    b = [0.0] * 8
    p = st.paired_permutation_test(a, b, n_resamples=2000, seed=0)
    assert p < 0.05  # 系统性差异 → 显著


def test_paired_bootstrap_delta_ci_and_significance():
    a = [0.8, 0.9, 0.85, 0.95, 0.88]
    b = [0.5, 0.55, 0.52, 0.6, 0.58]
    ci = st.paired_bootstrap_delta_ci(a, b, n_resamples=500, seed=0)
    assert ci.point > 0
    assert st.is_significant(ci)  # CI 不含 0
    assert not st.is_significant(st.CI(point=0.0, low=-0.1, high=0.2, level=0.95))


def test_mcnemar_counts_and_empty():
    a = [True, True, False, False, False]
    b = [True, False, True, True, False]
    out = st.mcnemar_test(a, b)
    assert out["b01"] == 2.0  # a 错 b 对
    assert out["b10"] == 1.0  # a 对 b 错
    assert st.mcnemar_test([True, True], [True, True])["p_value"] == 1.0  # 无分歧


def test_stats_shape_mismatch_raises():
    with pytest.raises(ValueError):
        st.paired_bootstrap_delta_ci([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        st.paired_permutation_test([1], [1, 2])
