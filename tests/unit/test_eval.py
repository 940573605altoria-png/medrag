"""评估指标单测（T049/T050）—— 纯逻辑、CPU、无 torch。

用合成的、TP/FP 已知的 DetectionResult 与文本对手算对拍 FROC/sensitivity@FP/mAP 与实体F1，
守住"创新增益"那把尺子的口径不漂（constitution III）。
"""

from __future__ import annotations

import math

import pytest

from src.contracts.schemas import AreaBand, BBox, DetectionResult, ROI
from src.eval import metrics_detection as md
from src.eval import metrics_report as mr


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
