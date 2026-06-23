"""检测评估指标（T049）—— FROC / sensitivity@FP（主）+ mAP（辅），按面积带分层。

**为什么这是归因底座的一块**：创新 C（高斯退火定位）和 B（双路融合）声称"专治小病灶"。
要证明这句话，就得有一把**对面积分层敏感**的尺子——否则小病灶的增益会被大病灶的多数样本
淹没。本模块就是这把尺子。

主指标用 **FROC / sensitivity@FP** 而非 mAP：医学病灶检测里"每张图能容忍几个假阳"是临床
关切，FROC 直接答这个问题；mAP 作辅助对齐通用检测惯例。

分层主轴 = [schemas.AreaBand]（<2% / 2–5% / >5%），由 GT 病灶的 `area_band` 决定。
**不另造 stratify 工具**：`ROI.area_band` / `AreaBand.from_fraction` 已是唯一事实源。

匹配语义（FROC 与 AP 通用，constitution III"一次一变量"要求口径固定）：
- 每张图内按预测置信度降序贪心匹配，与某未匹配 GT 的 IoU ≥ 阈值则记 TP（并占用该 GT），
  否则记 FP。匹配只做一次（在全量预测上），扫阈值时只做累积——标准 FROC/AP 口径。
- 扫描所有预测置信度作为操作点：FP/图 = 累积FP / 图数；sensitivity = 累积TP / GT 总数。
  分层 sensitivity = 该带累积TP / 该带 GT 数（FP 轴是全局的，与带无关）。

纯 numpy/Python，无 torch；本地可全量单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.contracts.schemas import AreaBand, BBox, DetectionResult, ROI

# 默认读 sensitivity 的 FP/图 操作点（CAD 文献常用档位）。
DEFAULT_FP_POINTS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_IOU_THR: float = 0.3  # 小病灶定位宽松些（呼应小病灶命题），可按需调


def bbox_iou(a: BBox, b: BBox) -> float:
    """两个轴对齐框的 IoU ∈ [0, 1]。无重叠或零面积 → 0。"""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class _PredFlag:
    """单条预测的匹配结果（一次匹配后定型，扫阈值时复用）。"""

    confidence: float
    is_tp: bool
    gt_band: AreaBand | None  # TP 命中的 GT 所属面积带；FP 为 None


def match_image(
    pred_rois: Sequence[ROI], gt_rois: Sequence[ROI], iou_thr: float
) -> list[_PredFlag]:
    """单张图内贪心匹配，返回每条预测的 TP/FP 标记（按置信度降序）。

    每个 GT 至多被一条预测占用；预测优先匹配 IoU 最大的未占用 GT。
    """
    order = sorted(range(len(pred_rois)), key=lambda i: pred_rois[i].confidence, reverse=True)
    gt_taken = [False] * len(gt_rois)
    flags: list[_PredFlag] = []
    for i in order:
        pred = pred_rois[i]
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_rois):
            if gt_taken[j]:
                continue
            v = bbox_iou(pred.bbox, gt.bbox)
            if v >= iou_thr and v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            gt_taken[best_j] = True
            flags.append(_PredFlag(pred.confidence, True, gt_rois[best_j].area_band))
        else:
            flags.append(_PredFlag(pred.confidence, False, None))
    return flags


def _gt_rois(item: DetectionResult | Sequence[ROI]) -> Sequence[ROI]:
    """GT 既可传 DetectionResult 也可直接传 ROI 列表。"""
    return item.rois if isinstance(item, DetectionResult) else item


@dataclass
class FROCCurve:
    """FROC 曲线：随置信度阈值下降，(FP/图, 整体 sensitivity, 各带 sensitivity)。"""

    fp_per_image: np.ndarray            # 单调不减
    sensitivity: np.ndarray             # 整体，单调不减
    sensitivity_by_band: dict[AreaBand, np.ndarray] = field(default_factory=dict)
    total_gt: int = 0
    total_gt_by_band: dict[AreaBand, int] = field(default_factory=dict)
    n_images: int = 0


def froc_curve(
    preds: Sequence[DetectionResult],
    gts: Sequence[DetectionResult | Sequence[ROI]],
    *,
    iou_thr: float = DEFAULT_IOU_THR,
) -> FROCCurve:
    """对成对的（预测, GT）逐图匹配并扫置信度，算出 FROC 曲线（含分层）。"""
    if len(preds) != len(gts):
        raise ValueError(f"preds 与 gts 数量不一致：{len(preds)} vs {len(gts)}")
    n_images = len(preds)

    all_flags: list[_PredFlag] = []
    total_gt = 0
    total_gt_by_band: dict[AreaBand, int] = {b: 0 for b in AreaBand}
    for pred, gt in zip(preds, gts):
        gt_list = _gt_rois(gt)
        total_gt += len(gt_list)
        for roi in gt_list:
            total_gt_by_band[roi.area_band] += 1
        all_flags.extend(match_image(pred.rois, gt_list, iou_thr))

    # 按置信度降序扫描，累积 TP/FP。
    all_flags.sort(key=lambda f: f.confidence, reverse=True)
    fp = tp = 0
    tp_by_band: dict[AreaBand, int] = {b: 0 for b in AreaBand}
    fp_list, sens_list = [0.0], [0.0]
    band_lists: dict[AreaBand, list[float]] = {b: [0.0] for b in AreaBand}
    for f in all_flags:
        if f.is_tp:
            tp += 1
            if f.gt_band is not None:
                tp_by_band[f.gt_band] += 1
        else:
            fp += 1
        fp_list.append(fp / n_images if n_images else 0.0)
        sens_list.append(tp / total_gt if total_gt else 0.0)
        for b in AreaBand:
            denom = total_gt_by_band[b]
            band_lists[b].append(tp_by_band[b] / denom if denom else 0.0)

    return FROCCurve(
        fp_per_image=np.asarray(fp_list, dtype=float),
        sensitivity=np.asarray(sens_list, dtype=float),
        sensitivity_by_band={b: np.asarray(v, dtype=float) for b, v in band_lists.items()},
        total_gt=total_gt,
        total_gt_by_band=total_gt_by_band,
        n_images=n_images,
    )


def sensitivity_at_fp(
    curve: FROCCurve, fp_points: Sequence[float] = DEFAULT_FP_POINTS
) -> dict[float, float]:
    """在给定 FP/图 操作点读整体 sensitivity（阶梯保守取：≤该FP的最高 sensitivity）。"""
    return _read_at_fp(curve.fp_per_image, curve.sensitivity, fp_points)


def sensitivity_at_fp_by_band(
    curve: FROCCurve, fp_points: Sequence[float] = DEFAULT_FP_POINTS
) -> dict[AreaBand, dict[float, float]]:
    """分层版 sensitivity@FP：FP 轴全局，sensitivity 取各带曲线。"""
    out: dict[AreaBand, dict[float, float]] = {}
    for band, sens in curve.sensitivity_by_band.items():
        out[band] = _read_at_fp(curve.fp_per_image, sens, fp_points)
    return out


def _read_at_fp(
    fp_axis: np.ndarray, sens: np.ndarray, fp_points: Sequence[float]
) -> dict[float, float]:
    """在阶梯曲线上取 fp_axis ≤ point 的最大 sensitivity（无满足点则 0）。"""
    out: dict[float, float] = {}
    for p in fp_points:
        mask = fp_axis <= p + 1e-9
        out[float(p)] = float(sens[mask].max()) if mask.any() else 0.0
    return out


def average_precision(
    preds: Sequence[DetectionResult],
    gts: Sequence[DetectionResult | Sequence[ROI]],
    *,
    iou_thr: float = DEFAULT_IOU_THR,
) -> float:
    """VOC all-point 插值 AP（辅助指标）。无 GT 时返回 0。"""
    if len(preds) != len(gts):
        raise ValueError(f"preds 与 gts 数量不一致：{len(preds)} vs {len(gts)}")
    flags: list[_PredFlag] = []
    total_gt = 0
    for pred, gt in zip(preds, gts):
        gt_list = _gt_rois(gt)
        total_gt += len(gt_list)
        flags.extend(match_image(pred.rois, gt_list, iou_thr))
    if total_gt == 0 or not flags:
        return 0.0

    flags.sort(key=lambda f: f.confidence, reverse=True)
    tp = fp = 0
    precisions, recalls = [], []
    for f in flags:
        if f.is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_gt)
    return _voc_ap(np.asarray(recalls), np.asarray(precisions))


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """all-point 插值：包络 precision（从右往左取最大）后按 recall 增量积分。"""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_metrics(
    preds: Sequence[DetectionResult],
    gts: Sequence[DetectionResult | Sequence[ROI]],
    *,
    iou_thr: float = DEFAULT_IOU_THR,
    fp_points: Sequence[float] = DEFAULT_FP_POINTS,
) -> dict:
    """一站式：整体 + 分层 FROC sensitivity@FP（主）+ mAP（辅）。

    返回 dict 便于灌进 [schemas.EvalRecord]：`metrics`（整体）+ `stratified`（按带）。
    """
    curve = froc_curve(preds, gts, iou_thr=iou_thr)
    overall = sensitivity_at_fp(curve, fp_points)
    by_band = sensitivity_at_fp_by_band(curve, fp_points)
    ap = average_precision(preds, gts, iou_thr=iou_thr)

    metrics = {f"sens@{p}fp": v for p, v in overall.items()}
    metrics["mAP"] = ap
    stratified = {
        band.value: {f"sens@{p}fp": v for p, v in pts.items()}
        for band, pts in by_band.items()
    }
    return {"metrics": metrics, "stratified": stratified,
            "n_images": curve.n_images, "total_gt": curve.total_gt}


__all__ = [
    "DEFAULT_FP_POINTS",
    "DEFAULT_IOU_THR",
    "bbox_iou",
    "match_image",
    "FROCCurve",
    "froc_curve",
    "sensitivity_at_fp",
    "sensitivity_at_fp_by_band",
    "average_precision",
    "detection_metrics",
]
