"""数据管线单测（T063 起步）—— 先覆盖 T024 绿框提取。

合成图（numpy 画绿框，不依赖 cv2 构造）验证 `extract_green_boxes`：
- 无 cv2 的本机：跑"缺包报清晰错 + 模块可 import"两条；功能测自动 skip。
- AutoDL（装了 opencv）：功能测真跑，验证坐标/面积带正确。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts.schemas import BBox, ROI, AreaBand
from src.data import coreset, ct_box, ct_inpaint, ct_label


def _img_with_green_box(h, w, x1, y1, x2, y2, thickness=2):
    """numpy 直接画一个纯绿(0,255,0)矩形**边框**（空心），模拟数据集里的 ROI 框。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    green = (0, 255, 0)
    img[y1:y1 + thickness, x1:x2] = green          # 上边
    img[y2 - thickness:y2, x1:x2] = green          # 下边
    img[y1:y2, x1:x1 + thickness] = green          # 左边
    img[y1:y2, x2 - thickness:x2] = green          # 右边
    return img


# ── 无重依赖也能跑的两条（本机 CI 即覆盖）─────────────────────────────

def test_module_imports_and_exports():
    assert hasattr(ct_box, "extract_green_boxes")
    assert hasattr(ct_box, "green_mask")
    assert ct_box.GreenBoxConfig().hsv_upper == (85, 255, 255)


def test_missing_cv2_raises_clear_error():
    """无 opencv 时报清晰 RuntimeError（含 opencv 字样），而非晦涩 ImportError。"""
    try:
        import cv2  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="opencv"):
            ct_box.extract_green_boxes(np.zeros((8, 8, 3), np.uint8))
    else:
        pytest.skip("本环境已装 cv2，跳过缺依赖分支")


# ── 需 opencv 的功能测（AutoDL 上真跑）────────────────────────────────

def test_extract_single_box_coords_and_band():
    pytest.importorskip("cv2")
    img = _img_with_green_box(200, 200, 50, 40, 150, 160, thickness=2)
    rois = ct_box.extract_green_boxes(img)
    assert len(rois) == 1
    roi = rois[0]
    assert isinstance(roi, ROI)
    # 外接矩形应贴合所画框（容差几个像素）
    assert abs(roi.bbox.x1 - 50) <= 2 and abs(roi.bbox.y1 - 40) <= 2
    assert abs(roi.bbox.x2 - 150) <= 2 and abs(roi.bbox.y2 - 160) <= 2
    # (100*120)/(200*200)=0.3 → LARGE
    assert roi.area_band is AreaBand.LARGE
    assert roi.confidence == 1.0 and roi.label is None


def test_small_lesion_band():
    pytest.importorskip("cv2")
    # 12x12 框 / 300x300 ≈ 0.0016 < 2% → SMALL（项目核心命题分层）
    img = _img_with_green_box(300, 300, 100, 100, 112, 112, thickness=1)
    rois = ct_box.extract_green_boxes(img)
    assert len(rois) == 1
    assert rois[0].area_band is AreaBand.SMALL


def test_no_box_returns_empty():
    pytest.importorskip("cv2")
    assert ct_box.extract_green_boxes(np.zeros((64, 64, 3), np.uint8)) == []


def test_min_area_fraction_filters_noise():
    pytest.importorskip("cv2")
    img = np.zeros((200, 200, 3), np.uint8)
    img[10:12, 10:12] = (0, 255, 0)  # 2x2 绿点：占比 4/40000=1e-4 边界
    rois = ct_box.extract_green_boxes(
        img, ct_box.GreenBoxConfig(min_area_fraction=1e-3)
    )
    assert rois == []  # 高于阈值才保留 → 噪点被滤


# ── T025 高斯热图标签（纯 numpy，本机即可全跑）─────────────────────────

def _roi(x1, y1, x2, y2):
    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return ROI(bbox=bbox, area_fraction=bbox.area / (200 * 200))


def test_heatmap_peak_at_box_center():
    hm = ct_label.render_heatmap([_roi(60, 60, 140, 140)], 200, 200)
    assert hm.shape == (200, 200)
    # 峰值≈1 且落在框中心 (100,100)
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    assert abs(ix - 100) <= 1 and abs(iy - 100) <= 1
    assert hm.max() == pytest.approx(1.0, abs=1e-3)


def test_empty_rois_all_zero():
    hm = ct_label.render_heatmap([], 64, 64)
    assert hm.shape == (64, 64) and float(hm.max()) == 0.0


def test_anneal_sharpens_heatmap():
    """退火系数越小 → 高斯越尖 → 距中心固定偏移处的值越小（粗→细）。"""
    roi = _roi(50, 50, 150, 150)  # 中心 (100,100)
    coarse = ct_label.render_heatmap([roi], 200, 200, anneal=1.0)
    fine = ct_label.render_heatmap([roi], 200, 200, anneal=0.25)
    # 离中心 15px 处：粗热图仍较亮，细热图明显更暗
    assert coarse[100, 115] > fine[100, 115]


def test_anneal_factor_monotone_decreasing():
    f0 = ct_label.sigma_anneal_factor(0, 100)
    fmid = ct_label.sigma_anneal_factor(50, 100)
    f1 = ct_label.sigma_anneal_factor(100, 100)
    assert f0 == pytest.approx(1.0) and f1 == pytest.approx(0.25)
    assert f0 > fmid > f1


def test_out_size_downsamples_and_maps_center():
    """输出到下采样网格：尺寸正确，峰值落在缩放后的中心。"""
    roi = _roi(0, 0, 200, 100)  # 原图中心 (100,50)，原图 200x200
    cfg = ct_label.HeatmapConfig(out_size=(50, 50))  # 1/4 网格
    hm = ct_label.render_heatmap([roi], 200, 200, config=cfg)
    assert hm.shape == (50, 50)
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    assert abs(ix - 25) <= 1 and abs(iy - 12) <= 1  # (100,50)*0.25=(25,12.5)


# ── T026 inpaint 抹框 + 二阶防泄露（需 opencv）───────────────────────

def _textured_with_box(h, w, x1, y1, x2, y2, thickness=2, seed=0):
    """灰度噪声背景（R=G=B，绿掩码只命中真实绿框）+ 一个绿色矩形边框。

    用灰度而非彩噪：彩色随机像素会大量被绿掩码误判，干扰"绿框是否被抹净"的判定。
    灰度仍有纹理 → inpaint 会真实改变像素，足以检出诱饵区差异。
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(h, w), dtype=np.uint8)
    img = np.stack([base, base, base], axis=-1)
    g = (0, 255, 0)
    img[y1:y1 + thickness, x1:x2] = g
    img[y2 - thickness:y2, x1:x2] = g
    img[y1:y2, x1:x1 + thickness] = g
    img[y1:y2, x2 - thickness:x2] = g
    return img


def test_inpaint_removes_green_box():
    pytest.importorskip("cv2")
    img = _textured_with_box(200, 200, 60, 60, 140, 140)
    before = int((ct_box.green_mask(img) > 0).sum())
    out = ct_inpaint.inpaint_image(img, config=ct_inpaint.InpaintConfig(decoy_count=0))
    after = int((ct_box.green_mask(out) > 0).sum())
    assert before > 0 and after == 0            # 绿框被彻底抹除
    assert out.shape == img.shape and out.dtype == np.uint8


def test_no_box_returns_unchanged():
    pytest.importorskip("cv2")
    img = np.full((64, 64, 3), 100, np.uint8)   # 无绿框
    out = ct_inpaint.inpaint_image(img)
    assert np.array_equal(out, img)             # 原样返回（训推一致：本就该是干净图）


def test_decoy_inpaints_region_outside_box():
    pytest.importorskip("cv2")
    img = _textured_with_box(200, 200, 80, 80, 130, 130)
    box = ct_box.extract_green_boxes(img)[0].bbox
    cfg0 = ct_inpaint.InpaintConfig(decoy_count=0)
    cfg2 = ct_inpaint.InpaintConfig(decoy_count=2)
    out0 = ct_inpaint.inpaint_image(img, config=cfg0, rng=np.random.default_rng(7))
    out2 = ct_inpaint.inpaint_image(img, config=cfg2, rng=np.random.default_rng(7))
    diff = np.any(out0 != out2, axis=-1)        # 两者差异 = 诱饵区
    assert diff.any()
    # 差异应落在真实框之外（诱饵不与框重叠）
    ys, xs = np.where(diff)
    inside = ((xs >= box.x1) & (xs <= box.x2) & (ys >= box.y1) & (ys <= box.y2))
    assert not inside.all()                     # 至少部分差异在框外


def test_inpaint_deterministic_with_seed():
    pytest.importorskip("cv2")
    img = _textured_with_box(200, 200, 70, 70, 120, 120)
    cfg = ct_inpaint.InpaintConfig(decoy_count=2)
    a = ct_inpaint.inpaint_image(img, config=cfg, rng=np.random.default_rng(0))
    b = ct_inpaint.inpaint_image(img, config=cfg, rng=np.random.default_rng(0))
    assert np.array_equal(a, b)                 # 同 seed → 诱饵位置可复现


# ── T027 CT coreset 选样（numpy + sklearn）──────────────────────────

def _blobs(per_cluster=20, seed=0):
    """三个分得很开的 2D 高斯团 + 各样本面积占比（覆盖三个 AreaBand）。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 10], [-10, 8]], dtype=float)
    X = np.vstack([c + rng.normal(0, 0.4, (per_cluster, 2)) for c in centers])
    # 面积占比：分别落进 small(<2%) / medium(2-5%) / large(>5%)
    fracs = np.concatenate([
        np.full(per_cluster, 0.01), np.full(per_cluster, 0.03), np.full(per_cluster, 0.08),
    ])
    return X, list(fracs)


def test_coreset_missing_sklearn_raises_clear_error():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        X, fracs = _blobs()
        with pytest.raises(RuntimeError, match="scikit-learn"):
            coreset.select_coreset(X, fracs)
    else:
        pytest.skip("本环境已装 sklearn，跳过缺依赖分支")


def test_coreset_respects_budget_and_subset():
    pytest.importorskip("sklearn")
    X, fracs = _blobs()
    ids = [f"s{i}" for i in range(len(X))]
    sel = coreset.select_coreset(X, fracs, ids, config=coreset.CoresetConfig(budget=0.2))
    assert 0 < len(sel) <= round(0.2 * len(X)) + 3   # 约束在预算附近
    assert len(set(sel)) == len(sel)                 # 无重复
    assert set(sel).issubset(set(ids))               # 合法子集


def test_coreset_returns_all_when_budget_exceeds_n():
    pytest.importorskip("sklearn")
    X, fracs = _blobs(per_cluster=5)
    sel = coreset.select_coreset(X, fracs, config=coreset.CoresetConfig(budget=999))
    assert len(sel) == len(X)


def test_coreset_covers_all_area_bands():
    pytest.importorskip("sklearn")
    X, fracs = _blobs()
    sel = coreset.select_coreset(X, fracs, config=coreset.CoresetConfig(budget=0.3))
    bands = {AreaBand.from_fraction(fracs[i]) for i in sel}
    assert bands == {AreaBand.SMALL, AreaBand.MEDIUM, AreaBand.LARGE}  # 三层都有代表


def test_coreset_deterministic():
    pytest.importorskip("sklearn")
    X, fracs = _blobs()
    cfg = coreset.CoresetConfig(budget=0.25)
    assert coreset.select_coreset(X, fracs, config=cfg) == \
        coreset.select_coreset(X, fracs, config=cfg)
