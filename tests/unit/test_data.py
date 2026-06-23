"""数据管线单测（T063 起步）—— 先覆盖 T024 绿框提取。

合成图（numpy 画绿框，不依赖 cv2 构造）验证 `extract_green_boxes`：
- 无 cv2 的本机：跑"缺包报清晰错 + 模块可 import"两条；功能测自动 skip。
- AutoDL（装了 opencv）：功能测真跑，验证坐标/面积带正确。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts.schemas import BBox, ROI, AreaBand
from src.data import ct_box, ct_label


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
