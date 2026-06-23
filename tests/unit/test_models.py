"""模型层单测 —— CPU 可跑，不下载 30B 权重、不需要 torch。

只验证加载器的"无重依赖也能 import + 配置逻辑正确 + 缺依赖给清晰错误"，
真正的加载/推理在 AutoDL 上由 `scripts/smoke_gpu.py --with-model` 覆盖。
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from src.contracts.schemas import BBox, ROI
from src.models import loc_head, losses, qwen3vl


def test_module_imports_without_torch():
    """守卫导入：本机无 torch/transformers 也能 import 模块（供契约/单测引用）。"""
    importlib.reload(qwen3vl)
    assert qwen3vl.DEFAULT_MODEL_ID == "Qwen/Qwen3-VL-30B-A3B-Instruct"


def test_resolve_model_id_precedence(monkeypatch):
    # 显式参数优先
    assert qwen3vl.resolve_model_id("foo/bar") == "foo/bar"
    # 环境变量次之
    monkeypatch.setenv("MEDRAG_BASE_MODEL", "env/model")
    assert qwen3vl.resolve_model_id() == "env/model"
    # 默认兜底
    monkeypatch.delenv("MEDRAG_BASE_MODEL", raising=False)
    assert qwen3vl.resolve_model_id() == qwen3vl.DEFAULT_MODEL_ID


def test_config_model_kwargs_defaults():
    cfg = qwen3vl.Qwen3VLConfig(model_id="x/y")
    kw = cfg.model_kwargs()
    assert kw["device_map"] == "auto"
    assert kw["dtype"] == "auto"
    assert kw["trust_remote_code"] is True
    # 默认无量化、无 attn_implementation 注入
    assert "quantization_config" not in kw
    assert "attn_implementation" not in kw


def test_config_attn_impl_injected():
    cfg = qwen3vl.Qwen3VLConfig(attn_implementation="flash_attention_2")
    assert cfg.model_kwargs()["attn_implementation"] == "flash_attention_2"


def test_load_base_without_transformers_raises_clear_error():
    """无 transformers 时 load_base 报清晰 RuntimeError，而非晦涩 ImportError。"""
    pytest.importorskip  # noqa: B018 - 占位，确保 pytest 可用
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="transformers"):
            qwen3vl.load_base()
    else:
        pytest.skip("本环境已装 transformers，跳过缺依赖分支")


# ── T028 定位头：热图→DetectionResult 解码（纯 numpy，本机即可跑）────

def test_loc_head_module_imports_without_torch():
    importlib.reload(loc_head)
    assert hasattr(loc_head, "heatmap_to_detection")
    assert loc_head.LocHeadConfig().in_dim == 1152


def test_heatmap_to_detection_finds_blob():
    hm = np.zeros((20, 20), dtype=float)
    hm[8:12, 8:12] = 0.9                       # 一个高响应方块
    det = loc_head.heatmap_to_detection(hm, threshold=0.3, score_min=0.3)
    assert not det.abstained and len(det.rois) == 1
    roi = det.rois[0]
    assert (roi.bbox.x1, roi.bbox.y1, roi.bbox.x2, roi.bbox.y2) == (8, 8, 12, 12)
    assert roi.confidence == pytest.approx(0.9)


def test_heatmap_to_detection_empty_abstains():
    det = loc_head.heatmap_to_detection(np.zeros((16, 16)), threshold=0.3)
    assert det.abstained and det.rois == []     # 无显著病灶 → 拒答，不强行框


def test_heatmap_to_detection_scales_to_image():
    hm = np.zeros((10, 10), dtype=float)
    hm[2:4, 2:4] = 0.8
    det = loc_head.heatmap_to_detection(hm, image_hw=(40, 40))  # 4x 放大
    b = det.rois[0].bbox
    assert (b.x1, b.y1, b.x2, b.y2) == (8, 8, 16, 16)


# ── T029 定位损失（需 torch；本机 skip，AutoDL 真跑）────────────────

def test_focal_lower_when_pred_matches_target():
    torch = pytest.importorskip("torch")
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    good = losses.penalty_reduced_focal_loss(target.clone(), target)
    bad = losses.penalty_reduced_focal_loss(torch.full((1, 1, 8, 8), 0.5), target)
    assert float(good) < float(bad)


def test_dice_zero_when_perfect():
    torch = pytest.importorskip("torch")
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3:5, 3:5] = 1.0
    assert float(losses.dice_loss(target.clone(), target)) == pytest.approx(0.0, abs=1e-4)


def test_localization_loss_returns_components():
    torch = pytest.importorskip("torch")
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    total, parts = losses.localization_loss(torch.full((1, 1, 8, 8), 0.4), target)
    assert {"focal", "dice"} <= set(parts) and float(total) > 0


def test_make_annealed_target_shape_and_peak():
    torch = pytest.importorskip("torch")
    roi = ROI(bbox=BBox(x1=20, y1=20, x2=44, y2=44), area_fraction=0.1)
    tgt = losses.make_annealed_target([[roi]], image_hw=(64, 64), heatmap_hw=(16, 16),
                                      step=0, total_steps=100)
    assert tuple(tgt.shape) == (1, 1, 16, 16)
    assert float(tgt.max()) == pytest.approx(1.0, abs=1e-3)


def test_loc_head_forward_shape():
    pytest.importorskip("torch")
    import torch
    head = loc_head.build_localization_head(loc_head.LocHeadConfig(in_dim=8, hidden=16, upscale=2))
    out = head(torch.randn(2, 16, 8))           # N=16 → 4x4 grid，upscale 2 → 8x8
    assert tuple(out.shape) == (2, 1, 8, 8)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
