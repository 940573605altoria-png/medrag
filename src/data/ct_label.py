"""框 → 宽高斯热图标签（T025）—— 创新 C 的定位监督目标。

C 的"高斯退火定位"：ViT 上的辅助定位头预测一张热图（即系统"检测器"，推理无框可用），
其**训练目标**就是本模块从 ROI 框生成的高斯热图。核心是 **σ 退火**——训练早期用大 σ（宽、
粗，易学到大致位置），随训练步推进 σ 由大到小（窄、细，逼近精确中心）。

设计（对齐 plan.md / constitution）：
- **CenterNet 式 splat**：每个框在其中心放一个峰值=1 的二维高斯，多框取逐元素 max。
  损失（CenterNet penalty-reduced focal + Dice）在 T029，本模块只产 target。
- **σ 随框大小自适应 + 随训练退火**：σ = sigma_scale × 框边长 × anneal(t)，下限 sigma_min。
  小病灶框小 → σ 小（峰更尖），符合"小病灶重点"命题。
- **可输出到下采样网格**：ViT 特征图有 stride，`HeatmapConfig.out_size` 指定网格尺寸，
  中心/σ 自动按缩放映射（训推一致）。
- **纯 numpy**：无 cv2/torch 依赖，CPU 可单测；模型内再转 torch。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.contracts.schemas import BBox, ROI

if TYPE_CHECKING:
    import numpy as np


@dataclass
class HeatmapConfig:
    """热图标签生成配置。"""

    out_size: tuple[int, int] | None = None  # (H, W) 输出网格；None=用原图尺寸
    sigma_scale: float = 0.1   # σ = sigma_scale × 框边长（随框大小自适应）
    sigma_min: float = 1.0     # σ 下限：退火到最细时仍可见，避免热图塌成 0
    elliptical: bool = True    # True: 按框宽/高分别定 σx/σy；False: 用边长均值各向同性


def sigma_anneal_factor(
    step: int, total_steps: int, *, start: float = 1.0, end: float = 0.25,
    schedule: str = "cosine",
) -> float:
    """σ 退火系数（粗→细）：随训练步从 `start` 单调降到 `end`，乘到 σ 上。

    `cosine`（默认，平滑）或 `linear`。`total_steps<=0` 或越界自动夹紧。用法：
    `render_heatmap(..., anneal=sigma_anneal_factor(step, total))`。
    """
    if total_steps <= 0:
        return end
    t = min(max(step / total_steps, 0.0), 1.0)
    if schedule == "linear":
        f = 1.0 - t
    else:  # cosine
        f = 0.5 * (1.0 + math.cos(math.pi * t))
    return end + (start - end) * f


def _draw_gaussian(hm: "np.ndarray", cx: float, cy: float, sx: float, sy: float) -> None:
    """在热图上以 (cx,cy) 为中心 splat 一个二维高斯（峰值1），就地取 max。"""
    import numpy as np  # noqa: PLC0415

    H, W = hm.shape
    rx, ry = int(math.ceil(3 * sx)), int(math.ceil(3 * sy))  # ±3σ 窗口
    icx, icy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, icx - rx), min(W, icx + rx + 1)
    y0, y1 = max(0, icy - ry), min(H, icy + ry + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1) - cx          # 距中心的浮点偏移（中心可为亚像素）
    ys = np.arange(y0, y1) - cy
    gx = np.exp(-(xs ** 2) / (2.0 * sx * sx))
    gy = np.exp(-(ys ** 2) / (2.0 * sy * sy))
    g = np.outer(gy, gx)                 # (y, x)
    np.maximum(hm[y0:y1, x0:x1], g, out=hm[y0:y1, x0:x1])


def _as_bbox(item: Any) -> BBox:
    """接受 ROI 或 BBox，统一取 BBox。"""
    return item.bbox if isinstance(item, ROI) else item


def render_heatmap(
    rois: list[Any], height: int, width: int, *,
    anneal: float = 1.0, config: HeatmapConfig | None = None,
) -> "np.ndarray":
    """把若干 ROI/BBox 渲染成一张高斯热图标签 (float32, 值域 0–1)。

    `height/width` 是原图尺寸；`config.out_size` 给定时输出到该网格（中心/σ 按比例缩放）。
    `anneal` 是 σ 退火系数（见 `sigma_anneal_factor`），1.0=最粗。空 rois → 全 0（背景）。
    """
    import numpy as np  # noqa: PLC0415

    cfg = config or HeatmapConfig()
    out_h, out_w = cfg.out_size or (height, width)
    hm = np.zeros((out_h, out_w), dtype=np.float32)
    if not rois:
        return hm

    sx_scale = out_w / float(width)   # 原图坐标 → 输出网格坐标的缩放
    sy_scale = out_h / float(height)

    for item in rois:
        b = _as_bbox(item)
        cx = (b.x1 + b.x2) / 2.0 * sx_scale
        cy = (b.y1 + b.y2) / 2.0 * sy_scale
        bw = (b.x2 - b.x1) * sx_scale
        bh = (b.y2 - b.y1) * sy_scale
        if cfg.elliptical:
            sx = max(cfg.sigma_min, cfg.sigma_scale * bw * anneal)
            sy = max(cfg.sigma_min, cfg.sigma_scale * bh * anneal)
        else:
            s = max(cfg.sigma_min, cfg.sigma_scale * (bw + bh) / 2.0 * anneal)
            sx = sy = s
        _draw_gaussian(hm, cx, cy, sx, sy)

    return hm


__all__ = ["HeatmapConfig", "sigma_anneal_factor", "render_heatmap"]
