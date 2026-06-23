"""ViT 辅助定位头（T028）—— 创新 C 的"检测器"，推理无框可用。

训练时它学着把 ViT 的 patch 特征解码成一张定位热图（监督=T025 的退火高斯标签，损失=T029）；
推理时**无需绿框**，直接前向出热图 → 解码成 `DetectionResult`（框/面积带/置信），即系统的
病灶检测能力。这正是 C "绿框只当标签、推理不依赖框"的落点。

两块互相独立：
- `LocalizationHead`（torch nn.Module）：patch token → 热图概率。需 torch。
- `heatmap_to_detection`（纯 numpy）：热图 → `DetectionResult`，实现检测契约。**无需 torch**，
  也接受 torch 张量（自动搬到 numpy），本机即可单测。

守卫导入：无 torch 也能 import 本模块（`heatmap_to_detection` 可用），只有构建/前向 nn.Module
才需 torch；缺包给清晰错误。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.contracts.schemas import BBox, ROI, DetectionResult

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 环境相关
    _HAS_TORCH = False


@dataclass
class LocHeadConfig:
    """定位头配置（按基座 ViT 隐藏维调 in_dim）。"""

    in_dim: int = 1152    # ViT 隐藏维（Qwen3-VL vision 约 1152；按实测调）
    hidden: int = 256
    upscale: int = 1      # 卷积后上采样倍数，把 token 网格放大到目标热图分辨率


# ── 纯 numpy：热图 → DetectionResult（无 torch 也能跑）────────────────

def _connected_components(mask) -> list:
    """4-邻接连通域（numpy 栈式洪泛填充）。热图网格小，足够用。"""
    import numpy as np  # noqa: PLC0415

    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                pts = []
                while stack:
                    cy, cx = stack.pop()
                    pts.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(np.array(pts))
    return comps


def heatmap_to_detection(
    heatmap: Any, *, threshold: float = 0.3, score_min: float = 0.3,
    image_hw: tuple[int, int] | None = None, image_id: str | None = None,
    include_heatmap: bool = False,
) -> DetectionResult:
    """热图 → `DetectionResult`。阈值二值化 → 连通域外接框 → 每框置信=域内峰值。

    `heatmap`: (H,W) 概率，np.ndarray 或 torch 张量（多余维度自动 squeeze）。
    `image_hw` 给定则把框坐标缩放回原图像素；否则用热图网格坐标。`area_fraction` 用占比
    （尺度无关，恒在热图空间算）。无显著域 → `abstained=True`（不强行框，呼应禁编造）。
    """
    import numpy as np  # noqa: PLC0415

    hm = heatmap
    if hasattr(hm, "detach"):           # torch 张量
        hm = hm.detach().cpu().numpy()
    hm = np.squeeze(np.asarray(hm, dtype=float))
    if hm.ndim != 2:
        raise ValueError(f"heatmap squeeze 后需为 2D，实得 shape={hm.shape}")
    H, W = hm.shape
    total = float(H * W) or 1.0

    rois: list[ROI] = []
    for comp in _connected_components(hm >= threshold):
        ys, xs = comp[:, 0], comp[:, 1]
        conf = float(hm[ys, xs].max())
        if conf < score_min:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        frac = ((x2 - x1) * (y2 - y1)) / total
        if image_hw is not None:
            ih, iw = image_hw
            sx, sy = iw / W, ih / H
            bbox = BBox(x1=x1 * sx, y1=y1 * sy, x2=x2 * sx, y2=y2 * sy)
        else:
            bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
        rois.append(ROI(bbox=bbox, area_fraction=min(frac, 1.0), confidence=conf))

    rois.sort(key=lambda r: r.confidence, reverse=True)
    return DetectionResult(
        rois=rois,
        heatmap=hm.tolist() if include_heatmap else None,
        image_id=image_id,
        abstained=len(rois) == 0,
    )


# ── torch nn.Module：patch token → 热图概率（需 torch）──────────────

if _HAS_TORCH:

    class LocalizationHead(nn.Module):
        """轻量卷积解码头：ViT patch token (B,N,C) → 定位热图概率 (B,1,H,W)。"""

        def __init__(self, config: LocHeadConfig | None = None):
            super().__init__()
            cfg = config or LocHeadConfig()
            c = cfg.hidden
            self.proj = nn.Conv2d(cfg.in_dim, c, 1)
            self.body = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(min(8, c), c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(min(8, c), c), nn.ReLU(inplace=True),
            )
            self.out = nn.Conv2d(c, 1, 1)
            self.upscale = cfg.upscale

        def forward(self, tokens, grid_hw: tuple[int, int] | None = None):
            """tokens: (B, N, in_dim)。`grid_hw` 缺省按 √N 方阵推断。返回 (B,1,H,W) 概率。"""
            B, N, C = tokens.shape
            if grid_hw is None:
                g = int(round(N ** 0.5))
                grid_hw = (g, g)
            gh, gw = grid_hw
            x = tokens.transpose(1, 2).reshape(B, C, gh, gw)
            x = self.body(self.proj(x))
            x = self.out(x)
            if self.upscale and self.upscale > 1:
                x = F.interpolate(x, scale_factor=self.upscale, mode="bilinear",
                                  align_corners=False)
            return torch.sigmoid(x)


def build_localization_head(config: LocHeadConfig | None = None):
    """构建定位头；无 torch 时给清晰错误。"""
    if not _HAS_TORCH:  # pragma: no cover - 环境相关
        raise RuntimeError("构建 LocalizationHead 需要 torch")
    return LocalizationHead(config)


__all__ = ["LocHeadConfig", "heatmap_to_detection", "build_localization_head"]
