"""定位损失（T029）—— CenterNet penalty-reduced focal + Dice + σ(t) 退火目标。

这是创新 C "高斯退火"那条线的闭合点：T025 生成随训练步收窄的高斯**目标**，本模块用它监督
T028 定位头的预测热图。

- **penalty-reduced focal**（CenterNet）：对正样本峰值与负样本分别加权，`(1-target)^β` 让靠近
  峰的"近似正确"负样本惩罚更轻，专治热图回归的极端正负不均衡。
- **Dice**：补充区域重叠度量，缓解小目标在像素级 focal 下信号过弱（呼应小病灶命题）。
- **σ(t) 退火**：`make_annealed_target` 调 [ct_label] 的 `sigma_anneal_factor` + `render_heatmap`，
  早期宽（粗定位）、后期窄（精定位），把数据侧标签生成与训练步耦合起来。

纯函数（非 nn.Module），torch 延迟导入：无 torch 也能 import 本模块，调用才需 torch。
"""

from __future__ import annotations

from typing import Any, Sequence


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError("定位损失计算需要 torch") from exc


def penalty_reduced_focal_loss(pred, target, *, alpha: float = 2.0, beta: float = 4.0,
                               eps: float = 1e-6):
    """CenterNet penalty-reduced focal。pred/target 同形 (…,H,W)，概率∈[0,1]，峰值处 target=1。"""
    torch = _torch()
    pred = pred.clamp(eps, 1.0 - eps)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    neg_weights = (1.0 - target).pow(beta)

    pos_loss = torch.log(pred) * (1.0 - pred).pow(alpha) * pos
    neg_loss = torch.log(1.0 - pred) * pred.pow(alpha) * neg_weights * neg
    n_pos = pos.sum()
    if n_pos > 0:
        return -(pos_loss.sum() + neg_loss.sum()) / n_pos
    return -neg_loss.sum()          # 无正峰（极少见）：仅负项


def dice_loss(pred, target, *, eps: float = 1e-6):
    """软 Dice 损失：1 - 2·∑(p·t)/(∑p+∑t)。对小目标比像素级损失更稳。"""
    _torch()
    num = 2.0 * (pred * target).sum() + eps
    den = pred.sum() + target.sum() + eps
    return 1.0 - num / den


def localization_loss(pred, target, *, w_focal: float = 1.0, w_dice: float = 1.0,
                      alpha: float = 2.0, beta: float = 4.0):
    """组合损失 = w_focal·focal + w_dice·dice。返回 (total, {'focal','dice'}) 供记录。"""
    focal = penalty_reduced_focal_loss(pred, target, alpha=alpha, beta=beta)
    dice = dice_loss(pred, target)
    total = w_focal * focal + w_dice * dice
    return total, {"focal": focal, "dice": dice}


def make_annealed_target(
    rois_batch: Sequence[Sequence[Any]], image_hw: tuple[int, int],
    heatmap_hw: tuple[int, int], step: int, total_steps: int, *,
    device: Any = None, dtype: Any = None,
):
    """按当前训练步生成退火高斯目标张量 (B,1,Hh,Hw)。

    `rois_batch`: 长度 B 的列表，每项是该样本的 ROI 列表；`image_hw` 原图尺寸；
    `heatmap_hw` 目标网格（与定位头输出一致）。σ 由 `sigma_anneal_factor(step,total)` 决定。
    """
    import numpy as np  # noqa: PLC0415

    from src.data.ct_label import HeatmapConfig, render_heatmap, sigma_anneal_factor

    torch = _torch()
    anneal = sigma_anneal_factor(step, total_steps)
    H, W = image_hw
    cfg = HeatmapConfig(out_size=heatmap_hw)
    maps = [render_heatmap(rois, H, W, anneal=anneal, config=cfg) for rois in rois_batch]
    arr = np.stack(maps)[:, None, :, :]     # (B,1,Hh,Hw)
    return torch.as_tensor(arr, dtype=dtype or torch.float32, device=device)


__all__ = [
    "penalty_reduced_focal_loss",
    "dice_loss",
    "localization_loss",
    "make_annealed_target",
]
