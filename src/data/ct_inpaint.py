"""inpaint 抹绿框 + 二阶随机区域防泄露（T026）—— 训推同分布的关键（FR-004）。

创新 C 的铁律：**绿框只当标签，绝不进模型输入**。训练时把图上的绿框 inpaint 抹掉，喂
"干净图"；推理时本就无框、也是干净图——两端分布一致（constitution II）。

**为什么还要二阶随机 inpaint**：只抹真实框会留下一道"恰好在病灶处"的 inpaint 痕迹，定位头
可能偷学这道痕迹（= 标签泄露），而非真正的病灶影像特征；推理无此痕迹则失效。对策：在随机
**非重叠**位置画同尺寸矩形一起 inpaint，制造若干一模一样的痕迹，打断"痕迹↔病灶位置"的相关，
逼模型靠真实特征定位。

复用 [ct_box.green_mask] 取框像素（与"被当框提取"的像素同源，保证抹的就是提的），膨胀后
`cv2.inpaint`。守卫导入 cv2，缺包给清晰错误。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.contracts.schemas import BBox, ROI
from src.data.ct_box import GreenBoxConfig, _load_rgb, extract_green_boxes, green_mask

if TYPE_CHECKING:
    import numpy as np


@dataclass
class InpaintConfig:
    """inpaint 配置。"""

    dilate_ksize: int = 5        # 膨胀核：完整覆盖框线+边缘，避免残留绿边
    inpaint_radius: int = 3      # cv2.inpaint 半径
    method: str = "telea"        # telea | ns
    decoy_count: int = 1         # 每个真实框额外 inpaint 的随机诱饵区数（二阶防泄露）
    box_line_thickness: int = 2  # 诱饵矩形线宽（模拟真实细框线）
    max_place_tries: int = 20    # 诱饵随机放置避让真实框的最大尝试次数


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415

        return cv2
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "inpaint 需要 opencv-python（data 依赖组）：pip install opencv-python"
        ) from exc


def _overlaps(a: tuple[int, int, int, int], b: BBox, margin: int = 2) -> bool:
    """轴对齐矩形 a=(x1,y1,x2,y2) 与 bbox b 是否相交（含 margin 余量）。"""
    return not (
        a[2] + margin < b.x1 or a[0] - margin > b.x2
        or a[3] + margin < b.y1 or a[1] - margin > b.y2
    )


def inpaint_image(
    image: Any, rois: list[Any] | None = None, *,
    config: InpaintConfig | None = None,
    green_config: GreenBoxConfig | None = None,
    rng: "np.random.Generator | None" = None,
) -> "np.ndarray":
    """抹掉图上绿框（+二阶诱饵区）并返回干净 RGB 图 (uint8 HxWx3)。

    `rois` 不给则用 `extract_green_boxes` 自行提取（仅用于诱饵尺寸）。无框 → 原样返回副本。
    传入 seeded `rng`（`np.random.default_rng(seed)`）可复现诱饵位置。
    """
    cv2 = _require_cv2()
    import numpy as np  # noqa: PLC0415

    cfg = config or InpaintConfig()
    gcfg = green_config or GreenBoxConfig()
    rng = rng if rng is not None else np.random.default_rng()

    rgb = _load_rgb(image)
    h, w = rgb.shape[:2]

    boxes = rois if rois is not None else extract_green_boxes(rgb, gcfg)
    if not boxes:
        return rgb.copy()  # 无框：训推一致下本就该是干净图，不动

    # 真实框像素掩码（与提取同源）
    mask = green_mask(rgb, gcfg).copy()

    # 二阶：每个真实框在随机非重叠处画同尺寸矩形线，混入同一张掩码
    bboxes = [b.bbox if isinstance(b, ROI) else b for b in boxes]
    for bb in bboxes:
        bw, bh = int(bb.x2 - bb.x1), int(bb.y2 - bb.y1)
        if bw <= 0 or bh <= 0 or bw >= w or bh >= h:
            continue
        for _ in range(cfg.decoy_count):
            for _try in range(cfg.max_place_tries):
                dx = int(rng.integers(0, w - bw))
                dy = int(rng.integers(0, h - bh))
                cand = (dx, dy, dx + bw, dy + bh)
                if not any(_overlaps(cand, ob) for ob in bboxes):
                    cv2.rectangle(mask, (dx, dy), (dx + bw, dy + bh),
                                  255, cfg.box_line_thickness)
                    break

    # 膨胀确保完整覆盖线宽，再 inpaint
    if cfg.dilate_ksize > 1:
        kernel = np.ones((cfg.dilate_ksize, cfg.dilate_ksize), np.uint8)
        mask = cv2.dilate(mask, kernel)

    flags = cv2.INPAINT_NS if cfg.method == "ns" else cv2.INPAINT_TELEA
    out = cv2.inpaint(rgb, mask, cfg.inpaint_radius, flags)
    return out


__all__ = ["InpaintConfig", "inpaint_image"]
