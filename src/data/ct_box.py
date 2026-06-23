"""绿框提取（T024）—— 创新 C 的定位监督入口。

**为什么是入口**：实测（`scripts/probe_dataset_c.py`，2026-06-23）确认 MedTrinity-25M
`25M_demo` 的 CT 图上**画了一个绿色规整矩形 ROI 框**（每图约 0.5–0.75% 绿像素），且
caption 只用文字描述位置、**无数字坐标**——所以那个绿框就是 C 唯一可用的定位信号。

本模块做一件事：`cv2.inRange` 在 HSV 空间提绿 → 连通域外接矩形 → 坐标。产出对齐数据契约
`ROI`（`bbox` + `area_fraction` 定 `AreaBand`，`confidence=1.0` 因是标注真值，`label=None`
因 CT 无疾病标签）。下游：
- T025 `ct_label.py` 把 `ROI.bbox` → 宽高斯热图标签；
- T026 `ct_inpaint.py` 复用本模块的 `green_mask` 抹掉绿框喂干净图（训推一致、抗泄露）。

**守卫导入**：cv2(opencv-python) 是 `data` 可选依赖组，本文件无 cv2 也能 import（供契约/
单测引用），只有真正调用提取函数才需要 cv2；缺包给清晰 RuntimeError。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.contracts.schemas import BBox, ROI

if TYPE_CHECKING:  # 仅类型检查期引用，运行期不强依赖
    import numpy as np


# 纯绿框 RGB(0,255,0) 在 OpenCV HSV(H:0–179) 下 H≈60；留足带宽容抗锯齿。
@dataclass
class GreenBoxConfig:
    """绿框检测阈值——按实图微调（probe 存的 PNG 是调参依据）。"""

    hsv_lower: tuple[int, int, int] = (35, 80, 80)    # 绿色下界 (H,S,V)
    hsv_upper: tuple[int, int, int] = (85, 255, 255)  # 绿色上界
    min_area_fraction: float = 1e-4   # 外接矩形面积占比下限——滤掉绿字/噪点
    close_ksize: int = 3              # 形态学闭运算核：连补细框断线；<=1 关闭


def _require_cv2():
    """延迟导入 cv2，缺包给清晰错误（而非晦涩 ImportError）。"""
    try:
        import cv2  # noqa: PLC0415

        return cv2
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "绿框提取需要 opencv-python（data 依赖组）：pip install opencv-python"
        ) from exc


def _load_rgb(image: Any) -> "np.ndarray":
    """把输入规整为 uint8 RGB ndarray (HxWx3)。

    接受：PIL.Image / np.ndarray(RGB) / 图像路径 str。统一到 RGB，避免 BGR/RGB 混淆。
    """
    import numpy as np  # noqa: PLC0415

    if isinstance(image, str):
        from PIL import Image  # noqa: PLC0415

        image = Image.open(image)

    # PIL.Image（用 mode 属性鸭子判定，避免硬依赖 PIL 于类型签名）
    if hasattr(image, "convert") and hasattr(image, "size"):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    arr = np.asarray(image)
    if arr.ndim == 2:  # 灰度 → 复制三通道
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:  # RGBA → 丢 alpha
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def green_mask(image: Any, config: GreenBoxConfig | None = None) -> "np.ndarray":
    """返回绿色像素掩码 (uint8 HxW, 0/255)。

    T026 inpaint 会复用这张掩码（再膨胀以完整覆盖框线）抹除绿框——**两处共用同一阈值，
    保证"被当作框提取"的像素与"被抹掉"的像素一致**，训推同分布。
    """
    cv2 = _require_cv2()
    import numpy as np  # noqa: PLC0415

    config = config or GreenBoxConfig()
    rgb = _load_rgb(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(config.hsv_lower, np.uint8),
                       np.array(config.hsv_upper, np.uint8))
    if config.close_ksize and config.close_ksize > 1:
        kernel = np.ones((config.close_ksize, config.close_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_green_boxes(image: Any, config: GreenBoxConfig | None = None) -> list[ROI]:
    """从图上的绿色矩形框提取 ROI 列表（按框面积降序）。

    流程：HSV inRange 得绿掩码 → `findContours` 取外接矩形 → 过滤噪点 → 构造 `ROI`
    （`area_fraction = 框面积 / 全图面积`，据此定 `AreaBand`；`confidence=1.0` 标注真值）。
    通常每图一个框；返回 list 以兼容多病灶。无框 → 返回 `[]`（交由上层决定）。
    """
    cv2 = _require_cv2()
    config = config or GreenBoxConfig()

    rgb = _load_rgb(image)
    h, w = rgb.shape[:2]
    total = float(h * w) or 1.0
    mask = green_mask(rgb, config)

    # findContours 跨 OpenCV 版本返回值个数不同：contours 恒为倒数第二个。
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]

    rois: list[ROI] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        frac = (bw * bh) / total
        if frac < config.min_area_fraction:
            continue  # 噪点/绿字
        bbox = BBox(x1=float(x), y1=float(y), x2=float(x + bw), y2=float(y + bh))
        rois.append(ROI(bbox=bbox, area_fraction=min(frac, 1.0), confidence=1.0))

    rois.sort(key=lambda r: r.bbox.area, reverse=True)
    return rois


__all__ = ["GreenBoxConfig", "green_mask", "extract_green_boxes"]
