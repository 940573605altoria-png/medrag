"""C 图像嵌入（T031）—— 全图 + ROI 双向量，独立冻结编码器。

创新 C/B 的检索侧表征（plan.md）：同一图像出**两种向量**——
- **全图向量**：抓全局上下文；
- **ROI 向量**：把病灶区裁出来单独编码，抓小病灶判别力（呼应 B 双路、对抗背景稀释）。

ROI 来源：索引端用绿框坐标，查询端用 C 定位头输出（系统"检测器"）。**铁律**：检索用**独立冻结
编码器**（BiomedCLIP/通用 VL），**不用正在微调的模型自身 ViT 特征**——防循环耦合（同 coreset 铁律）。

裁剪 + 归一化纯逻辑本地测；真实编码器后端守卫导入（open_clip / transformers），本地用注入 `encode_fn` 桩。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from src.contracts.schemas import ROI, BBox
from src.rag.embed_text import l2_normalize

# 医学专用 CLIP（待 AutoDL 实测 BiomedCLIP vs PMC-CLIP vs 通用 VL，见 plan.md）。
DEFAULT_IMAGE_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# 后端：一批图像数组 → (n, d) 向量（未归一化）。
ImageEncodeFn = Callable[[Sequence[np.ndarray]], np.ndarray]


def crop_roi(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """按 BBox 从图像裁 ROI（HxW[xC]）；越界裁剪到边界，空区返回 0 尺寸数组。"""
    h, w = image.shape[0], image.shape[1]
    x1 = max(0, min(int(bbox.x1), w))
    x2 = max(0, min(int(bbox.x2), w))
    y1 = max(0, min(int(bbox.y1), h))
    y2 = max(0, min(int(bbox.y2), h))
    return image[y1:y2, x1:x2]


@dataclass
class ImageEmbedConfig:
    model_id: str = field(
        default_factory=lambda: os.environ.get("MEDRAG_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    )
    normalize: bool = True


@dataclass
class ImageEmbedder:
    """全图 + ROI 双向量嵌入器（冻结编码器）。`encode_fn` 可注入（测试/自定义后端）。"""

    config: ImageEmbedConfig = field(default_factory=ImageEmbedConfig)
    encode_fn: ImageEncodeFn | None = None
    _backend: ImageEncodeFn | None = field(default=None, init=False, repr=False)

    def _encoder(self) -> ImageEncodeFn:
        if self.encode_fn is not None:
            return self.encode_fn
        if self._backend is None:
            self._backend = self._load_backend()
        return self._backend

    def _load_backend(self) -> ImageEncodeFn:
        try:
            import open_clip  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "图像嵌入需要 open_clip + BiomedCLIP 权重（冻结编码器）；本地可注入 encode_fn 测。"
            ) from exc
        model, _, preprocess = open_clip.create_model_and_transforms(self.config.model_id)
        model.eval()

        def _encode(images: Sequence[np.ndarray]) -> np.ndarray:
            import torch  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415

            batch = torch.stack([preprocess(Image.fromarray(im)) for im in images])
            with torch.no_grad():
                feats = model.encode_image(batch)
            return feats.cpu().numpy()

        return _encode

    def _embed(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=float)
        vecs = np.asarray(self._encoder()(images), dtype=float)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        return l2_normalize(vecs) if self.config.normalize else vecs

    def encode_whole(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """全图向量（全局上下文）。"""
        return self._embed(images)

    def encode_rois(self, image: np.ndarray, rois: Sequence[ROI]) -> np.ndarray:
        """ROI 向量：逐 ROI 裁剪后编码（病灶判别力）。"""
        crops = [crop_roi(image, roi.bbox) for roi in rois]
        return self._embed(crops)


__all__ = [
    "DEFAULT_IMAGE_MODEL",
    "ImageEncodeFn",
    "crop_roi",
    "ImageEmbedConfig",
    "ImageEmbedder",
]
