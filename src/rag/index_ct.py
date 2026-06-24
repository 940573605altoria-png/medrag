"""c 病例多向量入库（T032）—— 同一 case_id 串 报告文本 + 全图 + ROI 三类向量。

CT 图文病例检索（plan.md）：一个病例存**多向量**到三个 collection，`case_id` 串起来——
- `c_text`      报告文本向量（Qwen3-Embedding，供级联文本过滤）
- `c_img_whole` 全图向量（全局上下文）
- `c_img_roi`   ROI/病灶区向量（小病灶判别力）

图/文查询都能命中同一病例（级联检索 T033 据此跨通道融合）。文本走 [embed_text]、图像走 [embed_image]
（**各自空间、不混**，分治）。chromadb 本地已装→入库本地功能测；真实编码器在 AutoDL。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.contracts.schemas import CTSample
from src.rag.embed_image import ImageEmbedder
from src.rag.embed_text import TextEmbedder
from src.rag.store import VectorStore


@dataclass
class CTIndexResult:
    case_id: str
    text: int = 0
    whole: int = 0
    roi: int = 0


def index_ct_case(
    case: CTSample,
    image: np.ndarray | None,
    *,
    store: VectorStore,
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
) -> CTIndexResult:
    """把一个病例的报告文本/全图/ROI 编码入对应 collection（id 前缀 = case_id）。"""
    cid = case.case_id
    res = CTIndexResult(case_id=cid)

    if case.report_text and text_embedder is not None:
        v = text_embedder.encode_documents([case.report_text])
        store.add("c_text", ids=[f"{cid}:text"], embeddings=v.tolist(),
                  metadatas=[{"case_id": cid}], documents=[case.report_text])
        res.text = 1

    if image is not None and image_embedder is not None:
        wv = image_embedder.encode_whole([image])
        store.add("c_img_whole", ids=[f"{cid}:whole"], embeddings=wv.tolist(),
                  metadatas=[{"case_id": cid}])
        res.whole = 1

        if case.rois:
            rv = image_embedder.encode_rois(image, case.rois)
            ids = [f"{cid}:roi{i}" for i in range(len(case.rois))]
            metas = [{"case_id": cid, "area_band": roi.area_band.value} for roi in case.rois]
            store.add("c_img_roi", ids=ids, embeddings=rv.tolist(), metadatas=metas)
            res.roi = len(case.rois)

    return res


def index_ct_cases(
    cases: Sequence[tuple[CTSample, np.ndarray | None]],
    *,
    store: VectorStore,
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
) -> list[CTIndexResult]:
    """批量入库（跳过 in_eval 的样本：固定评估集 never-touched 守卫）。"""
    out: list[CTIndexResult] = []
    for case, image in cases:
        if case.in_eval_set:           # 固定评估集 never-touched 守卫
            continue
        out.append(index_ct_case(case, image, store=store,
                                 text_embedder=text_embedder, image_embedder=image_embedder))
    return out


__all__ = ["CTIndexResult", "index_ct_case", "index_ct_cases"]
