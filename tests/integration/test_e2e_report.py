"""US1 端到端报告集成测试（T037）—— 干净 CT → 带来源报告 + 图像兜底通道（FR-005）。

串 检测(ROI) → 视觉级联检索(T033) → 报告生成(T034)：验证每条结论锚到证据/ROI、可溯源；并断言
**检测草稿空/缺时，图像主通道仍能兜底召回**（FR-005）。检索/报告用注入桩，结构本地端到端测。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("chromadb")

from src.contracts.schemas import BBox, CTSample, DetectionResult, Modality, ROI
from src.models.report import ReportGenerator
from src.rag import cascade_visual as cv
from src.rag import index_ct as ic
from src.rag.embed_image import ImageEmbedder
from src.rag.embed_text import TextEmbedder


def _img_encode(images):
    return np.array([[b / 255.0 for b in hashlib.md5(im.tobytes()).digest()[:4]]
                     for im in images], dtype=float)


def _txt_encode(texts):
    return np.array([[b / 255.0 for b in hashlib.md5(t.encode()).digest()[:4]]
                     for t in texts], dtype=float)


def _roi(x1, y1, x2, y2, frac=0.015):
    return ROI(bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), area_fraction=frac)


def _fresh_store():
    from src.rag.store import StoreConfig, VectorStore

    store = VectorStore(StoreConfig())
    for c in ("c_text", "c_img_whole", "c_img_roi"):
        store.reset(c)
    return store


def _cascade_with_cases():
    store = _fresh_store()
    img_emb = ImageEmbedder(encode_fn=_img_encode)
    txt_emb = TextEmbedder(encode_fn=_txt_encode)
    img = np.full((8, 8, 3), 1, np.uint8)
    ic.index_ct_case(CTSample(case_id="case1", image_path="c1.png", report_text="肝占位",
                              rois=[_roi(0, 0, 4, 4)]), img,
                     store=store, text_embedder=txt_emb, image_embedder=img_emb)
    return cv.VisualCascade(store=store, image_embedder=img_emb, text_embedder=txt_emb), img


def test_clean_ct_to_cited_report():
    cascade, img = _cascade_with_cases()
    detection = DetectionResult(rois=[_roi(0, 0, 4, 4)], image_id="q")  # 检测出小病灶
    retrieval = cascade.retrieve(img, draft_text="肝占位")
    report = ReportGenerator().generate(
        detection, retrieval,
        draft_fn=lambda prompt, image: "可疑小病灶，结合相似病例 [S1][ROI1]",
    )
    assert not report.abstain
    f = report.findings[0]
    assert f.evidence and f.roi is not None          # 同时锚检索证据 + 定位 ROI
    assert f.evidence[0].modality is Modality.IMAGE


def test_image_fallback_when_detection_draft_empty():
    # FR-005：检测草稿为空（draft_text=None）时，图像主通道仍兜底召回
    cascade, img = _cascade_with_cases()
    res = cascade.retrieve(img, draft_text=None)
    assert not res.abstain and res.evidence[0].case_id == "case1"


def test_abstains_when_no_detection_and_no_image():
    cascade, _ = _cascade_with_cases()
    detection = DetectionResult(abstained=True)         # 无病灶
    retrieval = cascade.retrieve(None, draft_text=None)  # 无图无草稿
    report = ReportGenerator().generate(
        detection, retrieval, draft_fn=lambda p, i: "不应被调用",
    )
    assert report.abstain                                # 整体无据 → 拒答
