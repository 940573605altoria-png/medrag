"""视觉 RAG 链单测（T031/T032/T033）—— 裁剪/双向量 + 多向量入库 + 级联，桩编码器本地测。"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.contracts.schemas import BBox, CTSample, Modality, ROI
from src.rag import cascade_visual as cv
from src.rag import embed_image as ei
from src.rag import index_ct as ic
from src.rag.embed_image import ImageEmbedder
from src.rag.embed_text import TextEmbedder


def _img_encode(images):
    return np.array([[b / 255.0 for b in hashlib.md5(im.tobytes()).digest()[:4]]
                     for im in images], dtype=float)


def _txt_encode(texts):
    return np.array([[b / 255.0 for b in hashlib.md5(t.encode()).digest()[:4]]
                     for t in texts], dtype=float)


def _roi(x1, y1, x2, y2, frac=0.01):
    return ROI(bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), area_fraction=frac)


# ── T031 图像嵌入 ─────────────────────────────────────────────────

def test_crop_roi_shape_and_clamp():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert ei.crop_roi(img, BBox(x1=10, y1=20, x2=30, y2=50)).shape[:2] == (30, 20)
    assert ei.crop_roi(img, BBox(x1=10, y1=0, x2=200, y2=10)).shape[:2] == (10, 90)  # 越界裁到边界


def test_image_embedder_whole_and_roi():
    emb = ImageEmbedder(encode_fn=lambda ims: np.array([[3.0, 4.0] for _ in ims]))
    w = emb.encode_whole([np.zeros((8, 8, 3), np.uint8)])
    assert np.allclose(np.linalg.norm(w, axis=1), 1.0)              # 归一化
    img = np.zeros((20, 20, 3), np.uint8)
    rv = emb.encode_rois(img, [_roi(0, 0, 10, 10), _roi(5, 5, 15, 15)])
    assert rv.shape[0] == 2                                          # 每 ROI 一条


def test_image_embedder_guarded():
    try:
        import open_clip  # noqa: F401
        pytest.skip("open_clip 已装，跳过缺依赖分支")
    except ImportError:
        with pytest.raises(RuntimeError):
            ImageEmbedder().encode_whole([np.zeros((4, 4, 3), np.uint8)])


# ── T032 多向量入库 ───────────────────────────────────────────────

def _fresh_store():
    from src.rag.store import StoreConfig, VectorStore

    store = VectorStore(StoreConfig())
    for c in ("c_text", "c_img_whole", "c_img_roi"):
        store.reset(c)
    return store


def _case(cid, **kw):
    return CTSample(case_id=cid, image_path=f"{cid}.png", **kw)


def test_index_ct_case_multivector():
    pytest.importorskip("chromadb")
    store = _fresh_store()
    img = np.full((8, 8, 3), 7, np.uint8)
    case = _case("A", report_text="肝占位报告", rois=[_roi(0, 0, 4, 4), _roi(2, 2, 6, 6)])
    res = ic.index_ct_case(case, img, store=store,
                           text_embedder=TextEmbedder(encode_fn=_txt_encode),
                           image_embedder=ImageEmbedder(encode_fn=_img_encode))
    assert (res.text, res.whole, res.roi) == (1, 1, 2)
    assert store.count("c_text") == 1 and store.count("c_img_roi") == 2


def test_index_ct_cases_skips_eval_set():
    pytest.importorskip("chromadb")
    store = _fresh_store()
    img = np.zeros((8, 8, 3), np.uint8)
    cases = [(_case("A"), img), (_case("B", in_eval_set=True), img)]  # B 属固定评估集
    out = ic.index_ct_cases(cases, store=store,
                            image_embedder=ImageEmbedder(encode_fn=_img_encode))
    assert [r.case_id for r in out] == ["A"]                       # never-touched 守卫


# ── T033 视觉级联 ─────────────────────────────────────────────────

def test_visual_cascade_retrieves_case_and_abstains():
    pytest.importorskip("chromadb")
    store = _fresh_store()
    img_emb = ImageEmbedder(encode_fn=_img_encode)
    txt_emb = TextEmbedder(encode_fn=_txt_encode)
    imgA = np.full((8, 8, 3), 1, np.uint8)
    imgB = np.full((8, 8, 3), 2, np.uint8)
    ic.index_ct_case(_case("A", report_text="肝占位", rois=[_roi(0, 0, 4, 4)]), imgA,
                     store=store, text_embedder=txt_emb, image_embedder=img_emb)
    ic.index_ct_case(_case("B", report_text="肺结节"), imgB,
                     store=store, text_embedder=txt_emb, image_embedder=img_emb)

    vc = cv.VisualCascade(store=store, image_embedder=img_emb, text_embedder=txt_emb)
    res = vc.retrieve(imgA, draft_text="肝占位")
    assert not res.abstain
    assert res.evidence[0].case_id == "A"                          # 命中正确病例
    assert res.evidence[0].modality is Modality.IMAGE
    assert res.evidence[0].score == pytest.approx(1.0)            # 归一化 top

    # 空查询 → 拒答
    empty = cv.VisualCascade(store=_fresh_store(), image_embedder=img_emb)
    assert empty.retrieve(None).abstain
