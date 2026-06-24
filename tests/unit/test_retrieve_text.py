"""hybrid 检索单测（T045）—— RRF/AutoMerge 纯逻辑 + dense 端到端（内存 chroma + 桩嵌入）。

BM25 走守卫分支（rank_bm25 未装则测缺依赖报错、装了则功能跑）。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.contracts.schemas import AbstainReason, Modality
from src.rag import chunk as ck
from src.rag import index_text as ix
from src.rag import retrieve_text as rt
from src.rag.embed_text import TextEmbedder


def _stub_encode(texts):
    out = []
    for t in texts:
        h = hashlib.md5(t.encode("utf-8")).digest()[:4]
        out.append([b / 255.0 for b in h])
    return np.array(out, dtype=float)


def _embedder():
    return TextEmbedder(encode_fn=_stub_encode)


def _fresh_store():
    from src.rag.store import StoreConfig, VectorStore

    store = VectorStore(StoreConfig())
    store.reset("b_medqa")
    return store


# ── RRF（纯逻辑、手算）─────────────────────────────────────────────

def test_rrf_fuse_known_scores():
    fused = rt.rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    score = dict(fused)
    assert score["a"] == pytest.approx(1 / 60 + 1 / 61)
    assert score["b"] == pytest.approx(1 / 61 + 1 / 60)
    assert score["a"] == pytest.approx(score["b"])       # 对称命中 → 同分
    assert set(score) == {"a", "b", "c", "d"}
    assert score["a"] > score["c"]                       # 双路命中 > 单路


def test_tokenize_multilingual():
    assert rt.tokenize("Aspirin 阿司匹林") == ["aspirin", "阿", "司", "匹", "林"]


# ── AutoMerging 上浮（纯逻辑）─────────────────────────────────────

def _long_qa_docstore():
    cfg = ck.ChunkConfig(short_qa_tokens=5, child_tokens=4, min_chunk_tokens=1)
    chunks = ck.chunk_qa("q", "肝占位是异常。需要复查。", cfg=cfg)
    ds = ix.DocStore()
    ds.add_chunks(chunks)
    parent = next(c for c in chunks if not c.is_leaf)
    return ds, ck.leaves(chunks), parent


def test_automerge_merges_when_enough_children():
    ds, leaves, parent = _long_qa_docstore()
    assert len(leaves) == 2
    scored = [(leaves[0].chunk_id, 0.9), (leaves[1].chunk_id, 0.8)]
    merged = rt.automerge(scored, ds, merge_ratio=0.5)
    assert merged == [(parent.chunk_id, 0.9)]            # 上浮成父，取子块最高分


def test_automerge_keeps_leaf_below_threshold():
    ds, leaves, _ = _long_qa_docstore()
    scored = [(leaves[0].chunk_id, 0.9)]                 # 仅 1/2 子块命中
    merged = rt.automerge(scored, ds, merge_ratio=1.0)
    assert merged == scored                              # 未达阈值 → 不上浮


# ── HybridRetriever dense 路（端到端，内存 chroma）─────────────────

def _index_two_qa(store, ds, emb):
    ix.index_qa("什么是阿司匹林", "一种解热镇痛药。", store=store, embedder=emb,
                collection="b_medqa", docstore=ds)
    ix.index_qa("什么是布洛芬", "一种非甾体抗炎药。", store=store, embedder=emb,
                collection="b_medqa", docstore=ds)


def test_hybrid_dense_only_returns_normalized_evidence():
    pytest.importorskip("chromadb")
    store, ds, emb = _fresh_store(), ix.DocStore(), _embedder()
    _index_two_qa(store, ds, emb)
    retr = rt.HybridRetriever(store, emb, "b_medqa", ds)   # bm25=None → 纯 dense
    res = retr.retrieve("阿司匹林")
    assert not res.abstain
    assert len(res.evidence) >= 1
    assert res.evidence[0].score == pytest.approx(1.0)     # 归一化后 top=1
    assert all(e.modality is Modality.TEXT for e in res.evidence)
    assert all(e.text for e in res.evidence)               # 带原文（溯源）


def test_hybrid_abstains_on_empty_store():
    pytest.importorskip("chromadb")
    store, ds, emb = _fresh_store(), ix.DocStore(), _embedder()
    retr = rt.HybridRetriever(store, emb, "b_medqa", ds)
    res = retr.retrieve("无任何索引")
    assert res.abstain and res.abstain_reason is AbstainReason.NO_EVIDENCE


def test_bm25_guarded_or_functional():
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError):
            rt.BM25Index(["x"], ["some text"])
        return
    # 装了 rank_bm25：功能跑一把
    idx = rt.BM25Index(["x", "y"], ["aspirin pain relief", "ibuprofen anti inflammatory"])
    assert idx.search("aspirin", top_n=1) == ["x"]
