"""文本入库单测（T044）—— DocStore 纯逻辑 + 入库端到端（内存 chroma + 桩嵌入器）。

chromadb 本地已装，故入库可端到端功能测；嵌入器用确定性桩（同文本→同向量），真实模型在 AutoDL。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.rag import chunk as ck
from src.rag import index_text as ix
from src.rag.embed_text import TextEmbedder


def _stub_encode(texts):
    """确定性桩嵌入：md5 前 4 字节归一到 [0,1]，同文本→同向量。"""
    out = []
    for t in texts:
        h = hashlib.md5(t.encode("utf-8")).digest()[:4]
        out.append([b / 255.0 for b in h])
    return np.array(out, dtype=float)


def _embedder():
    return TextEmbedder(encode_fn=_stub_encode)


def _fresh_store():
    """内存 store；chromadb 的 EphemeralClient 进程内共享，故先 reset 保证测试隔离。"""
    from src.rag.store import StoreConfig, VectorStore

    store = VectorStore(StoreConfig())
    store.reset("b_medqa")
    return store


# ── DocStore（纯逻辑）─────────────────────────────────────────────

def test_docstore_parent_child_lookup():
    cfg = ck.ChunkConfig(short_qa_tokens=5, child_tokens=8)
    chunks = ck.chunk_qa("什么是肝占位", "肝占位是肝内异常区域。需要进一步检查确认。", cfg=cfg)
    ds = ix.DocStore()
    ds.add_chunks(chunks)
    leaves = ck.leaves(chunks)
    parent = next(c for c in chunks if not c.is_leaf)
    assert len(ds) == len(chunks)
    assert ds.parent_of(leaves[0].chunk_id).node_id == parent.chunk_id
    assert {c.node_id for c in ds.children_of(parent.chunk_id)} == {l.chunk_id for l in leaves}


def test_docstore_save_load_roundtrip(tmp_path):
    chunks = ck.chunk_qa("q", "短答案。", cfg=ck.ChunkConfig())
    ds = ix.DocStore()
    ds.add_chunks(chunks)
    path = ds.save(tmp_path / "docstore.json")
    back = ix.DocStore.load(path)
    assert len(back) == len(ds)
    cid = chunks[0].chunk_id
    assert back.get(cid).text == ds.get(cid).text


# ── 入库端到端（内存 chroma）───────────────────────────────────────

def test_index_qa_short_indexes_one_leaf():
    pytest.importorskip("chromadb")
    store = _fresh_store()
    ds = ix.DocStore()
    res = ix.index_qa("什么是阿司匹林", "一种解热镇痛药。",
                      store=store, embedder=_embedder(), collection="b_medqa",
                      docstore=ds, lang="zh", source_ids=["s1"])
    assert res.n_leaves_indexed == 1
    assert store.count("b_medqa") == 1
    assert len(ds) == 1


def test_index_qa_long_indexes_all_leaves_parent_in_docstore_only():
    pytest.importorskip("chromadb")
    store = _fresh_store()
    ds = ix.DocStore()
    cfg = ck.ChunkConfig(short_qa_tokens=5, child_tokens=8)
    res = ix.index_qa("什么是肝占位", "肝占位是肝内异常区域。它可能良性或恶性。需复查确认。",
                      store=store, embedder=_embedder(), collection="b_medqa",
                      docstore=ds, cfg=cfg)
    # 向量库只装叶块；docstore 装全部（含 1 个父）
    assert store.count("b_medqa") == res.n_leaves_indexed
    assert len(ds) == res.n_nodes_stored == res.n_leaves_indexed + 1


def test_index_then_query_returns_indexed_leaf():
    pytest.importorskip("chromadb")
    store = _fresh_store()
    ds = ix.DocStore()
    emb = _embedder()
    chunks = ck.chunk_qa("什么是阿司匹林", "一种解热镇痛药。", cfg=ck.ChunkConfig())
    ix.index_chunks(chunks, store=store, embedder=emb, collection="b_medqa", docstore=ds)
    leaf = ck.leaves(chunks)[0]
    # 用同一桩对叶块 embed_text 求向量去查 → 应命中该叶块
    q = emb.encode_documents([leaf.embed_text])
    res = store.query("b_medqa", q.tolist(), n_results=1)
    assert leaf.chunk_id in res["ids"][0]


def test_index_rejects_unknown_collection():
    pytest.importorskip("chromadb")
    store = _fresh_store()
    with pytest.raises(ValueError):
        ix.index_qa("q", "a。", store=store, embedder=_embedder(),
                    collection="bad_name", docstore=ix.DocStore())
