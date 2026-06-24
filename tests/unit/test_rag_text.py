"""US2 文本 RAG 单测（T043 父子分块 + T014 文本嵌入）—— 纯逻辑、CPU、无模型。

重依赖（llama-index / sentence-transformers）走守卫分支：本地若已装则跳过缺依赖测，功能跑在 AutoDL。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts.schemas import KnowledgeNode
from src.rag import chunk as ck
from src.rag import embed_text as et


# ── T043 父子分块 ──────────────────────────────────────────────────

def test_estimate_tokens_multilingual():
    assert ck.estimate_tokens("liver mass") == 2          # 拉丁词
    assert ck.estimate_tokens("肝占位") == 3              # CJK 各 1
    assert ck.estimate_tokens("肝 mass") == 2             # 1 CJK + 1 词


def test_chunk_qa_short_single_leaf():
    nodes = ck.chunk_qa("什么是阿司匹林", "一种解热镇痛药。",
                        cfg=ck.ChunkConfig(short_qa_tokens=50))
    assert len(nodes) == 1
    leaf = nodes[0]
    assert leaf.is_leaf and leaf.parent_id is None
    assert leaf.embed_text.startswith("什么是阿司匹林")   # contextual 前置 question
    assert leaf.text == "一种解热镇痛药。"                # 展示文本无前置


def test_chunk_qa_long_parent_children():
    cfg = ck.ChunkConfig(short_qa_tokens=5, child_tokens=8, contextual=True)
    answer = "肝占位是肝内异常区域。它可能良性或恶性。需要进一步检查确认诊断。"
    nodes = ck.chunk_qa("什么是肝占位", answer, doc_id="d1", cfg=cfg)
    parents = [n for n in nodes if not n.is_leaf]
    children = ck.leaves(nodes)
    assert len(parents) == 1
    assert parents[0].text == answer                      # 父=整段原文（无损）
    assert len(children) >= 1
    for c in children:
        assert c.parent_id == parents[0].chunk_id
        assert c.embed_text.startswith("什么是肝占位")    # 子块 embed 前置 question


def test_chunk_qa_contextual_off():
    nodes = ck.chunk_qa("q", "短答案。", cfg=ck.ChunkConfig(contextual=False))
    assert nodes[0].embed_text == nodes[0].text           # 关掉 → 不前置


def test_to_knowledge_nodes_maps_leaves_with_embed_text():
    cfg = ck.ChunkConfig(short_qa_tokens=5, child_tokens=8)
    nodes = ck.chunk_qa("什么是肝占位", "肝占位是肝内异常区域。需要进一步检查确认。",
                        cfg=cfg)
    kn = ck.to_knowledge_nodes(nodes, collection="b_medqa", lang="zh", source_ids=["s1"])
    assert all(isinstance(n, KnowledgeNode) for n in kn)
    assert all(n.collection == "b_medqa" for n in kn)
    assert all("embed_text" in n.metadata for n in kn)    # contextual 文本入 metadata
    assert all("parent_id" in n.metadata for n in kn)     # 子块带父引用


def test_hierarchical_documents_guarded():
    try:
        import llama_index.core  # noqa: F401
        pytest.skip("llama-index 已装，跳过缺依赖分支")
    except ImportError:
        with pytest.raises(RuntimeError):
            ck.hierarchical_chunk_documents(["a b c d e."])


# ── T014 文本嵌入 ──────────────────────────────────────────────────

def test_format_query_asymmetric():
    f = et.format_query("肝占位")
    assert f.startswith("Instruct:") and "Query: 肝占位" in f
    assert et.format_document("一段文档") == "一段文档"     # document 不加指令


def test_l2_normalize():
    v = et.l2_normalize(np.array([[3.0, 4.0]]))
    assert v[0] == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(et.l2_normalize(np.array([[0.0, 0.0]]))) == pytest.approx(0.0)


def test_embedder_query_vs_document_formatting():
    captured: list[str] = []

    def stub(texts):
        captured.extend(texts)
        return np.array([[3.0, 4.0] for _ in texts])

    emb = et.TextEmbedder(encode_fn=stub)
    q = emb.encode_queries(["liver"])
    assert captured[0].startswith("Instruct:") and "liver" in captured[0]
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)    # 归一化

    captured.clear()
    emb.encode_documents(["a doc"])
    assert captured[0] == "a doc"                         # 非对称：文档端无指令


def test_embedder_normalize_off():
    emb = et.TextEmbedder(config=et.EmbedConfig(normalize=False),
                          encode_fn=lambda t: np.array([[3.0, 4.0]]))
    assert emb.encode_documents(["x"])[0] == pytest.approx([3.0, 4.0])


def test_embed_config_env(monkeypatch):
    monkeypatch.delenv("MEDRAG_EMBED_MODEL", raising=False)
    assert et.EmbedConfig().model_id == et.DEFAULT_MODEL_ID
    monkeypatch.setenv("MEDRAG_EMBED_MODEL", "org/custom-embed")
    assert et.EmbedConfig().model_id == "org/custom-embed"


def test_embedder_guarded_without_backend():
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers 已装，跳过缺依赖分支")
    except ImportError:
        with pytest.raises(RuntimeError):
            et.TextEmbedder().encode_documents(["x"])
