"""无 PHI 外发贯查（T067 本地部分，FR-007）—— PHI 不入向量库/不进检索证据/不送外部 LLM。

端到端：含 PHI 的原始文本经 ingestion 清洗 → 入库 → 检索，断言全程无残留 PHI；QA 冲突送审前已去标识。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("chromadb")

from src.contracts.schemas import KnowledgeNode
from src.data import ingest as ig
from src.data.deid import has_phi
from src.rag import index_text as ix
from src.rag import retrieve_text as rt
from src.rag.embed_text import TextEmbedder


def _stub_encode(texts):
    return np.array([[b / 255.0 for b in hashlib.md5(t.encode()).digest()[:4]] for t in texts])


def _node(nid, text):
    return KnowledgeNode(node_id=nid, text=text, collection="b_medqa", source_ids=[nid])


def test_phi_scrubbed_through_ingest_index_retrieve():
    raw = [
        _node("n1", "患者服用阿司匹林解热镇痛 联系电话13800138000 邮箱 a@b.com"),
        _node("n2", "布洛芬抗炎 身份证 11010119900307123X"),
    ]
    # ingest 清洗（含去标识）
    res = ig.ingest_documents(raw, config=ig.IngestConfig(ner_filter=False))
    for node in res.nodes:
        assert not has_phi(node.text)                       # 入库前无 PHI

    # 入库 + 检索
    from src.rag.store import StoreConfig, VectorStore

    store = VectorStore(StoreConfig())
    store.reset("b_medqa")
    ds = ix.DocStore()
    emb = TextEmbedder(encode_fn=_stub_encode)
    ix.index_chunks(
        [__leaf(n) for n in res.nodes], store=store, embedder=emb,
        collection="b_medqa", docstore=ds,
    )
    result = rt.HybridRetriever(store, emb, "b_medqa", ds).retrieve("阿司匹林")
    for ev in result.evidence:
        assert ev.text and not has_phi(ev.text)             # 检索证据无 PHI


def __leaf(node: KnowledgeNode):
    """把清洗后的 KnowledgeNode 包成一个 leaf Chunk（最小化，直接入库）。"""
    from src.rag.chunk import Chunk

    return Chunk(chunk_id=node.node_id, text=node.text, embed_text=node.text,
                 level=0, is_leaf=True, metadata={"kind": "doc"})


def test_qa_conflict_judge_receives_no_phi():
    captured = []
    from src.data.qa_conflict import QAItem, Verdict

    items = [
        QAItem("怎么联系", "拨打13800138000咨询", "s1"),
        QAItem("怎么联系", "电话13800138000即可", "s2"),
    ]
    ig.ingest_qa(
        items, config=ig.IngestConfig(qa_conflict=True, ner_filter=False),
        judge_fn=lambda q, a: (captured.extend(a), Verdict.EQUIVALENT)[1],
    )
    assert captured and all(not has_phi(a) for a in captured)   # 送审文本无 PHI
