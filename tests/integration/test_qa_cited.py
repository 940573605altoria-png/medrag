"""US2 集成测试（T048）—— 药品/医学问题 → 带引用且**不超出来源**。

串起真实 US2 链：分块(T043)→入库(T044)→hybrid检索(T045)→精排+拒答(T046)→medical_qa(T047)。
用内存 chroma + 注入桩嵌入/精排/草稿，端到端验证 constitution I：每条结论锚到**已索引的来源**、
无据标 uncertain（不编造来源）、无关问题拒答。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("chromadb")

from src.contracts.schemas import AbstainReason
from src.rag import index_text as ix
from src.rag import retrieve_text as rt
from src.rag.embed_text import TextEmbedder
from src.rag.rerank import RerankConfig, Reranker
from src.serve.qa import MedicalQA


def _stub_encode(texts):
    out = []
    for t in texts:
        h = hashlib.md5(t.encode("utf-8")).digest()[:4]
        out.append([b / 255.0 for b in h])
    return np.array(out, dtype=float)


def _build_qa(pairs, *, draft_fn, rerank_score=5.0, min_score=0.5):
    """搭真实检索/入库 + 注入精排/草稿的 MedicalQA，返回 (qa, docstore)。"""
    from src.rag.store import StoreConfig, VectorStore

    store = VectorStore(StoreConfig())
    store.reset("b_medqa")
    ds = ix.DocStore()
    emb = TextEmbedder(encode_fn=_stub_encode)
    for q, a, src in pairs:
        ix.index_qa(q, a, store=store, embedder=emb, collection="b_medqa",
                    docstore=ds, lang="zh", source_ids=[src])
    retriever = rt.HybridRetriever(store, emb, "b_medqa", ds)
    reranker = Reranker(RerankConfig(min_score=min_score),
                        score_fn=lambda q, ps: [rerank_score for _ in ps])
    return MedicalQA(retriever=retriever, reranker=reranker, draft_fn=draft_fn), ds


_PAIRS = [
    ("阿司匹林的作用是什么", "阿司匹林用于解热镇痛。", "src-aspirin"),
    ("布洛芬的作用是什么", "布洛芬是非甾体抗炎药。", "src-ibuprofen"),
]


def test_drug_question_returns_cited_answer():
    qa, ds = _build_qa(_PAIRS, draft_fn=lambda prompt: "该药用于解热镇痛 [S1]")
    report = qa.answer("阿司匹林")
    assert not report.abstain
    assert report.findings
    f = report.findings[0]
    assert f.evidence and not f.uncertain          # 锚到证据、非编造
    assert "[S1]" not in f.text                     # 引用标签已剥离


def test_answer_does_not_exceed_sources():
    qa, ds = _build_qa(_PAIRS, draft_fn=lambda prompt: "该药用于解热镇痛 [S1]")
    report = qa.answer("阿司匹林")
    # 每条结论引用的证据都必须是**已索引的真实节点**（无凭空来源）
    for finding in report.findings:
        for ev in finding.evidence:
            assert ds.get(ev.source_id) is not None
            assert ev.text                          # 带原文，可回指


def test_fabricated_citation_becomes_uncertain_not_invented_source():
    # 草稿引用了不存在的 [S99] → 该结论标 uncertain，绝不挂凭空来源
    qa, _ = _build_qa(_PAIRS, draft_fn=lambda prompt: "凭空断言 [S99]")
    report = qa.answer("阿司匹林")
    assert report.findings
    assert all(not f.evidence for f in report.findings)   # 没有捏造证据
    assert any(f.uncertain for f in report.findings)


def test_out_of_scope_abstains_no_evidence():
    qa, _ = _build_qa([], draft_fn=lambda p: "x [S1]")     # 空库
    report = qa.answer("无关问题")
    assert report.abstain and report.abstain_reason is AbstainReason.NO_EVIDENCE


def test_low_confidence_abstains_without_calling_llm():
    called = []
    qa, _ = _build_qa(_PAIRS, draft_fn=lambda p: called.append(1) or "x [S1]",
                      rerank_score=-5.0)                    # 精排分低于门
    report = qa.answer("阿司匹林")
    assert report.abstain and report.abstain_reason is AbstainReason.LOW_CONFIDENCE
    assert not called                                       # 拒答不调 LLM


def test_pipeline_and_mcp_wrapping():
    from src.serve.mcp_server import _wrap
    from src.serve.pipeline import Pipeline

    qa, _ = _build_qa(_PAIRS, draft_fn=lambda prompt: "用于解热镇痛 [S1]")
    report = Pipeline(qa_service=qa).answer("阿司匹林")     # 经管线路由
    assert not report.abstain and report.findings[0].evidence
    # MCP 包络：非拒答 → status ok，payload 带证据
    tool_io = _wrap("medical_qa", report, abstain=report.abstain)
    assert tool_io.status.value == "ok"
    assert tool_io.payload["findings"][0]["evidence"]
