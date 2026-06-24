"""真实 medical_qa 单测（T047）—— 检索→精排→带引用回答；pipeline 接回。

检索用内存 chroma + 桩嵌入端到端；精排注入 score_fn；草稿注入 draft_fn → 全链本地可测。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.contracts.schemas import AbstainReason
from src.rag import chunk as ck
from src.rag import index_text as ix
from src.rag import retrieve_text as rt
from src.rag.embed_text import TextEmbedder
from src.rag.rerank import RerankConfig, Reranker
from src.serve.qa import MedicalQA, build_qa_prompt


def _stub_encode(texts):
    out = []
    for t in texts:
        h = hashlib.md5(t.encode("utf-8")).digest()[:4]
        out.append([b / 255.0 for b in h])
    return np.array(out, dtype=float)


def _build_qa(*, draft_fn, score=5.0, min_score=0.5, index=True):
    """搭一套真实检索 + 注入精排/草稿的 MedicalQA。"""
    pytest.importorskip("chromadb")
    from src.rag.store import StoreConfig, VectorStore

    store = VectorStore(StoreConfig())
    store.reset("b_medqa")
    ds = ix.DocStore()
    emb = TextEmbedder(encode_fn=_stub_encode)
    if index:
        ix.index_qa("什么是阿司匹林", "一种解热镇痛药。", store=store, embedder=emb,
                    collection="b_medqa", docstore=ds)
        ix.index_qa("什么是布洛芬", "一种非甾体抗炎药。", store=store, embedder=emb,
                    collection="b_medqa", docstore=ds)
    retriever = rt.HybridRetriever(store, emb, "b_medqa", ds)
    reranker = Reranker(RerankConfig(min_score=min_score),
                        score_fn=lambda q, ps: [score for _ in ps])
    return MedicalQA(retriever=retriever, reranker=reranker, draft_fn=draft_fn)


def test_build_qa_prompt_numbers_evidence():
    from src.contracts.schemas import EvidenceItem, RetrievalResult

    res = RetrievalResult(query="阿司匹林",
                          evidence=[EvidenceItem(source_id="a", citation="c", score=1.0,
                                                 text="解热镇痛")])
    prompt = build_qa_prompt(res.query, res)
    assert "Question: 阿司匹林" in prompt and "[S1] 解热镇痛" in prompt


def test_qa_answers_with_citation():
    qa = _build_qa(draft_fn=lambda prompt: "阿司匹林是一种解热镇痛药 [S1]")
    report = qa.answer("阿司匹林")
    assert not report.abstain
    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.evidence and not f.uncertain                 # 锚到证据、非编造
    assert "[S1]" not in f.text                            # 引用标签已剥离


def test_qa_ungrounded_sentence_marked_uncertain():
    # 草稿不带引用 → assemble_report 标 uncertain（不删、不编造）
    qa = _build_qa(draft_fn=lambda prompt: "这是一句没有引用的断言")
    report = qa.answer("阿司匹林")
    assert report.findings and report.findings[0].uncertain


def test_qa_abstains_on_low_confidence():
    # 精排分低于阈值 → 拒答（LOW_CONFIDENCE），不调用草稿生成
    called = []
    qa = _build_qa(draft_fn=lambda p: called.append(1) or "x [S1]",
                   score=-5.0, min_score=0.5)
    report = qa.answer("阿司匹林")
    assert report.abstain and report.abstain_reason is AbstainReason.LOW_CONFIDENCE
    assert not called                                     # 拒答路不调 LLM


def test_qa_abstains_on_empty_store():
    qa = _build_qa(draft_fn=lambda p: "x [S1]", index=False)
    report = qa.answer("无任何索引")
    assert report.abstain and report.abstain_reason is AbstainReason.NO_EVIDENCE


def test_qa_requires_draft_or_model():
    qa = _build_qa(draft_fn=None)
    qa.draft_fn = None
    with pytest.raises(RuntimeError):
        qa.answer("阿司匹林")


# ── pipeline 接回（替换桩）─────────────────────────────────────────

def test_pipeline_routes_to_injected_qa_service():
    from src.serve.pipeline import Pipeline

    qa = _build_qa(draft_fn=lambda prompt: "解热镇痛 [S1]")
    pipe = Pipeline(qa_service=qa)
    report = pipe.answer("阿司匹林")
    assert not report.abstain and report.findings[0].evidence

    # 默认管线（无 qa_service）仍走桩，不破坏向后兼容
    stub_report = Pipeline().answer("aspirin")
    assert stub_report is not None
