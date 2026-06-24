"""精排 + 拒答门单测（T046）—— 排序/拒答纯逻辑；reranker 后端守卫。"""

from __future__ import annotations

import math

import pytest

from src.contracts.schemas import AbstainReason, EvidenceItem, RetrievalResult
from src.rag import rerank as rr


def _result(texts, query="aspirin"):
    ev = [EvidenceItem(source_id=t, citation=f"c-{t}", score=0.5, text=t) for t in texts]
    return RetrievalResult(query=query, evidence=ev)


def test_sigmoid():
    assert rr._sigmoid(0.0) == pytest.approx(0.5)
    assert rr._sigmoid(100) == pytest.approx(1.0)
    assert rr._sigmoid(-100) == pytest.approx(0.0)


def test_rerank_reorders_by_cross_encoder_score():
    res = _result(["a", "b", "c"])
    # 反序打分：a→1, b→2, c→3 → 重排应 c, b, a
    score_fn = lambda q, ps: [float(i + 1) for i, _ in enumerate(ps)]
    out = rr.Reranker(rr.RerankConfig(min_score=0.0), score_fn=score_fn).rerank(res)
    assert [e.source_id for e in out.evidence] == ["c", "b", "a"]
    assert out.evidence[0].score == pytest.approx(rr._sigmoid(3.0))  # 统一分=sigmoid
    assert not out.abstain


def test_rerank_top_k_truncates():
    res = _result(["a", "b", "c", "d", "e", "f"])
    score_fn = lambda q, ps: [float(i) for i, _ in enumerate(ps)]
    out = rr.Reranker(rr.RerankConfig(top_k=2, min_score=0.0), score_fn=score_fn).rerank(res)
    assert len(out.evidence) == 2


def test_rerank_abstains_below_threshold():
    res = _result(["a", "b"])
    # 全给 0 logit → sigmoid 0.5；阈值 0.9 → 拒答
    score_fn = lambda q, ps: [0.0 for _ in ps]
    out = rr.Reranker(rr.RerankConfig(min_score=0.9), score_fn=score_fn).rerank(res)
    assert out.abstain and out.abstain_reason is AbstainReason.LOW_CONFIDENCE
    assert out.evidence                                   # 仍带 top 候选供展示


def test_rerank_passthrough_when_already_abstained():
    res = RetrievalResult(query="q", evidence=[], abstain=True,
                          abstain_reason=AbstainReason.NO_EVIDENCE)
    out = rr.Reranker(score_fn=lambda q, ps: []).rerank(res)
    assert out is res                                     # 上游已拒答 → 原样透传


def test_rerank_score_count_mismatch_raises():
    res = _result(["a", "b"])
    with pytest.raises(ValueError):
        rr.Reranker(score_fn=lambda q, ps: [1.0]).rerank(res)


def test_rerank_config_env(monkeypatch):
    monkeypatch.delenv("MEDRAG_RERANKER_MODEL", raising=False)
    assert rr.RerankConfig().model_id == rr.DEFAULT_RERANKER_ID
    monkeypatch.setenv("MEDRAG_RERANKER_MODEL", "org/custom-reranker")
    assert rr.RerankConfig().model_id == "org/custom-reranker"


def test_reranker_guarded_without_backend():
    res = _result(["a"])
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers 已装，跳过缺依赖分支")
    except ImportError:
        with pytest.raises(RuntimeError):
            rr.Reranker().rerank(res)
