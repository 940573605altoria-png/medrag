"""文本归一化 + 去重单测（T038/T039）—— 纯逻辑、本地全测（datasketch 路守卫）。"""

from __future__ import annotations

import pytest

from src.contracts.schemas import KnowledgeNode
from src.data import dedup as dd
from src.data.normalize import NormalizeConfig, normalize_text


def _node(node_id, text, *, source_ids=None, **meta):
    return KnowledgeNode(node_id=node_id, text=text, collection="b_medqa",
                         source_ids=source_ids or [], metadata=meta)


# ── T038 归一化 ───────────────────────────────────────────────────

def test_nfkc_fullwidth_to_halfwidth():
    assert normalize_text("５ｍｇ") == "5mg"               # 全角→半角，数值不变
    assert normalize_text("（注）") == "(注)"


def test_medical_dose_preserved_and_distinct():
    assert normalize_text("5mg") == "5mg"
    assert normalize_text("5mg") != normalize_text("50mg")  # 剂量绝不归一
    assert normalize_text("5mg") != normalize_text("5mL")   # 单位绝不归一


def test_collapse_whitespace_and_punct():
    assert normalize_text("a    b\tc") == "a b c"
    assert normalize_text("好的。。。") == "好的。"
    assert normalize_text("wow!!!") == "wow!"


def test_strip_boilerplate():
    text = "免责声明：本文仅供参考\n阿司匹林是解热镇痛药\n第 3 页"
    out = normalize_text(text)
    assert "阿司匹林是解热镇痛药" in out
    assert "免责声明" not in out and "第 3 页" not in out


def test_normalize_idempotent_and_switchable():
    s = "５ｍｇ    test。。。"
    assert normalize_text(normalize_text(s)) == normalize_text(s)
    # 关掉 nfkc → 保留全角
    assert "５" in normalize_text("５mg", NormalizeConfig(nfkc=False))


# ── T039 去重 ─────────────────────────────────────────────────────

def test_sha1_jaccard_shingles():
    assert dd.sha1_key("x") == dd.sha1_key("x")
    assert dd.jaccard(set(), set()) == 1.0
    assert dd.jaccard({"a", "b"}, {"a"}) == pytest.approx(0.5)
    assert dd.shingles("a b c d", n=3) == {"a b c", "b c d"}


def test_exact_dedup_merges_sources():
    # 文本归一化后相同（仅空白差异）→ 合并，来源并入存活节点
    a = _node("a", "阿司匹林 是 解热 镇痛 药", source_ids=["sa"])
    b = _node("b", "阿司匹林   是  解热 镇痛 药", source_ids=["sb"])
    c = _node("c", "布洛芬 是 抗炎 药", source_ids=["sc"])
    res = dd.exact_dedup([a, b, c])
    assert len(res.kept) == 2 and res.removed_ids == ["b"]
    surv = next(n for n in res.kept if n.node_id == "a")
    assert "sb" in surv.source_ids and "b" in surv.source_ids   # 溯源不丢
    assert "dedup" in surv.flags


def test_structured_dedup_by_metadata_key():
    a = _node("a", "section text 1", drug="阿司匹林", section="适应症")
    b = _node("b", "section text 2", drug="阿司匹林", section="适应症")  # 同键
    c = _node("c", "section text 3", drug="阿司匹林", section="禁忌")
    res = dd.structured_dedup([a, b, c], key_fields=["drug", "section"])
    assert len(res.kept) == 2 and "b" in res.removed_ids


def test_near_dedup_bruteforce_merges_similar_keeps_distinct():
    a = _node("a", "a b c d e f g h", source_ids=["sa"])
    b = _node("b", "a b c d e f g x", source_ids=["sb"])  # 与 a 仅末词不同
    c = _node("c", "p q r s t u v w")
    res = dd.near_dedup([a, b, c], threshold=0.7, n=3, backend="bruteforce")
    assert len(res.kept) == 2                              # a,b 合并；c 保留
    assert "b" in res.removed_ids
    surv = next(n for n in res.kept if n.node_id == "a")
    assert "sb" in surv.source_ids


def test_near_dedup_conservative_keeps_moderately_different():
    a = _node("a", "a b c d e f g h")
    b = _node("b", "a b c x y z w v")                     # 差异大
    res = dd.near_dedup([a, b], threshold=0.85, n=3, backend="bruteforce")
    assert len(res.kept) == 2                              # 保守阈值不误删


def test_near_dedup_minhash_guarded():
    nodes = [_node("a", "a b c d e f"), _node("b", "a b c d e f")]
    try:
        import datasketch  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError):
            dd.near_dedup(nodes, backend="minhash")
        return
    res = dd.near_dedup(nodes, threshold=0.7, backend="minhash")
    assert len(res.kept) <= len(nodes)
