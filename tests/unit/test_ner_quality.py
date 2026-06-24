"""双语 NER + 质量筛单测（T041/T042）—— 纯逻辑、可注入后端，本地全测。"""

from __future__ import annotations

import pytest

from src.contracts.schemas import KnowledgeNode
from src.data import ner as N
from src.data import ner_quality as nq
from src.data.ner import Entity, MedicalNER


def _node(node_id, text):
    return KnowledgeNode(node_id=node_id, text=text, collection="b_medqa")


# ── T041 NER ──────────────────────────────────────────────────────

def test_detect_language():
    assert N.detect_language("肝占位异常") == "zh"
    assert N.detect_language("liver mass found") == "en"
    assert N.detect_language("阿司匹林 aspirin 用于") == "zh"   # CJK 多于拉丁


def test_map_label_bilingual():
    assert N.map_label("CHEMICAL", "en") == N.DRUG
    assert N.map_label("Disease", "en") == N.DISEASE
    assert N.map_label("dru", "zh") == N.DRUG
    assert N.map_label("bod", "zh") == N.ANATOMY
    assert N.map_label("???", "en") == N.OTHER             # 未知 → OTHER


def test_medical_ner_dispatch_and_unify():
    ner = MedicalNER(
        en_backend=lambda t: [("aspirin", "CHEMICAL", 0, 7)],
        zh_backend=lambda t: [("阿司匹林", "dru", 0, 4)],
    )
    en = ner.extract("aspirin relieves pain")
    assert en[0].label == N.DRUG and en[0].lang == "en"
    zh = ner.extract("阿司匹林用于解热")
    assert zh[0].label == N.DRUG and zh[0].lang == "zh"


def test_as_entity_extractor_plugs_into_report_f1():
    from src.eval.metrics_report import report_entity_f1

    ner = MedicalNER(en_backend=lambda t: [("aspirin", "CHEMICAL", 0, 7)])
    fn = N.as_entity_extractor(ner)
    r = report_entity_f1([("aspirin", "aspirin")], ner_fn=fn)
    assert r.f1 == pytest.approx(1.0)                       # T050 解耦点闭环


def test_ner_backend_guarded():
    ner = MedicalNER()  # 无注入后端
    try:
        import spacy  # noqa: F401
        pytest.skip("spaCy 已装，跳过缺依赖分支")
    except ImportError:
        with pytest.raises(RuntimeError):
            ner.extract("liver mass")


# ── T042 质量筛 ───────────────────────────────────────────────────

def test_compute_signals_values():
    ents = [Entity("X", N.DRUG, cui="C1"), Entity("Y", N.DISEASE)]
    sig = nq.compute_signals("a b c d", ents, df={"x": 1}, n_docs=10)
    assert sig.density == pytest.approx(0.5)               # 2 实体 / 4 token
    assert sig.linkable_rate == pytest.approx(0.5)         # 1/2 有 cui
    assert sig.rare_rate == pytest.approx(0.5)             # y 不在 df → 稀有
    assert not sig.low_coverage and sig.score == pytest.approx(0.7)


def test_filter_zero_entity_hard_drop_and_low_coverage_soft():
    n0 = _node("n0", "纯叙述无实体")
    n1 = _node("n1", "aspirin ibuprofen pain")             # 高密度
    n2 = _node("n2", " ".join(["word"] * 100))             # 100 token、1 实体 → 低覆盖
    annotated = [
        (n0, []),
        (n1, [Entity("aspirin", N.DRUG), Entity("ibuprofen", N.DRUG)]),
        (n2, [Entity("aspirin", N.DRUG)]),
    ]
    res = nq.filter_nodes(annotated)
    assert res.dropped_ids == ["n0"]                       # 零实体硬丢
    assert {n.node_id for n in res.kept} == {"n1", "n2"}
    assert "low_coverage" in next(n for n in res.kept if n.node_id == "n2").flags
    assert n1.quality_score is not None and n1.entities    # 写回质量分+实体


def test_document_frequency_excludes_eval():
    a = [
        (_node("a", "t"), [Entity("x", N.DRUG)]),
        (_node("b", "t"), [Entity("x", N.DRUG), Entity("y", N.DISEASE)]),
        (_node("c", "t"), [Entity("z", N.DRUG)]),          # 评估集 → 排除
    ]
    df = nq.document_frequency(a, eval_mask=[False, False, True])
    assert df == {"x": 2, "y": 1}


def test_coverage_guard_catches_lost_unique_entity():
    nA, nB = _node("A", "t"), _node("B", "t")
    before = [(nA, [Entity("aspirin", N.DRUG)]), (nB, [Entity("rareonly", N.DRUG)])]
    nA.entities = ["aspirin"]                              # 筛后只剩 A（B 被丢）
    lost = nq.coverage_guard(before, [nA])
    assert lost == {"rareonly"}
    with pytest.raises(ValueError):
        nq.assert_coverage(before, [nA])
