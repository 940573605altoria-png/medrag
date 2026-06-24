"""数据 ingestion 管线单测（T008）—— raw 加载器 + 清洗管线编排，纯逻辑本地测。"""

from __future__ import annotations

import json

import pytest

from src.contracts.schemas import KnowledgeNode
from src.data import ingest as ig
from src.data import ner as N
from src.data.ner import Entity
from src.data.qa_conflict import QAItem, Verdict


class _FakeNER:
    """子串命中即抽实体（避开中文分词，便于测试）。"""

    def __init__(self, vocab):
        self.vocab = vocab

    def extract(self, text):
        return [Entity(v, N.DRUG) for v in self.vocab if v in text]


def _node(node_id, text, **meta):
    return KnowledgeNode(node_id=node_id, text=text, collection="a_drug",
                         source_ids=[node_id], metadata=meta)


# ── raw 加载器 ────────────────────────────────────────────────────

def test_load_jsonl_and_adapters(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"id": "1", "question": "q?", "answer": "a", "drug": "阿司匹林"}) + "\n"
        + json.dumps({"id": "2", "text": "药品说明"}) + "\n",
        encoding="utf-8",
    )
    recs = ig.load_jsonl(p)
    assert len(recs) == 2 and recs[0].metadata["drug"] == "阿司匹林"
    items = ig.qa_records_to_items(recs)
    assert len(items) == 1 and items[0].source_id == "1"   # 仅含 q+a 的记录
    nodes = ig.drug_records_to_nodes(recs)
    assert len(nodes) == 2 and nodes[0].collection == "a_drug"


# ── ingest_documents（a/通用）─────────────────────────────────────

def test_ingest_documents_clean_dedup_filter():
    n1 = _node("n1", "阿司匹林 解热镇痛 电话13800138000")
    n2 = _node("n2", "阿司匹林  解热镇痛  电话13800138000")   # 清洗后与 n1 同
    n3 = _node("n3", "无实体内容片段")
    res = ig.ingest_documents([n1, n2, n3], ner=_FakeNER({"阿司匹林"}))
    assert [x.node_id for x in res.nodes] == ["n1"]          # 去重+零实体丢
    assert "n3" in res.dropped_ids
    assert "n2" in n1.source_ids                             # 来源并入
    assert "13800138000" not in n1.text                      # PHI 抹掉
    assert res.coverage_lost == set()                        # FR-012 护栏
    assert res.stats.phi_found == 2 and res.stats.n_in == 3


def test_ingest_documents_deid_switch_off():
    n = _node("n", "电话13800138000")
    res = ig.ingest_documents([n], config=ig.IngestConfig(deidentify=False, ner_filter=False))
    assert "13800138000" in res.nodes[0].text               # 关 deid → 保留


def test_ingest_documents_structured_dedup():
    a = _node("a", "正文1", drug="阿司匹林", section="适应症")
    b = _node("b", "正文2", drug="阿司匹林", section="适应症")  # 同业务键
    cfg = ig.IngestConfig(structured_key_fields=("drug", "section"), ner_filter=False)
    res = ig.ingest_documents([a, b], config=cfg)
    assert len(res.nodes) == 1 and "b" in res.nodes[0].source_ids


# ── ingest_qa（b）─────────────────────────────────────────────────

def test_ingest_qa_conflict_and_deid():
    items = [
        QAItem("作用?", "解热镇痛 电话13800138000", "s1"),
        QAItem("作用?", "用于解热和镇痛", "s2"),
    ]
    cfg = ig.IngestConfig(qa_conflict=True, ner_filter=False)
    res = ig.ingest_qa(items, config=cfg, judge_fn=lambda q, a: Verdict.EQUIVALENT)
    assert len(res.nodes) == 1
    node = res.nodes[0]
    assert set(node.source_ids) == {"s1", "s2"} and "equivalent" in node.flags
    assert "13800138000" not in node.text                    # 送审/入库前已 deid
    assert node.metadata["question"] == "作用?"


def test_ingest_qa_passthrough_without_conflict():
    items = [QAItem("q1?", "答案一", "s1"), QAItem("q2?", "答案二", "s2")]
    res = ig.ingest_qa(items, config=ig.IngestConfig(qa_conflict=False, ner_filter=False))
    assert len(res.nodes) == 2
