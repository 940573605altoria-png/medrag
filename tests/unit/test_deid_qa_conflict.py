"""PHI 去标识 + QA 冲突单测（T009/T040）—— 纯逻辑、judge/merge 注入桩，本地全测。"""

from __future__ import annotations

import pytest

from src.data import deid as D
from src.data import qa_conflict as qc
from src.data.qa_conflict import QAItem, Verdict


# ── T009 去标识 ───────────────────────────────────────────────────

def test_deidentify_value_phi():
    res = D.deidentify("联系 13800138000，邮箱 a@b.com，身份证 11010119900307123X")
    assert "[PHONE]" in res.text and "[EMAIL]" in res.text and "[ID]" in res.text
    assert {m.category for m in res.found} >= {"phone", "email", "id_card"}


def test_deidentify_preserves_dose():
    # 剂量/普通数字不该被当 PHI 抹掉
    assert D.deidentify("每次 5mg，每日 3 次").text == "每次 5mg，每日 3 次"


def test_deidentify_labeled_field():
    out = D.deidentify("姓名：张三 患者男").text
    assert "张三" not in out and "[REDACTED]" in out


def test_has_phi_and_assert():
    assert D.has_phi("call 13800138000")
    assert not D.has_phi("已抹 [PHONE] 占位")            # 占位符不算 PHI
    with pytest.raises(ValueError):
        D.assert_no_phi("a@b.com")
    D.assert_no_phi("纯净文本 5mg")                       # 不抛


def test_deid_config_switch():
    cfg = D.DeidConfig(phone=False)
    assert "13800138000" in D.deidentify("13800138000", cfg).text  # 关掉手机 → 不抹


# ── T040 QA 冲突 ──────────────────────────────────────────────────

def _items():
    return [
        QAItem("阿司匹林的作用?", "解热镇痛。", "s1"),
        QAItem("阿司匹林的作用?", "用于解热和镇痛。", "s2"),
    ]


def test_group_by_question():
    g = qc.group_by_question(_items())
    assert len(g) == 1 and len(next(iter(g.values()))) == 2


def test_single_distinct_answer_merges_sources():
    items = [QAItem("q?", "同一答案。", "s1"), QAItem("q?", "同一答案。", "s2")]
    out = qc.resolve_conflicts(items, judge_fn=lambda q, a: Verdict.EQUIVALENT)
    assert len(out) == 1 and set(out[0].source_ids) == {"s1", "s2"}  # 未触发 judge


def test_equivalent_keeps_one_merges_sources():
    out = qc.resolve_conflicts(_items(), judge_fn=lambda q, a: Verdict.EQUIVALENT)
    assert len(out) == 1
    assert set(out[0].source_ids) == {"s1", "s2"} and "equivalent" in out[0].flags


def test_complementary_llm_merge():
    out = qc.resolve_conflicts(
        _items(),
        judge_fn=lambda q, a: Verdict.COMPLEMENTARY,
        merge_fn=lambda q, a: "解热、镇痛（合并）。",
    )
    assert len(out) == 1 and "llm_merged" in out[0].flags
    assert out[0].answer == "解热、镇痛（合并）。"
    assert set(out[0].source_ids) == {"s1", "s2"}


def test_conflicting_keeps_all_flagged():
    out = qc.resolve_conflicts(_items(), judge_fn=lambda q, a: Verdict.CONFLICTING)
    assert len(out) == 2 and all("conflict" in r.flags for r in out)


def test_disabled_passthrough():
    out = qc.resolve_conflicts(
        _items(), judge_fn=lambda q, a: Verdict.EQUIVALENT,
        config=qc.QAConflictConfig(enabled=False),
    )
    assert len(out) == 2 and all(not r.flags for r in out)


def test_deid_before_judge_no_phi_egress():
    captured = {}

    def judge(q, answers):
        captured["q"] = q
        captured["answers"] = answers
        return Verdict.EQUIVALENT

    items = [
        QAItem("怎么联系?", "拨打 13800138000 咨询。", "s1"),
        QAItem("怎么联系?", "电话 13800138000 即可。", "s2"),
    ]
    qc.resolve_conflicts(items, judge_fn=judge)
    # 送进 judge 的文本已去标识，无原始手机号
    assert all("13800138000" not in a for a in captured["answers"])
    assert "[PHONE]" in captured["answers"][0]
