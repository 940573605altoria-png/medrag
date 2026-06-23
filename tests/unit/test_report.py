"""报告生成单测（T034）—— 纯逻辑，验证"强制可溯源、禁编造"（本机即可全跑）。

重点验组装层：引用标签 [S*]/[ROI*] 把每条结论锚到证据/ROI；锚不上 → 标 uncertain 或丢弃；
整体无据 / 检索拒答 → 整体拒答。
"""

from __future__ import annotations

import pytest

from src.contracts.schemas import (
    AbstainReason,
    BBox,
    DetectionResult,
    EvidenceItem,
    Modality,
    RetrievalResult,
    ROI,
)
from src.models import report


def _detection(n=1):
    rois = [
        ROI(bbox=BBox(x1=10, y1=10, x2=30, y2=30), area_fraction=0.015, confidence=0.8)
        for _ in range(n)
    ]
    return DetectionResult(rois=rois, image_id="ct-1", abstained=not rois)


def _retrieval(n=2):
    ev = [
        EvidenceItem(source_id=f"s{i}", citation=f"cite {i}", score=1.0 - 0.1 * i,
                     modality=Modality.TEXT, text=f"evidence text {i}")
        for i in range(1, n + 1)
    ]
    return RetrievalResult(query="q", evidence=ev, abstain=False)


def test_prompt_includes_evidence_and_rois():
    p = report.build_report_prompt(_detection(1), _retrieval(2))
    assert "[ROI1]" in p and "[S1]" in p and "[S2]" in p
    assert "cite" in p or "evidence text" in p


def test_assemble_grounds_by_tags_and_strips_them():
    det, ret = _detection(1), _retrieval(2)
    cands = ["Hyperdensity consistent with hemorrhage [ROI1][S1].",
             "Supported by guideline [S2]."]
    res = report.assemble_report(cands, det, ret)
    assert not res.abstain and len(res.findings) == 2
    f0 = res.findings[0]
    assert f0.roi is det.rois[0]
    assert [e.source_id for e in f0.evidence] == ["s1"]
    assert "[ROI1]" not in f0.text and "[S1]" not in f0.text   # 标签已剥离
    assert res.findings[1].evidence[0].source_id == "s2"


def test_ungrounded_finding_marked_uncertain():
    det, ret = _detection(1), _retrieval(1)
    res = report.assemble_report(["A vague claim with no citation."], det, ret)
    # 该结论无据 → uncertain（不删、不编造）
    assert res.findings[0].uncertain is True
    assert res.findings[0].evidence == [] and res.findings[0].roi is None


def test_drop_ungrounded_config():
    det, ret = _detection(1), _retrieval(1)
    cfg = report.ReportConfig(drop_ungrounded=True)
    res = report.assemble_report(["grounded [S1]", "ungrounded claim"], det, ret, config=cfg)
    assert len(res.findings) == 1 and not res.findings[0].uncertain


def test_abstains_when_no_evidence_and_no_roi():
    det = DetectionResult(rois=[], abstained=True)
    ret = RetrievalResult(query="q", evidence=[], abstain=False)
    res = report.assemble_report(["anything [S1]"], det, ret)
    assert res.abstain and res.abstain_reason is AbstainReason.NO_EVIDENCE


def test_respects_retrieval_abstain():
    det = _detection(1)
    ret = RetrievalResult(query="q", evidence=[], abstain=True,
                          abstain_reason=AbstainReason.LOW_CONFIDENCE)
    res = report.assemble_report(["x [ROI1]"], det, ret)
    assert res.abstain and res.abstain_reason is AbstainReason.LOW_CONFIDENCE


def test_all_ungrounded_with_drop_abstains():
    det, ret = _detection(1), _retrieval(1)
    cfg = report.ReportConfig(drop_ungrounded=True)
    res = report.assemble_report(["no cite here", "also nothing"], det, ret, config=cfg)
    assert res.abstain and res.abstain_reason is AbstainReason.NO_EVIDENCE


# ── ReportGenerator：注入 draft_fn，端到端不依赖 torch ───────────────

def test_generator_with_injected_draft_fn():
    det, ret = _detection(1), _retrieval(2)
    captured = {}

    def fake_draft(prompt, image):
        captured["prompt"] = prompt
        return "Lesion noted [ROI1][S1].\nGuideline note [S2]."

    res = report.ReportGenerator().generate(det, ret, draft_fn=fake_draft)
    assert "[ROI1]" in captured["prompt"]          # 提示确实带了引用脚手架
    assert not res.abstain and len(res.findings) == 2
    assert res.findings[0].roi is det.rois[0]


def test_generator_abstains_without_calling_draft():
    det = DetectionResult(rois=[], abstained=True)
    ret = RetrievalResult(query="q", evidence=[], abstain=False)
    called = {"n": 0}

    def draft(prompt, image):
        called["n"] += 1
        return "should not be called"

    res = report.ReportGenerator().generate(det, ret, draft_fn=draft)
    assert res.abstain and called["n"] == 0        # 无据直接拒答，不调模型


def test_generator_requires_model_or_draft_fn():
    det, ret = _detection(1), _retrieval(1)
    with pytest.raises(RuntimeError, match="model"):
        report.ReportGenerator().generate(det, ret)   # 无 model 无 draft_fn
