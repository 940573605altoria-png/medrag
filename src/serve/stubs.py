"""桩组件（T020）—— walking skeleton 的"假肉"。

每个桩**只实现 T007 契约、返回确定性假数据**，不含任何真实模型/检索逻辑。
作用：让 `pipeline.py` 把端到端串通、`mcp_server.py` 把 4 工具暴露出来，
先锁死接口/数据流/服务边界。US1/US2 用真实实现替换这些桩时**签名不变**。

确定性原则（可复现，constitution III）：桩输出不含随机性，固定输入→固定输出。
溯源占位原则（constitution I）：桩也填**结构完整**的证据链（假 source_id/citation），
让"可溯源"的接口契约从骨架阶段就成立、被测试覆盖。

替换映射（后续批次）：
    stub_detect   → T028/T033 C 定位头 + 视觉检索
    stub_retrieve → T045/T046 hybrid 检索 + reranker
    stub_report   → T034 真实报告生成
"""

from __future__ import annotations

from src.contracts.schemas import (
    AbstainReason,
    BBox,
    DetectionResult,
    EvidenceItem,
    Finding,
    Modality,
    ReportResult,
    RetrievalResult,
    ROI,
)


def stub_detect(image_id: str = "stub-image") -> DetectionResult:
    """假检测：返回一个固定的小病灶 ROI（<2% 面积带，呼应核心命题）。"""
    roi = ROI(
        bbox=BBox(x1=20.0, y1=30.0, x2=40.0, y2=50.0),
        area_fraction=0.015,  # SMALL band
        confidence=0.80,
        label=None,  # CT 无疾病标签
    )
    return DetectionResult(
        rois=[roi],
        heatmap=None,  # 桩不产热图，避免大数组占传输
        image_id=image_id,
        abstained=False,
    )


def stub_retrieve(query: str, top_k: int = 5) -> RetrievalResult:
    """假检索：返回带完整溯源字段的假证据；空 query 走拒答分支（演示 FR-003）。"""
    if not query.strip():
        return RetrievalResult(
            query=query,
            evidence=[],
            abstain=True,
            abstain_reason=AbstainReason.NO_EVIDENCE,
        )

    evidence = [
        EvidenceItem(
            source_id=f"stub-src-{i}",
            citation=f"[stub] source {i} for query={query!r}",
            score=1.0 - 0.1 * i,
            modality=Modality.TEXT,
            text=f"stub evidence snippet {i}",
        )
        for i in range(min(top_k, 3))
    ]
    return RetrievalResult(query=query, evidence=evidence, abstain=False)


def stub_report(
    detection: DetectionResult,
    retrieval: RetrievalResult,
) -> ReportResult:
    """假报告：把桩检测/检索拼成结构化报告，每条结论挂证据或 ROI（契约护栏）。

    无任何证据且无 ROI → 整体拒答（演示端到端拒答门）。
    """
    has_evidence = bool(retrieval.evidence)
    has_roi = bool(detection.rois)

    if retrieval.abstain or (not has_evidence and not has_roi):
        return ReportResult(
            findings=[],
            summary="",
            abstain=True,
            abstain_reason=(
                retrieval.abstain_reason
                if retrieval.abstain
                else AbstainReason.NO_EVIDENCE
            ),
            detection=detection,
        )

    findings: list[Finding] = []
    # 检测结论：挂 ROI（图像侧溯源）
    for roi in detection.rois:
        findings.append(
            Finding(
                text=f"[stub] suspected lesion at {roi.area_band.value} band",
                roi=roi,
                evidence=retrieval.evidence[:1],  # 同时挂一条检索证据
            )
        )
    # 知识结论：挂检索证据（文本侧溯源）
    if has_evidence:
        findings.append(
            Finding(
                text="[stub] knowledge-grounded note",
                evidence=retrieval.evidence,
            )
        )

    return ReportResult(
        findings=findings,
        summary=f"[stub] report with {len(findings)} grounded finding(s)",
        abstain=False,
        abstain_reason=AbstainReason.NONE,
        detection=detection,
    )
