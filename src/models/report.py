"""真实报告生成（T034）—— 替换 stub_report，强制可溯源、禁编造（FR-002/003）。

constitution I 铁律：报告里**每条结论都必须挂证据或定位 ROI**，否则要么标 `uncertain`、
要么整体拒答——绝不输出无据断言。难点不在"让 VLM 写字"，而在"写完之后强制把每句话锚回
来源"。所以拆成两层：

1. **生成草稿（可换）**：把检测 ROI + 检索证据组装成提示，让 VLM 产出候选结论，并要求它对
   每条结论标引用 `[S1]`(证据)/`[ROI1]`(定位)。这层需要基座模型（或测试时注入 `draft_fn`）。
2. **强制溯源组装（纯逻辑、本模块核心）**：`assemble_report` 解析每条结论的引用标签，把它锚到
   对应证据/ROI；锚不上的结论 → 标 `uncertain`（不删、不编造）或按配置丢弃；整体无据 → 拒答。

第 2 层不依赖 torch，可本地完整单测；第 1 层的 VLM 调用走 [qwen3vl.quick_infer]。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.contracts.schemas import (
    AbstainReason,
    DetectionResult,
    Finding,
    ReportResult,
    RetrievalResult,
)

_SRC_TAG = re.compile(r"\[S(\d+)\]")      # 证据引用，如 [S1]
_ROI_TAG = re.compile(r"\[ROI(\d+)\]")    # 定位引用，如 [ROI1]


@dataclass
class ReportConfig:
    max_new_tokens: int = 256
    drop_ungrounded: bool = False          # True 丢弃无据结论；False 标 uncertain 保留
    max_evidence_per_finding: int = 3


def build_report_prompt(detection: DetectionResult, retrieval: RetrievalResult,
                        *, instruction: str | None = None) -> str:
    """组装报告提示：编号证据 [S*] + 检测区域 [ROI*]，要求模型逐条引用。"""
    instr = instruction or (
        "You are a radiology report assistant. Write concise factual findings. "
        "EVERY finding MUST cite its support: evidence as [S<n>] and/or the lesion "
        "region as [ROI<n>]. Never state anything you cannot cite."
    )
    lines = [instr]
    if detection.rois:
        lines.append("\nDetected regions:")
        for i, roi in enumerate(detection.rois, 1):
            lines.append(f"  [ROI{i}] {roi.area_band.value} band, conf {roi.confidence:.2f}")
    if retrieval.evidence:
        lines.append("\nEvidence:")
        for i, ev in enumerate(retrieval.evidence, 1):
            lines.append(f"  [S{i}] {ev.text or ev.citation}")
    lines.append("\nFindings (one per line, each with citations):")
    return "\n".join(lines)


def _strip_tags(text: str) -> str:
    return _ROI_TAG.sub("", _SRC_TAG.sub("", text)).strip()


def assemble_report(candidate_texts: Sequence[str], detection: DetectionResult,
                    retrieval: RetrievalResult, *, config: ReportConfig | None = None) -> ReportResult:
    """把候选结论按引用标签强制锚到证据/ROI，产出可溯源 ReportResult。"""
    cfg = config or ReportConfig()
    has_ev, has_roi = bool(retrieval.evidence), bool(detection.rois)

    # 整体无据 / 检索已拒答 → 拒答（不强行编报告）
    if retrieval.abstain or (not has_ev and not has_roi):
        return ReportResult(
            findings=[], summary="", abstain=True,
            abstain_reason=retrieval.abstain_reason if retrieval.abstain else AbstainReason.NO_EVIDENCE,
            detection=detection,
        )

    findings: list[Finding] = []
    for raw in candidate_texts:
        text = raw.strip()
        if not text:
            continue
        src_ids = [int(m) for m in _SRC_TAG.findall(text)]
        roi_ids = [int(m) for m in _ROI_TAG.findall(text)]
        evidence = [
            retrieval.evidence[i - 1] for i in src_ids
            if 1 <= i <= len(retrieval.evidence)
        ][: cfg.max_evidence_per_finding]
        roi = next((detection.rois[i - 1] for i in roi_ids
                    if 1 <= i <= len(detection.rois)), None)

        clean = _strip_tags(text)
        if evidence or roi is not None:
            findings.append(Finding(text=clean, evidence=evidence, roi=roi))
        elif not cfg.drop_ungrounded:
            findings.append(Finding(text=clean, uncertain=True))  # 无据 → 标不确定，不编造

    if not findings:   # 全部无据且被丢弃 → 拒答
        return ReportResult(
            findings=[], summary="", abstain=True,
            abstain_reason=AbstainReason.NO_EVIDENCE, detection=detection,
        )

    grounded = sum(1 for f in findings if not f.uncertain)
    return ReportResult(
        findings=findings,
        summary=f"{len(findings)} finding(s), {grounded} grounded",
        abstain=False, abstain_reason=AbstainReason.NONE, detection=detection,
    )


class ReportGenerator:
    """报告生成器：VLM 出草稿 → 强制溯源组装。`model` 缺省时只能用注入的 `draft_fn`。"""

    def __init__(self, model: Any = None, processor: Any = None,
                 config: ReportConfig | None = None):
        self.model = model
        self.processor = processor
        self.config = config or ReportConfig()

    def generate(self, detection: DetectionResult, retrieval: RetrievalResult,
                 image: Any = None, *,
                 draft_fn: Callable[[str, Any], str] | None = None) -> ReportResult:
        """生成可溯源报告。`draft_fn(prompt, image)->str` 优先（测试/自定义）；否则用 VLM。"""
        # 整体无据时直接拒答，不必调模型
        if retrieval.abstain or (not retrieval.evidence and not detection.rois):
            return assemble_report([], detection, retrieval, config=self.config)

        prompt = build_report_prompt(detection, retrieval)
        if draft_fn is not None:
            raw = draft_fn(prompt, image)
        elif self.model is not None:
            raw = self._vlm_draft(prompt, image)
        else:
            raise RuntimeError("ReportGenerator 需要 model 或 draft_fn 之一")

        candidates = [ln for ln in raw.splitlines() if ln.strip()]
        return assemble_report(candidates, detection, retrieval, config=self.config)

    def _vlm_draft(self, prompt: str, image: Any) -> str:
        from src.models.qwen3vl import quick_infer  # noqa: PLC0415

        return quick_infer(self.model, self.processor, image, prompt,
                           max_new_tokens=self.config.max_new_tokens)


__all__ = ["ReportConfig", "build_report_prompt", "assemble_report", "ReportGenerator"]
