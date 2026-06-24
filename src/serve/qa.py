"""真实 medical_qa（T047）—— hybrid 检索 → 精排+拒答 → 带引用回答，替换骨架桩、接回 MCP。

US2 的端到端收口：把 [retrieve_text.HybridRetriever]（T045）+ [rerank.Reranker]（T046）串成一个
`answer(query)`，产出**带引用的回答**——每句结论必须挂 `[S<n>]` 锚到检索证据，无据→标 uncertain、
低置信/无证据→拒答（constitution I，禁编造）。

**复用而非重造**：grounding（引用标签解析→锚证据→拒答）直接用 [report.assemble_report]（T034）——
报告与 QA 的"强制溯源"是同一道关，只是 QA 无图像/ROI。本模块只多做"QA 提示词 + 串检索/精排"。

LLM 草稿用注入 `draft_fn`（测试/自定义）或基座模型；检索/精排/草稿全可注入，故**编排逻辑本地可测**，
真实模型在 AutoDL。接回方式：`Pipeline(qa_service=...)` → MCP `medical_qa` 工具自动走真实路（默认仍桩）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.contracts.schemas import DetectionResult, ReportResult, RetrievalResult
from src.models.report import ReportConfig, assemble_report
from src.rag.rerank import Reranker
from src.rag.retrieve_text import HybridRetriever

QA_INSTRUCTION = (
    "You are a medical QA assistant. Answer the question using ONLY the evidence below. "
    "EVERY sentence MUST cite its support as [S<n>]. If the evidence does not answer the "
    "question, say you cannot answer. Never state anything you cannot cite."
)


def build_qa_prompt(
    query: str, retrieval: RetrievalResult, *, instruction: str | None = None
) -> str:
    """组装 QA 提示：编号证据 [S*]，要求逐句引用（与 report 的 [S*] 解析对齐）。"""
    lines = [instruction or QA_INSTRUCTION, f"\nQuestion: {query}", "\nEvidence:"]
    for i, ev in enumerate(retrieval.evidence, 1):
        lines.append(f"  [S{i}] {ev.text or ev.citation}")
    lines.append("\nAnswer (cite [S<n>] inline, one sentence per line):")
    return "\n".join(lines)


@dataclass
class MedicalQA:
    """医学问答服务：检索 → 精排+拒答 → 带引用回答。

    `draft_fn(prompt)->str` 优先（测试/自定义）；否则用基座 `model`（+`processor`）。
    """

    retriever: HybridRetriever
    reranker: Reranker
    draft_fn: Callable[[str], str] | None = None
    model: Any = None
    processor: Any = None
    report_config: ReportConfig = field(default_factory=ReportConfig)

    def answer(self, query: str, top_k: int = 5) -> ReportResult:
        """检索→精排→带引用回答。低置信/无证据按 RetrievalResult 的拒答原因拒答。"""
        retrieval = self.retriever.retrieve(query)          # T045
        reranked = self.reranker.rerank(retrieval)          # T046（拒答门在此）
        no_detection = DetectionResult(abstained=True)      # QA 无图像侧

        # 检索/精排已拒答或无证据 → 直接拒答，不调 LLM、不编造。
        if reranked.abstain or not reranked.evidence:
            return assemble_report([], no_detection, reranked, config=self.report_config)

        prompt = build_qa_prompt(query, reranked)
        raw = self._draft(prompt)
        candidates = [ln for ln in raw.splitlines() if ln.strip()]
        return assemble_report(candidates, no_detection, reranked, config=self.report_config)

    def _draft(self, prompt: str) -> str:
        if self.draft_fn is not None:
            return self.draft_fn(prompt)
        if self.model is not None:
            from src.models.qwen3vl import quick_infer  # noqa: PLC0415

            return quick_infer(self.model, self.processor, None, prompt,
                               max_new_tokens=self.report_config.max_new_tokens)
        raise RuntimeError("MedicalQA 需要 draft_fn 或 model 之一")


__all__ = ["QA_INSTRUCTION", "build_qa_prompt", "MedicalQA"]
