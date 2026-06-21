"""端到端管线（T021）—— 把 detect → retrieve → report 串成一具骨架。

**这具骨架是后续所有 US 的填肉对象**：每个真实组件做完就替回这里对着"活的整体"
端到端验证。当前各步走 `stubs.py` 的桩。

按 `AppConfig.flags` 选实现（骨架阶段全为 "stub"）：真实组件接入时，在
`_select_*` 里加分支即可，**pipeline 的对外签名与数据流不变**。
MCP server（T022）只是薄薄包住本管线 —— 本地直调与 MCP 调用同一条 pipeline，
保证评估 harness 照测管线、归因不受接口层影响（plan.md 部署节）。
"""

from __future__ import annotations

from src.config.config import AppConfig, load_config
from src.contracts.schemas import (
    DetectionResult,
    ReportResult,
    RetrievalResult,
)
from src.serve import stubs


class Pipeline:
    """医学多模态推理管线：检测 + 检索 + 报告。

    四个对外能力对应 4 个 MCP 工具：
        detect_lesions    → detect()
        retrieve_evidence → retrieve()
        medical_qa        → answer()
        generate_report   → generate_report()
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()

    # ── 单步能力 ────────────────────────────────────────────────

    def detect(self, image_id: str = "stub-image") -> DetectionResult:
        """病灶检测（C 定位头）。flags['detect']: stub | lochead。"""
        impl = self.config.flags.get("detect", "stub")
        if impl == "stub":
            return stubs.stub_detect(image_id)
        raise NotImplementedError(f"detect 实现 {impl!r} 尚未接入（US1: T028/T033）")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """证据检索（RAG 级联）。flags['retrieve']: stub | cascade。"""
        impl = self.config.flags.get("retrieve", "stub")
        if impl == "stub":
            return stubs.stub_retrieve(query, top_k)
        raise NotImplementedError(f"retrieve 实现 {impl!r} 尚未接入（US2: T045/T046）")

    def answer(self, question: str, top_k: int = 5) -> ReportResult:
        """医学问答（a/b 知识库）：纯文本检索 → 带引用回答，无图像阶段。"""
        retrieval = self.retrieve(question, top_k)
        # 复用报告生成器把检索结果组织成带引用的回答（无检测）。
        return self._report(DetectionResult(abstained=True), retrieval)

    def generate_report(
        self, image_id: str = "stub-image", *, top_k: int = 5
    ) -> ReportResult:
        """端到端：CT → 检测 → 用检测草稿驱动检索 → 生成带溯源报告。"""
        detection = self.detect(image_id)
        # query-adaptive（plan.md）：用检测草稿作检索 query（骨架版极简）。
        draft_query = self._detection_to_query(detection)
        retrieval = self.retrieve(draft_query, top_k)
        return self._report(detection, retrieval)

    # ── 内部 ────────────────────────────────────────────────────

    def _report(
        self, detection: DetectionResult, retrieval: RetrievalResult
    ) -> ReportResult:
        impl = self.config.flags.get("report", "stub")
        if impl == "stub":
            return stubs.stub_report(detection, retrieval)
        raise NotImplementedError(f"report 实现 {impl!r} 尚未接入（US1: T034）")

    @staticmethod
    def _detection_to_query(detection: DetectionResult) -> str:
        """把检测草稿转成检索 query（骨架版：拼面积带文本）。

        真实版（US1 T033）：C 检测头草稿 findings/病灶带/器官属性 → 文本 query，
        并行图像主通道兜底。这里仅占位以打通数据流。
        """
        if detection.abstained or not detection.rois:
            return ""
        bands = ", ".join(roi.area_band.value for roi in detection.rois)
        return f"lesion findings in bands: {bands}"


# 模块级默认管线（MCP server 复用，避免每次请求重建）。
_default_pipeline: Pipeline | None = None


def get_pipeline(config: AppConfig | None = None) -> Pipeline:
    """取默认管线单例；传 config 则新建（消融/测试用）。"""
    global _default_pipeline
    if config is not None:
        return Pipeline(config)
    if _default_pipeline is None:
        _default_pipeline = Pipeline()
    return _default_pipeline
