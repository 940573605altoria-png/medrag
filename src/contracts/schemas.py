"""共享数据契约（schema）—— 全系统的接口地基（T007）。

**为什么这是地基**：walking skeleton 的桩实现与后续 US1/US2/US3 的真实实现
**共同遵守这一组契约**。先把它定准，桩→真肉替换时接口不变，集成风险前置归零。

贯穿其中的 constitution 铁律：
- **I 可溯源**：凡结论（报告 finding / QA 回答）必挂证据链（`EvidenceItem`）或定位 ROI；
  无证据 → 经 `abstain` 显式拒答，绝不编造（FR-002/003）。
- **小病灶为核心命题**：病灶面积分层 `AreaBand`（<2% / 2–5% / >5%）贯穿检测/评估全程。
- **II 训推一致**：ROI 既可来自索引端绿框坐标，也可来自推理端定位头输出，字段统一。

实现用 pydantic v2（骨架轻量依赖）。热图等数组在契约层用 `list[list[float]]` 表达
（可 JSON 序列化、跨 MCP 传输无 numpy 依赖）；模型内部计算时再转 numpy/torch。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SchemaVersion = Literal["0.1.0"]
SCHEMA_VERSION: SchemaVersion = "0.1.0"


# ════════════════════════════════════════════════════════════════════
# 枚举
# ════════════════════════════════════════════════════════════════════

class AreaBand(str, Enum):
    """病灶面积带——项目核心命题（小病灶 <2% 召回）的分层主轴。"""

    SMALL = "small"     # <2%   面积占比，过采样 + 重点评估
    MEDIUM = "medium"   # 2–5%
    LARGE = "large"     # >5%

    @classmethod
    def from_fraction(cls, frac: float) -> "AreaBand":
        """由病灶面积占比（0–1）判定所属带。"""
        if frac < 0.02:
            return cls.SMALL
        if frac <= 0.05:
            return cls.MEDIUM
        return cls.LARGE


class Modality(str, Enum):
    """证据/检索通道的模态。"""

    TEXT = "text"
    IMAGE = "image"


class AbstainReason(str, Enum):
    """拒答原因（FR-003）——结构化记录"为什么不回答"，供拒答正确性评估。"""

    NONE = "none"                  # 未拒答
    NO_EVIDENCE = "no_evidence"    # 检索无据
    LOW_CONFIDENCE = "low_confidence"  # rerank 最高分低于阈值
    UNCERTAIN = "uncertain"        # 证据矛盾/不确定


# ════════════════════════════════════════════════════════════════════
# 检测（C 定位头 / detect_lesions）
# ════════════════════════════════════════════════════════════════════

class BBox(BaseModel):
    """像素坐标轴对齐框 [x1, y1, x2, y2]（左上、右下）。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_order(self) -> "BBox":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"BBox 坐标非法（x2<x1 或 y2<y1）: {self}")
        return self

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class ROI(BaseModel):
    """病灶感兴趣区。

    索引端来自绿框坐标，推理端来自 C 定位头输出——**字段统一**，下游 B 双路融合
    与 c 图像检索都消费它（训推一致，铁律 II）。
    """

    bbox: BBox
    area_fraction: float = Field(ge=0.0, le=1.0, description="占全图面积比，定 AreaBand")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    label: str | None = Field(default=None, description="可选病灶类型；CT 无疾病标签时为 None")

    @property
    def area_band(self) -> AreaBand:
        return AreaBand.from_fraction(self.area_fraction)


class DetectionResult(BaseModel):
    """C 定位头输出：热图 + 框 + 面积带 + 置信。

    `heatmap` 为高斯定位热图（推理无框可用，= 系统"检测器"）；可为空（无显著病灶）。
    """

    rois: list[ROI] = Field(default_factory=list)
    heatmap: list[list[float]] | None = Field(
        default=None, description="HxW 定位热图（0–1）；省略以节省传输时为 None"
    )
    image_id: str | None = None
    abstained: bool = Field(default=False, description="无可信病灶时为 True（不强行框）")

    @property
    def area_bands(self) -> list[AreaBand]:
        return [roi.area_band for roi in self.rois]


# ════════════════════════════════════════════════════════════════════
# 检索证据（RAG / retrieve_evidence）
# ════════════════════════════════════════════════════════════════════

class EvidenceItem(BaseModel):
    """一条可溯源证据（constitution I 的最小单元）。

    `source_id` 串起去重/合并后的来源（含被合并副本 ID，证据不丢）；
    `citation` 是可展示引用串；`modality` 区分文本/图像通道。
    """

    source_id: str
    citation: str
    score: float = Field(description="检索/重排分（通道内可比；跨通道前需归一）")
    modality: Modality = Modality.TEXT
    text: str | None = Field(default=None, description="文本证据原文片段")
    case_id: str | None = Field(default=None, description="图像证据：关联 CT 病例")
    extra_source_ids: list[str] = Field(
        default_factory=list, description="被去重/合并的副本来源 ID（保溯源）"
    )
    flags: list[str] = Field(
        default_factory=list, description="如 conflict / llm_merged（透传到溯源）"
    )


class RetrievalResult(BaseModel):
    """检索输出：Top-K 证据 + 拒答信号（FR-003）。"""

    query: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    abstain: bool = Field(default=False, description="rerank 后最高分低于阈值则拒答")
    abstain_reason: AbstainReason = AbstainReason.NONE

    @model_validator(mode="after")
    def _abstain_consistency(self) -> "RetrievalResult":
        if self.abstain and self.abstain_reason is AbstainReason.NONE:
            raise ValueError("abstain=True 必须给出非 NONE 的 abstain_reason")
        if not self.abstain and self.abstain_reason is not AbstainReason.NONE:
            raise ValueError("abstain=False 时 abstain_reason 必须为 NONE")
        return self


# ════════════════════════════════════════════════════════════════════
# 报告生成（generate_report / medical_qa）
# ════════════════════════════════════════════════════════════════════

class Finding(BaseModel):
    """报告中的一条结论——**必须可溯源**：挂证据链或定位 ROI（constitution I）。"""

    text: str
    evidence: list[EvidenceItem] = Field(
        default_factory=list, description="支撑本结论的检索证据"
    )
    roi: ROI | None = Field(default=None, description="支撑本结论的定位 ROI（图像侧）")
    uncertain: bool = Field(default=False, description="证据不足/矛盾 → 标不确定，不删")

    @model_validator(mode="after")
    def _must_be_grounded(self) -> "Finding":
        # 非不确定结论必须有至少一项支撑（证据或 ROI），否则就是编造。
        if not self.uncertain and not self.evidence and self.roi is None:
            raise ValueError(
                "结论无证据且无 ROI 又未标 uncertain —— 违反 constitution I（禁编造）"
            )
        return self


class ReportResult(BaseModel):
    """端到端报告：结构化 findings + 每条结论证据链 + 拒答/不确定标注。"""

    findings: list[Finding] = Field(default_factory=list)
    summary: str = ""
    abstain: bool = Field(default=False, description="整体无据时拒答")
    abstain_reason: AbstainReason = AbstainReason.NONE
    detection: DetectionResult | None = None

    @model_validator(mode="after")
    def _abstain_consistency(self) -> "ReportResult":
        if self.abstain and self.abstain_reason is AbstainReason.NONE:
            raise ValueError("abstain=True 必须给出非 NONE 的 abstain_reason")
        return self

    def all_evidence(self) -> list[EvidenceItem]:
        out: list[EvidenceItem] = []
        for f in self.findings:
            out.extend(f.evidence)
        return out


# ════════════════════════════════════════════════════════════════════
# MCP 工具 IO（对 LangGraph client 暴露的接口契约）
# ════════════════════════════════════════════════════════════════════

class ToolStatus(str, Enum):
    OK = "ok"
    ABSTAINED = "abstained"   # 按 FR-003 拒答，非错误
    ERROR = "error"


class ToolIO(BaseModel):
    """MCP 工具统一返回包络——结构化 + 携带溯源（plan.md 部署节）。

    `payload` 装具体结果（DetectionResult/RetrievalResult/ReportResult 的 dump），
    跨项目（LangGraph）调用时溯源不丢；拒答经 `status=ABSTAINED` 显式传达。
    """

    tool: str
    status: ToolStatus = ToolStatus.OK
    payload: dict[str, Any] = Field(default_factory=dict)
    schema_version: SchemaVersion = SCHEMA_VERSION
    message: str = ""

    @classmethod
    def ok(cls, tool: str, result: BaseModel) -> "ToolIO":
        return cls(tool=tool, status=ToolStatus.OK, payload=result.model_dump(mode="json"))

    @classmethod
    def abstained(cls, tool: str, result: BaseModel, message: str = "") -> "ToolIO":
        return cls(
            tool=tool,
            status=ToolStatus.ABSTAINED,
            payload=result.model_dump(mode="json"),
            message=message,
        )


# ════════════════════════════════════════════════════════════════════
# 评估记录（eval harness / 归因底座）
# ════════════════════════════════════════════════════════════════════

class EvalRecord(BaseModel):
    """一次评估 run 的结构化记录（config → metrics），归因底座（constitution III）。

    与 `src/config/run_record.py` 互补：RunRecord 管环境/版本留痕，EvalRecord 管
    "哪套组件开关 + 哪个分层 → 什么指标"，供消融 runner 出 delta 表。
    """

    run_id: str
    flags: dict[str, Any] = Field(default_factory=dict, description="组件开关快照")
    metrics: dict[str, float] = Field(default_factory=dict, description="整体指标")
    stratified: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="按 AreaBand 分层指标：band -> {metric: value}"
    )
    baseline_id: str | None = Field(default=None, description="对照基线 run_id")
    significant: bool | None = Field(
        default=None, description="vs 基线是否显著（bootstrap CI/配对检验）"
    )


# ════════════════════════════════════════════════════════════════════
# 知识/样本元数据（ingestion / 入库）
# ════════════════════════════════════════════════════════════════════

class KnowledgeNode(BaseModel):
    """文本知识库（a 药品 / b 医学QA）的一个节点（去重/分块后入 ChromaDB）。"""

    node_id: str
    text: str
    collection: str = Field(description="a | b | c_text（按数据集类型分 collection）")
    lang: str | None = Field(default=None, description="zh | en（双语分治）")
    source_ids: list[str] = Field(default_factory=list, description="含被合并副本 ID")
    entities: list[str] = Field(default_factory=list, description="NER 抽取实体（覆盖统计）")
    quality_score: float | None = None
    flags: list[str] = Field(default_factory=list, description="conflict / llm_merged / low_coverage")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CTSample(BaseModel):
    """c CT-QA 的一个样本（视觉训练 + 检索）。"""

    case_id: str
    image_path: str
    report_text: str | None = None
    rois: list[ROI] = Field(default_factory=list, description="绿框/标注→ROI（标签源）")
    area_band: AreaBand | None = None
    in_eval_set: bool = Field(
        default=False, description="固定评估集守卫：True 则禁入 train/coreset/dedup"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "SCHEMA_VERSION",
    "SchemaVersion",
    "AreaBand",
    "Modality",
    "AbstainReason",
    "BBox",
    "ROI",
    "DetectionResult",
    "EvidenceItem",
    "RetrievalResult",
    "Finding",
    "ReportResult",
    "ToolStatus",
    "ToolIO",
    "EvalRecord",
    "KnowledgeNode",
    "CTSample",
]
