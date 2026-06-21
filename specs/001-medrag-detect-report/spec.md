# Feature Specification: 医学多模态病灶检测与报告生成系统

**Feature Branch**: `001-medrag-detect-report`

**Created**: 2026-06-21

**Status**: Draft（B/C 已定，数据/部署细节待澄清）

**Input**: 基于 RAG + Qwen3-VL 的"病灶检测 + 报告生成"垂直系统，解决通用 VLM 的微小病灶特征
湮灭与幻觉问题。本 spec 记录 WHAT/WHY；技术方案（双路融合 B、高斯退火定位 C 等 HOW）见同目录 `plan.md`。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 上传 CT 得到可溯源的检测+报告 (Priority: P1)

用户上传一张（推理时无标注框的）CT 影像，系统自动定位病灶区域，并生成结构化诊断报告；报告中
每条关键结论都附带证据来源（命中的病例/知识，或检测出的病灶 ROI）。

**Why this priority**: 这是系统的核心价值与 Demo 主线——检测 + 报告 + 溯源三合一，直接对应面试展示。

**Independent Test**: 给定一张干净 CT，端到端跑通"定位→检索→生成带溯源报告"，即可单独验证并交付价值。

**Acceptance Scenarios**:

1. **Given** 一张无框 CT 影像, **When** 提交系统, **Then** 返回病灶定位（热图/框）+ 结构化报告，且报告每条结论可点开对应证据。
2. **Given** 一张无明显病灶的 CT, **When** 提交系统, **Then** 系统不编造病灶，报告明确给出"未见明确病灶"或不确定标注。
3. **Given** 一个微小病灶（<2% 图像面积）的 CT, **When** 提交系统, **Then** 病灶被召回的概率显著高于单路基线。

### User Story 2 - 基于知识库的可溯源医学问答 (Priority: P2)

用户就药品信息或医学问题提问，系统经父子层级检索 + 级联（文本过滤→视觉）+ 重排，给出带引用的回答。

**Why this priority**: 支撑报告生成的证据底座，也是独立可用的医学 QA 能力（数据集 a/b 的落点）。

**Independent Test**: 对一组医学/药品问题，验证回答可溯源到检索来源且 faithfulness 指标达标。

**Acceptance Scenarios**:

1. **Given** 一个药品/医学问题, **When** 提交, **Then** 返回答案 + Top-K 引用来源，答案不超出来源支持范围。

### User Story 3 - 基线+消融的可归因评估 (Priority: P3)

研究者在固定测试集上对各模块（双路融合、定位损失、RAG、重排）做开关消融，量化各自增益。

**Why this priority**: 面试核心方法论——无消融不可归因（见 constitution III）。

**Independent Test**: 评估 harness 能对任一改动单独开关并产出对比指标（含小病灶分层）。

**Acceptance Scenarios**:

1. **Given** 固定测试集与基线, **When** 仅开启一个改动, **Then** 产出该改动相对基线的指标增量与分层结果。

### Edge Cases

- 推理输入意外带有标注框时如何处理（避免读框走捷径）？
- 检索库为空/无相关证据时，系统 MUST 拒答或标注不确定，而非编造。
- 中/英文混合提问与报告语言一致性。
- 病灶被填充色块完全遮盖（若数据如此）→ inpainting 无法还原，需在数据层处理。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 对干净 CT 影像输出病灶定位（热图或框）。
- **FR-002**: 系统 MUST 生成结构化诊断报告，且每条关键结论 MUST 关联可追溯证据（检索来源或 ROI）。
- **FR-003**: 系统 MUST NOT 编造无证据支撑的医学结论；无证据时 MUST 标注不确定或拒答。
- **FR-004**: 数据处理 MUST 将标注绿框仅用作监督标签并从模型输入抹除（训推同分布），并防范二阶（inpainting 痕迹）泄露。
- **FR-005**: 检索 MUST 支持父子层级检索 + 文本过滤→视觉的级联 + Top-5 重排。
- **FR-006**: 系统 MUST 提供可单独开关每个改动的评估 harness，并支持按病灶面积分层报告。
- **FR-007**: 数据 MUST 去标识化，不得在日志/向量库明文/外发中泄露 PHI。
- **FR-008**: 输入数据集分工 [NEEDS CLARIFICATION: a 药品/b 医学QA/c CT-QA 各自进 RAG 库还是训练，待定]。
- **FR-009**: 部署形态 [NEEDS CLARIFICATION: 推理时是否一直带框；绿框为细边框还是填充块]。
- **FR-010**: 运行环境 MUST 适配 AutoDL [NEEDS CLARIFICATION: 卡型/预算/多卡，待定]。

### Key Entities

- **CT 影像样本**: 原始（带绿框）图、清洗后（抹框）图、绿框坐标、高斯热图标签、病灶面积分层标签。
- **知识/病例条目**: 来自药品数据集、医学QA、病例库的文本/图文条目；含 metadata（用于文本过滤）与向量。
- **检索证据**: 一次查询命中的 Top-K 条目及其分数，供报告/回答溯源。
- **评估记录**: 模块开关配置 → 指标（faithfulness、实体F1、检测 mAP、小病灶分层等）。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 微小病灶（<2% 面积）召回相对单路 global-only 基线提升（目标待基线确定后设阈值）。
- **SC-002**: 报告 faithfulness / 上下文精确率（ragas）相对无 RAG 基线显著提升，幻觉率下降。
- **SC-003**: 病灶检测 mAP / sensitivity@FP 达到可展示水平（阈值待基线确定）。
- **SC-004**: 每个核心改动（B 双路、C 定位、RAG、重排）均有相对基线的消融增益数据，可归因。
- **SC-005**: 全流程在 AutoDL 上可一键复现（固定种子+锁版本），他人按文档可重跑。

## Assumptions

- 推理时输入为无框干净 CT（真检测场景）；若部署始终带框，则 C 退化为 cv2 取框、任务改名（见 FR-009）。
- 绿框为细边框（病灶本体可见、可 inpaint 还原）；填充块情形需换数据策略。
- 训练用 LLaMA-Factory/ms-swift，基座 Qwen3-VL 4B 起步；GPU 为 AutoDL 单卡起步。
- 评估以固定测试集为准，基线为 vanilla 零样本 + 朴素 LoRA。
