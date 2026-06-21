<!--
Sync Impact Report
- Version change: (template) → 1.0.0
- Ratification: initial adoption 2026-06-21
- Modified principles: all 5 placeholders → concrete principles
    I. 证据可溯源 (NON-NEGOTIABLE)
    II. 防标签泄露与训推一致 (NON-NEGOTIABLE)
    III. 基线先行 · 逐项消融 (NON-NEGOTIABLE)
    IV. 隐私与医疗合规
    V. 可复现与云环境适配
- Added sections: 技术栈与约束; 开发工作流与质量门; Governance
- Templates checked:
    ✅ .specify/templates/plan-template.md (Constitution Check 段对齐，无冲突)
    ✅ .specify/templates/spec-template.md (无强制段冲突)
    ✅ .specify/templates/tasks-template.md (任务分类兼容)
- Deferred TODOs:
    TODO(绿框样式): 细边框 vs 填充块，待确认，影响 inpainting 可行性
    TODO(数据集分工细节/AutoDL 适配): 后续讨论后并入 spec
-->

# 医学多模态 RAG（Qwen3-VL）系统 Constitution

本项目为求职面试项目：基于 RAG + Qwen3-VL 的"病灶检测 + 报告生成"垂直系统，针对通用 VLM 的
微小病灶特征湮灭与幻觉问题。以下原则为不可绕过的工程与方法论底线。

## Core Principles

### I. 证据可溯源 (NON-NEGOTIABLE)
模型输出的每一条临床结论 **MUST** 能追溯到具体证据来源——检索命中的文档/病例，或定位头给出的
病灶 ROI。禁止编造医学结论（drug/诊断/数值）。无证据支撑的断言 **MUST** 被标注为不确定或拒答。
理由：医疗场景下不可溯源的生成等于不可用，且是幻觉的温床。

### II. 防标签泄露与训推一致 (NON-NEGOTIABLE)
标签 **MUST NOT** 进入模型输入。CT 绿框只能用于生成监督目标（高斯热图），**MUST** 从输入图中
抹除，喂给模型的是与推理同分布的干净图。**MUST** 防范二阶泄露（如 inpainting 痕迹与病灶相关）——
缓解手段（随机区域同样处理、使痕迹去相关）为必做项。任何"训练有、推理无"的输入差异 **MUST** 被消除。
理由：标签泄露会让指标虚高、上线即崩，是面试与产品的双重红线。

### III. 基线先行 · 逐项消融 (NON-NEGOTIABLE)
任何改动 **MUST** 在固定基线（vanilla Qwen3-VL 零样本、朴素 LoRA）和固定测试集之上度量。
**MUST** 一次只引入一个变量；每个"提升"结论 **MUST** 有对应消融支撑。涉及小病灶的改进
（如双路融合 B）**MUST** 附按病灶面积的分层指标，而非仅总体平均。
理由：无基线、无消融的"提升"不可归因，面试官一问即破。

### IV. 隐私与医疗合规
数据 **MUST** 去标识化（去 PHI）。不得将受保护患者信息写入日志、向量库明文或外发请求。
医学术语/实体处理 **SHOULD** 使用判别式医学 NER（scispaCy/medspaCy/BERT-NER）而非小生成模型臆造。

### V. 可复现与云环境适配
实验 **MUST** 可复现：固定随机种子、锁定依赖版本、记录数据划分。代码 **MUST** 适配 AutoDL
云环境（环境复刻、数据盘 `/root/autodl-tmp`、学术加速、可一键启动与断点续训）。
重大坑与环境约定记录于项目 memory，避免重复踩坑。

## 技术栈与约束

- 基座：Qwen3-VL 30B-A3B (MoE)（激活小、单/双卡 LoRA 可训）。
- 训练：LLaMA-Factory 或 ms-swift（原生支持多图+LoRA+DeepSpeed），**不手写训练循环**。
  DeepSpeed 配置常备、默认关：单卡 LoRA+梯度检查点优先，多卡时切 ZeRO-2（不上 ZeRO-3）。
- 检索：Qwen3-Embedding（文本 a/b，4B 起）/ 多模态 VL 嵌入（c）索引 + LlamaIndex 父子层级检索 + 级联（文本过滤→视觉）+
  Qwen3-VL-Reranker Top-5；向量库用 ChromaDB（按数据集类型分 collection，支持 metadata 过滤）。
- 评估：ragas（faithfulness/context precision）**+ 临床指标**（实体级 F1、RadGraph-F1 等）
  **+ 检测 mAP/sensitivity@FP + 小病灶分层**。单一 ragas 不足以支撑"幻觉抑制"结论。

## 开发工作流与质量门

- 遵循 Spec-Kit：constitution → specify →（clarify）→ plan → tasks → implement。
- **质量门**：每个 milestone **MUST** 跑固定测试集评估 harness，与上一基线对比；增益不显著
  **MUST NOT** 进入下一阶段。每个改动可单独开关、单独测。
- 与本 constitution 冲突的设计（尤其 I/II/III）**MUST** 在 plan 的 Constitution Check 段显式说明并整改。

## Governance

本 constitution 优先于其它实践约定。修订 **MUST** 记录变更、理由与版本号，并按语义化版本：
MAJOR=原则移除/不兼容重定义，MINOR=新增原则/章节，PATCH=措辞澄清。
每次进入 plan/implement 前 **MUST** 复核是否违反三条 NON-NEGOTIABLE 原则。

**Version**: 1.0.0 | **Ratified**: 2026-06-21 | **Last Amended**: 2026-06-21
