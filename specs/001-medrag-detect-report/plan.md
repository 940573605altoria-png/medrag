# Implementation Plan: 医学多模态病灶检测与报告生成系统

**Branch**: `001-medrag-detect-report` | **Date**: 2026-06-21 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-medrag-detect-report/spec.md`

**Note**: 本 plan 仅记录已讨论敲定的部分（B 双路融合、C 高斯退火定位、整体架构）。数据集分工、
AutoDL 适配、RAG 链路实现细节标记为 TODO，后续讨论完再增量补入。

## Summary

基于 Qwen3-VL（共享 ViT）构建单模型多任务系统：辅助定位头（C）从干净 CT 预测病灶高斯热图 →
产出 ROI → 双路融合（B）把全局图与 ROI 细节图在视觉 token 层融合 → LLM 生成带证据溯源的报告；
外接 Qwen3-VL-Embedding + 父子层级检索 + 级联 + Reranker 抑制幻觉。方法论以固定基线 + 逐项消融
保证可归因（constitution III）。

## Technical Context

**Language/Version**: Python 3.11（conda env `F:\miniconda\envs\medrag`；AutoDL 侧复刻）

**Primary Dependencies**: Qwen3-VL（4B 起，按预算升 8B）、LLaMA-Factory 或 ms-swift（LoRA+多图+DeepSpeed）、
PEFT、LlamaIndex、Qwen3-VL-Embedding、Qwen3-VL-Reranker、Qdrant/Milvus、ragas、scispaCy/medspaCy、OpenCV

**Storage**: 向量库（Qdrant/Milvus，需标量过滤）；影像与清洗产物落数据盘（AutoDL `/root/autodl-tmp`）

**Testing**: pytest（逻辑）+ 固定测试集评估 harness（指标门控）

**Target Platform**: AutoDL 云 GPU（Linux）[NEEDS CLARIFICATION: 卡型/数量/预算]

**Project Type**: 单仓多模块（数据管线 + 训练 + RAG + 服务/Demo + 评估）

**Performance Goals**: 小病灶(<2%)召回↑、faithfulness↑/幻觉率↓、检测 mAP 可展示（阈值待基线定）

**Constraints**: 训推同分布（无标签泄露）；单卡可训（LoRA+梯度检查点）优先；可复现

**Scale/Scope**: 面试级垂直系统；数据集 a 药品 / b 医学QA / c CT-QA [NEEDS CLARIFICATION: 分工待定]

## Constitution Check

*GATE: 进入实现前必须通过；设计变更后复检。*

- **I. 证据可溯源**: ✅ 报告每条结论挂检索来源或定位 ROI；无证据拒答（FR-002/003）。
- **II. 防标签泄露与训推一致**: ✅ 绿框→坐标→热图标签，输入抹框；随机区域同样 inpaint 去相关二阶泄露（C 设计）。
- **III. 基线先行·逐项消融**: ✅ vanilla 零样本 + 朴素 LoRA 基线；B 三臂消融 + 小病灶分层；每改动单独开关。
- 结论：当前设计**无 NON-NEGOTIABLE 违例**。DeepSpeed 等复杂度见 Complexity Tracking。

## Project Structure

### Documentation (this feature)

```text
specs/001-medrag-detect-report/
├── spec.md       # WHAT/WHY（已填）
├── plan.md       # 本文件：HOW（B/C 已填，其余 TODO）
├── research.md   # 待：数据集/编码器/AutoDL 选型调研
├── data-model.md # 待：样本/标签/证据/评估记录的数据结构
└── tasks.md      # 待：/speckit-tasks 生成
```

### Source Code (repository root)

```text
src/
├── data/          # ingestion、去标识、NER 筛选、绿框→标签+抹除、coreset
├── models/        # Qwen3-VL 封装、双路融合(B)、辅助定位头(C)、自定义损失
├── train/         # LLaMA-Factory/ms-swift 配置、LoRA、DeepSpeed(可选)
├── rag/           # embedding 索引、父子层级检索、级联、reranker
├── eval/          # 评估 harness：ragas + 临床指标 + 检测 mAP + 分层
└── serve/         # 端到端 Demo（CT→定位→检索→带溯源报告）

tests/
├── integration/   # 端到端管线
└── unit/          # 数据/损失/检索单元
```

**Structure Decision**: 单仓多模块；训练不手写循环（走 LLaMA-Factory/ms-swift），故 `train/` 主要是配置与脚本。

---

## B — 双路融合（已敲定：全做）

**动机**: CT 病灶占比小、特征被背景稀释。双路 = 全局图 + ROI 放缩图，保全局又留局部。

**注入点**: merger（2×2 token 合并）之后、进 LLM 之前的视觉 token 序列；改 `modeling_qwen3_vl.py`
中"图像特征 scatter 进 input_embeds"的 forward。

**三臂消融（核心实验）**:
| 臂 | 公式 | token数 | 备注 |
|---|---|---|---|
| 相加 | `V = V_g + tanh(α)·g(V_roi)` | 不变(省显存) | 需黑边填充凑等长；空间错位为固有代价 |
| 拼接 | `V = [V_g ; V_roi]` | ×2 | 无错位、信息全留 |
| 多图输入 | `<img>全局<img>ROI` | ×2 | 在预训练分布内、零改架构、**最强基线** |

**门控残差（相加臂关键）**: `V_fused = V_global + tanh(α)·g(V_roi)`，**α 初始化为 0** → 起步即等于
预训练分布（Flamingo gated-xattn / ReZero / LoRA 同款零初始化）；g 为轻量 linear/MLP 对齐；α 标量起步。

**黑边填充隐患**: 大块纯黑偏 OOD 且致 ROI token 空间错位 → 仅"相加"臂被迫如此；多图/拼接无需填充。

**评估**: 按病灶面积**分层**报指标，证明增益集中在小病灶(<2%)、大病灶几乎无变化；必配 global-only 单路基线。

## C — 高斯退火定位损失（已敲定：框转标签+抹除+定位头）

**数据处理（消除泄露）**:
1. `cv2.inRange` 提取绿框坐标 → 生成宽方差高斯热图 target；
2. inpaint 抹掉绿框像素 → 干净 CT 作输入（训推同分布）；
3. ViT 视觉特征上的**辅助定位头**预测热图（= 推理时无框定位来源 = 系统"检测器"）。

**二阶防护**: 随机在非病灶区也做同样 inpaint，使补丁痕迹与病灶去相关；细边框用高质量 inpaint。
[NEEDS CLARIFICATION: 绿框为细边框 or 填充块]。

**损失（治类别极不平衡）**: CenterNet penalty-reduced focal + Dice：
```
L_loc = -1/N Σ { (1-p)^α·log(p)         , Y=1
                 (1-Y)^β·p^α·log(1-p)    , 否则 }   (α=2,β=4)
L = L_报告生成(LM) + λ(t)·(L_focal + γ·L_dice)
```
**退火（粗→细课程）**: target 高斯 σ(t) 由大变小（早期宽=稳收敛，后期窄=要精度）——驱动模型自缩
不确定区的真正机制；可叠 λ(t) 衰减或早期输入淡框 α(t):1→0（末期必须无框）。

---

## 待补模块（后续讨论后增量并入）

- **数据集分工**: a 药品 + b 医学QA → RAG 文本库；c CT-QA → 视觉训练+检索。NER 用判别式医学模型。
  coreset(K-means+分层) 作假设验证；0.6B 改写先不做，需消融证明净增益。[TODO: 细化]
- **RAG 链路**: Qwen3-VL-Embedding 索引；LlamaIndex `HierarchicalNodeParser`+`AutoMergingRetriever`；
  级联(metadata 文本过滤→视觉)；Reranker Top-5。[TODO: 实现细节]
- **AutoDL 适配**: 镜像/环境复刻、`/root/autodl-tmp` 数据盘、学术加速、多卡启动与断点续训。[TODO]
- **评估 harness**: ragas + 实体F1/RadGraph-F1 + 检测 mAP/sensitivity@FP + 小病灶分层。[TODO: 实现]

## Complexity Tracking

| 违例/复杂度 | 为何需要 | 何时才引入 |
|---|---|---|
| 改 Qwen3-VL 源码（B 的 token 融合） | 面试核心创新点，需在 token 层注入 ROI | 多图基线跑通后再上 add/concat 臂 |
| DeepSpeed | 仅多卡/大模型训练才需 | 单卡 LoRA 阶段不引入，升 8B/多卡时切 ZeRO-2/3 |
| 自定义定位头+损失（C） | 无框定位与抗泄露的核心 | M4，基线与评估 harness 就绪后 |
