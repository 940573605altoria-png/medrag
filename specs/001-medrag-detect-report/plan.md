# Implementation Plan: 医学多模态病灶检测与报告生成系统

**Branch**: `001-medrag-detect-report` | **Date**: 2026-06-21 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-medrag-detect-report/spec.md`

**Note**: 本 plan 仅记录已讨论敲定的部分（B 双路融合、C 高斯退火定位、整体架构）。数据集分工、
AutoDL 适配、RAG 链路实现细节标记为 TODO，后续讨论完再增量补入。

## Summary

基于 Qwen3-VL（共享 ViT）构建单模型多任务系统：辅助定位头（C）从干净 CT 预测病灶高斯热图 →
产出 ROI → 双路融合（B）把全局图与 ROI 细节图在视觉 token 层融合 → LLM 生成带证据溯源的报告；
外接 Qwen3-Embedding（文本 a/b）+ 父子层级检索 + 级联 + Qwen3-VL-Reranker 抑制幻觉。方法论以固定基线 + 逐项消融
保证可归因（constitution III）。

## Technical Context

**Language/Version**: Python 3.11（conda env `F:\miniconda\envs\medrag`；AutoDL 侧复刻）

**Primary Dependencies**: Qwen3-VL 30B-A3B (MoE)、LLaMA-Factory 或 ms-swift（LoRA+多图+DeepSpeed ZeRO-2）、
PEFT、LlamaIndex、Qwen3-Embedding 4B（文本，a/b）+ c 图像编码器（医学 CLIP BiomedCLIP/PMC-CLIP vs 通用 VL GME/Qwen-VL，待实测）、Qwen3-VL-Reranker、ChromaDB、LlamaIndex BM25Retriever/rank_bm25（hybrid 检索）、ragas、scispaCy/medspaCy + 中文医疗 BERT-NER（CMeEE/CCKS）、datasketch、scipy/scikit-learn（bootstrap/显著性）、MCP Python SDK/FastMCP（对外服务化）、OpenCV

**Storage**: ChromaDB（按数据集类型分 collection，支持 metadata `where` 过滤）；影像与清洗产物落数据盘（AutoDL `/root/autodl-tmp`）

**Testing**: pytest（逻辑）+ 固定测试集评估 harness（指标门控）

**Target Platform**: AutoDL 云 GPU（Linux）；DeepSpeed ZeRO-2 配置就绪、默认可关；预算充足，资源以"保证完成任务"为准（高显存卡 A100/H100 级，必要时多卡），卡型按 AutoDL 可用选、不预锁

**Project Type**: 单仓多模块（数据管线 + 训练 + RAG + 服务/Demo + 评估）

**Performance Goals**: 小病灶(<2%)召回↑、faithfulness↑/幻觉率↓、检测 mAP 可展示（阈值待基线定）

**Constraints**: 训推同分布（无标签泄露）；单卡可训（LoRA+梯度检查点）优先；可复现

**Scale/Scope**: 面试级垂直系统；数据集 a 药品 / b 医学QA / c CT-QA 均入 ChromaDB（按类型分 collection），c CT-QA 同时用于视觉训练

## Constitution Check

*GATE: 进入实现前必须通过；设计变更后复检。*

- **I. 证据可溯源**: ✅ 报告每条结论挂检索来源或定位 ROI；无证据拒答（FR-002/003）。
- **II. 防标签泄露与训推一致**: ✅ 绿框→坐标→热图标签，输入抹框；随机区域同样 inpaint 去相关二阶泄露（C 设计）。
- **III. 基线先行·逐项消融**: ✅ vanilla 零样本 + 朴素 LoRA 基线；B 三臂消融 + 小病灶分层；每改动单独开关。
- **隐私合规（PHI，FR-007）**: ⚠️ QA-conflict 用外部 LLM-judge（tasks T040）→ **送审前 MUST 去标识、无 PHI 外发**（已在 T040 加前置去标识）。
- **设计偏差（兼容 constitution，记档防误读为违例）**: ① 临床指标用「域无关 实体F1+关系F1」替 RadGraph/CheXbert——constitution「RadGraph-F1 等」为非穷举，且数据未限定胸部；② c 用「专用冻结图像编码器(BiomedCLIP/GME)+分子库」替「统一多模态 VL 嵌入」——reranker 为 cross-encoder、分治无向量空间冲突（见「RAG — c 多模态嵌入」节）。
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
├── data/          # ingestion、去标识、a/b去重(精确+MinHash,QA冲突保留)、NER 覆盖率+多信号质量分(中英分治,低覆盖软降权)、绿框→标签+抹除、coreset(面积分层+层内K-means+三档选样)
├── models/        # Qwen3-VL 封装、双路融合(B)、辅助定位头(C)、自定义损失
├── train/         # LLaMA-Factory/ms-swift 配置、LoRA、DeepSpeed(可选)
├── rag/           # embedding 索引、父子层级检索、级联、reranker
├── eval/          # 评估 harness：ragas + 临床指标 + 检测 mAP + 分层
└── serve/         # 端到端 Demo + FastMCP server（暴露 generate_report/detect_lesions/retrieve_evidence/medical_qa 给 LangGraph client）

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

**二阶防护**: 随机在非病灶区也做同样 inpaint，使补丁痕迹与病灶去相关。绿框已定为**细边框 + 高质量 inpaint**（病灶本体可见、可还原）。

**损失（治类别极不平衡）**: CenterNet penalty-reduced focal + Dice：
```
L_loc = -1/N Σ { (1-p)^α·log(p)         , Y=1
                 (1-Y)^β·p^α·log(1-p)    , 否则 }   (α=2,β=4)
L = L_报告生成(LM) + λ(t)·(L_focal + γ·L_dice)
```
**退火（粗→细课程）**: target 高斯 σ(t) 由大变小（早期宽=稳收敛，后期窄=要精度）——驱动模型自缩
不确定区的真正机制；可叠 λ(t) 衰减或早期输入淡框 α(t):1→0（末期必须无框）。

---

## 数据清洗 — coreset 选样（已细化，CT 训练数据）

**动机**: 跑全量前用分布保真小子集快速验证 B/C 增益，省 AutoDL 预算（契合 constitution III）。
仅作用于 **c CT-QA 训练子集**；a/b 是 RAG 库（聚类目的为去重/覆盖，不同嵌入空间，分别处理、不混聚）。
**前提**: CT 只有绿框/可算病灶面积，**无疾病标签** → 面积为分层主轴，子群靠无监督聚类发现。

**分层（嵌套）**:
1. **轴1 = 病灶面积带**（`<2% / 2–5% / >5%`），`<2%` **过采样** —— 锁住小病灶覆盖（项目核心命题）。
2. **层内 K-means** 发现疾病/形态子群；特征用**冻结的通用图像/embedding 编码器**（**不用训练中模型自身特征**，避免循环耦合）；K 由 silhouette/elbow 选或固定小 K。

**每簇选样 = 三档结合 + 去噪护栏**:
- 先**滤掉最远 top-X%**（如 5%）疑似离群/伪影（或人工抽查）再选样；
- ① **近质心原型**（典型）+ ② **中等距离样本**（距质心中位数附近，过渡）+ ③ **最远点采样 FPS/k-center**（彼此最远，边界多样性覆盖）；
- **配额随簇大小**（√size），小病灶带整体过采样；不一刀切固定条数。

**铁律**: 固定随机种子、记录划分 → 可复现（SC-005）；coreset **只动训练子集，绝不碰固定评估集**（防归因污染）。

**验证时点**: M5「coreset 收益验证」—— 小子集上的 B/C 增益结论须与全量一致，否则 coreset 不可信。

---

## 数据清洗 — a/b RAG 去重（已细化，文本知识库）

**动机**: 语料重复 → 占满 Top-K 检索槽、证据冗余、reranker 见雷同候选 → 放大"重复多遍的错误"（多数错觉）→ 伤 faithfulness，并膨胀索引。去重服务的是**溯源质量**，非单纯清噪。

**范围/时点**: **库内**去重（a、b 各自，不跨库）；**文档级先去重，再做父子层级切块**（避免破坏 hierarchy）。
管线：ingestion → 去标识 → **去重** → NER 覆盖率筛选 → 父子切块 → embedding → 入 ChromaDB collection。

**力度（已定）**: 精确 + MinHash 近重复，**不上语义 embedding 去重** → 中/英同义版本都保留，护住语言覆盖。

**阶段**:
1. **归一化**: NFKC、空白/标点折叠、剔除样板（免责声明/页眉页脚）；**医学数值/单位/剂量不归一**（`5mg`≠`50mg`）。
2. **精确去重**: 归一化文本哈希（SHA1）删完全相同。a 的结构化字段另做 key 级精确去重（药名+章节）。
3. **近重复 MinHash+LSH**（datasketch）: word/char n-gram shingle，Jaccard ≥ τ（保守，如 0.85）→ 候选对 → 合并；保守阈值保覆盖。
4. **b 医学QA 特例**: 以 **Q+A 整体**为去重单位；MinHash 检出「问近似 / 答不同」→ 交 **LLM-judge（API，temp=0，记录模型/prompt 版本）** 判四档：
   - **both-correct-互补** → **LLM-merge** 整合成更好答案；**约束只整合两输入事实、不准添新事实**；存活节点**保留两个来源 ID + 标 `llm_merged`**（合成非原文，可溯源）。
   - **both-correct-冗余** → 普通近重复合并取一。
   - **矛盾 / 一方错** → **不合并，两条都留并标 `conflict`**（防把幻觉判成"对"）。
   - **判别不确定** → 保留 + 标记（fail-safe）。
   中/英双语**不语义去重，双语都留**。
   ⚠️ LLM-merge 属"LLM 改写数据"，与铁律「0.6B 改写先不做，需消融证明净增益」同源 → 设为**可消融开关**，须消融证明净涨 faithfulness 才保留。

**保溯源（constitution I）**: 被删/合并副本的来源 ID 并入存活节点 metadata；LLM 合成节点额外标 `llm_merged`，证据不丢失。
**可复现/可消融**: 固定阈值与种子、记录去重决策；去重 on/off 可消融，量化索引大小↓、检索多样性↑、faithfulness↑且覆盖无回退。
**覆盖护栏**: 去重前后用 NER 实体集对比，确认未删掉独有实体（与下游 NER 覆盖率筛选联动）。
**出范围**: 检索时 MMR 多样性是互补手段，归 RAG 链路，不在此。

---

## 数据清洗 — NER 覆盖率筛选 + 质量度量（已细化，文本知识库）

**动机**: 保证 RAG 库**实体覆盖**（缺概念→检索无据→被迫拒答）且**剔低质语料**。用**判别式医学 NER**（不编造实体，铁律 II）。

**位置**: `src/data/` 管线，**去重之后、父子切块之前**。
**双语分治**: langid 判定语种 → 英文 scispaCy/medspaCy(+UMLS linker)、中文 CMeEE/CCKS 训练的医疗 BERT-NER；中英实体类型**对齐到统一 schema**（drug/disease/anatomy/finding/procedure），否则覆盖统计不可比。
**护栏（铁律推论）**: 覆盖基准**绝不用评估集实体**（防反向泄露）。

**数据质量度量（多信号打分）**:
1. **实体信号(NER)**: 实体密度、类型多样性、**实体可链接率**（链到 UMLS/本体比例，低=泛词/噪声）、稀有实体命中（外部本体里的稀有概念→提权）。
2. **文本信号**: 语种一致性/乱码检测、长度信息量、样板比例（与去重联动）、可选小 LM 困惑度滤乱码。
3. **结构信号(a 结构化)**: 关键字段填充率（药名/适应症/禁忌/剂量非空率）。
4. **可信信号**: 复用去重的 `conflict`/`llm_merged` 标记 + 来源权威度。

**判定与动作（结合式）**:
- 多信号组合成**质量分**（规则门 + 加权）；
- **零医学实体 + 低文本质量 → 硬丢**（纯样板/乱码）；
- **低覆盖但含稀有实体 → 保留 + 软降权标记**（检索时降权，不丢知识）；
- 质量分与各信号**落 metadata** → 供检索加权、消融、溯源；**小规模人工抽检**校准阈值。

**覆盖基准**: 任务域实体类型密度阈值（质量门）+ 外部本体（UMLS/MeSH/药品表）稀有实体覆盖保障。
**覆盖护栏**: 去重/coreset 前后用统一 schema 实体集对比，确认未删独有实体。
**可复现/可消融**: 固定阈值与模型版本、记录每条质量分与决策（SC-005）；质量筛选设**可消融开关**，on/off 看 faithfulness/覆盖/索引大小。

---

## RAG — 父子层级分块（已细化）

**范式（已定）**: 结构层级 `AutoMergingRetriever`——**只索引子块、命中多则上浮返回父块**；**父=大段连续原文，非 LLM 摘要**（无损、可溯源）。RAPTOR 摘要式父块是另一范式，本链路不用。

**统一骨架**: LlamaIndex `HierarchicalNodeParser`（chunk_sizes 起步两层 `[1024, 256]` token，带 min/max 护栏）产父子+关系 → **子/叶块入 ChromaVectorStore，全节点入 docstore** → `AutoMergingRetriever` 上浮。

**切分方式（已定）**: **结构感知 + 尺寸约束**（子块可选叠 `SemanticSplitterNodeParser`）；非纯语义双层（贵、块大小不可控、对 AutoMerging 不友好）。

**按数据类型分治**:
- **a 药品（结构化）**: 文档结构即层级 —— 父=药品条目/主章节（适应症、禁忌、用法用量…），子=字段/子段（句界+尺寸）。metadata 存药名、章节类型（供级联文本过滤）。结构即层级，不需语义切分。
- **b 医学QA**: 短 QA → **flat 单节点，embed Q+A**；长答案 QA → 父=整条 QA、子=答案子片段（结构感知+尺寸，子可叠语义），**每子块把 question 挂 metadata 并前置进 embed 文本**（防孤儿块）。双语不混；`conflict`/`llm_merged` 随节点带走。
- **c CT-QA（图文）**: 文本（报告/QA）同理切，图与其报告/ROI 用 metadata 关联；级联中 c 走视觉路。

**实现要点**: 结构感知用自定义/`MarkdownNodeParser` 解析再交 hierarchical；contextual（前置 question/章节标题）用 metadata 模板纳入 embed 文本、不污染展示文本。
**可复现/可消融**: chunk_sizes、层数、是否叠语义、是否 contextual 均设**开关**，单独消融看检索命中率/faithfulness（constitution III）。

---

## RAG — embedding 接入（a/b + c 已细化）

**a/b 文本嵌入（已定）**: **Qwen3-Embedding 4B**（多语，中/英一模搞定、支持跨语检索，配套"双语都留"）。
- **距离 = cosine**（向量归一化），ChromaDB collection 配 cosine space。
- **非对称查询编码**: query 加模型推荐指令前缀（+ 可选医学任务指令）；document 端不加。
- **训推一致（铁律推论）**: 索引端与查询端**同一模型 + 同一预处理**（contextual 前置、归一化）；固定模型版本，可复现。
- 索引对象 = 分块阶段的子/叶块文本（含 contextual 前置）。

**架构 = 分治（已定）**: a/b 文本嵌入器 + c 多模态嵌入器，各自空间、不混。
- **关键依据（记档防误判）**: embedding 是 bi-encoder、reranker 是 **cross-encoder**——reranker 重新读 (query, 候选原文) 打分，**不消费 embedding 向量、不共享向量空间**。故分治召回 + Qwen3-VL-Reranker 统一重排**无冲突**；各 collection 独立检索、从不跨空间比较向量；级联分阶段（文本过滤→视觉）也不需单一跨模态空间。统一多模态嵌入器会牺牲 a/b 纯文本检索质量去解一个本架构下不存在的问题，故不取。

## RAG — c 多模态嵌入（已细化，CT 图文病例检索）

**用途**: 相似 CT 病例检索（支撑报告生成的证据溯源）+ 级联视觉阶段；查询可为图（输入 CT）或文本。

**病例多向量表征（已定）**: 同一 `case_id` 存多向量——① **报告文本向量**（Qwen3-Embedding，供级联文本过滤）② **图像全图向量** ③ **图像 ROI/病灶区向量**（②③同一图像编码器）。图/文查询都支持。
**嵌入粒度（已定）= 全图 + ROI 双向量**: 全图抓全局上下文、ROI 抓小病灶判别力（呼应 B 双路、对抗背景稀释）。ROI 来源：索引端用绿框坐标，查询端用 C 定位头输出（系统"检测器"）——**裁剪区来自检测器、再用独立冻结编码器嵌入，不算循环耦合**。
**图像编码器（待实测再定）**: 医学专用 CLIP（BiomedCLIP/PMC-CLIP，CT 强、文本偏英）vs 通用 VL（GME/Qwen-VL，多语、CT 偏弱）——上 AutoDL 在小集上比检索指标定夺。文本向量恒用 Qwen3-Embedding（多语）。
**铁律（直接定）**: 检索用**独立冻结嵌入器**，不用正在微调的模型自身 ViT 特征（防循环耦合，同 coreset 铁律）。

**⚠️ ChromaDB 库结构（维度约束）**: 文本向量与图像向量**维度不同，不能同 collection**。故 c 物理上拆**子-collection**：`c_text` / `c_img_whole` / `c_img_roi`（后两者同编码器同维，可合一带 `vector_type` metadata），用 `case_id` 串联。检索时多通道各自召回 → 候选取并集 → reranker 统一裁决。
**可复现/可消融**: 全图/ROI 双路、图像编码器选型、多通道分数融合权重 均设**开关**，单独消融看相似病例检索指标。

---

## RAG — 级联检索 + Reranker（已细化）

**整体漏斗**: 多通道 hybrid 召回 → AutoMerge 上浮父块 → 分阶段级联 + RRF 融合 → Qwen3-VL-Reranker 重排 → 拒答门 → Top-5。

**文本检索 = BM25 + dense 混合（已定）**: 药名/编码/剂量等精确词靠 BM25、语义靠 dense（Qwen3-Embedding），**RRF 融合**（基于排名、跨通道鲁棒）。实现复用 LlamaIndex `QueryFusionRetriever(reciprocal_rerank)` + `BM25Retriever`(docstore) + 向量检索，**不手写**。
- **阈值不卡 BM25 raw score**（无界、随语料变、不可比）→ 过滤用 **top-N / 归一化分**；最终**拒答阈值放 rerank 之后**的统一分上。

**c / 多模态查询两级级联（已定，用户设计）**:
1. 文本级 hybrid→RRF→阈值过滤→**Top-50**（图库再大只比这 50，省图像匹配耗时）；
2. 在 50 内做图像向量检索（全图 + ROI 通道）→ 阈值过滤 → 文本+图像结果**二次 RRF**→**Top-10~20**；
3. 仅 Top-10~20 送 reranker。

**报告生成图像流 = query-adaptive（已定）**: C 检测头草稿 findings/病灶带/器官属性 → 作文本 query 驱动上面文本级联；**并行图像主通道（全局 ANN top-K）兜底**，防草稿空/错过度过滤；a/b 文本知识作补充证据。文本/QA 查询走文本先行；图像查询走视觉为主+文本桥接。
**a/b 纯文本知识**: 仅 hybrid→RRF→rerank，无图像阶段。

**Reranker（已定）**: Qwen3-VL-Reranker，**cross-encoder、多模态**（候选含图像、query 可为图，这是选 VL reranker 的原因）；**先 AutoMerge 父块再 rerank**；输出 **Top-5**。
**拒答门（FR-003 强制）**: rerank 后最高分低于阈值 → 拒答/标不确定，绝不编造；阈值评估集校准。

**可复现/可消融**: hybrid/RRF、各阶段阈值、query-adaptive 分支、图像兜底通道、rerank 漏斗(10~20)、拒答阈值 均设**开关**消融（constitution III）。

---

## 评估 harness（已细化）

**架构**: **配置驱动的消融 runner** —— 一个 run = `{组件开关(B/C/RAG/reranker/dedup/coreset/hybrid/query-adaptive…) + checkpoint + 固定测试集 + seed}` → 产出结构化**评估记录**（config → metrics）→ 自动 vs 上一基线出 **delta 表**。

**固定测试集**: held-out、**按病灶面积分层**（<2%/2–5%/>5%）、**绝不被 train/coreset/dedup/NER 覆盖 touch**；检测有 GT 框/mask；报告/QA 有 GT 答案 + 证据。
**基线**: vanilla 零样本 + 朴素 LoRA（固定参照）；**一次一变量**消融；**小病灶分层贯穿所有指标**。

**分任务指标（已定）**:
- **检测**（C 定位头）: **FROC / sensitivity@FP 主 + mAP 辅**，按病灶面积分层。
- **报告生成**: **域无关 实体级 F1 + 关系 F1**（**复用 NER 覆盖率那套医学 NER** 抽实体/关系、生成 vs 参考；非胸部不绑 RadGraph/CheXbert）；+ faithfulness（见下）；NLG（BLEU/ROUGE/BERTScore）仅作辅助代理。
- **RAG 检索**: recall@k / nDCG@k / MRR（需标注相关）+ **ragas**（faithfulness / context precision-recall）→ 幻觉率。
- **端到端**: 证据可溯源率 + **拒答正确性**（abstention precision/recall：正确拒答 vs 过度拒答，对接 FR-003）。
- **ragas LLM judge**: 固定 judge 模型 temp=0、记录版本（可复现）。

**统计显著性（已定）**: 固定测试集上 **bootstrap 置信区间 + 配对显著性检验** → 质量门"增益显著"判定（CI 重叠/p 不过 → 视为不显著）。
**质量门**: 每 milestone vs 上一基线，**增益不显著不进下一阶段**（constitution 质量门）。
**可复现**: 固定测试集 + seed + 锁版本（含 judge/NER 模型）；评估记录全留档。

---

## 部署与对外服务 — AutoDL + MCP（已细化）

**形态**: medrag 服务化为 **MCP server**，把医学影像+RAG 能力暴露为工具；**多Agent 医疗项目（LangGraph，经 langchain-mcp-adapters）作 client** 编排调用。AutoDL 同时托管 GPU 推理 + MCP server。

**Transport（双模，已定）**: 用 MCP Python SDK（FastMCP）一套 server、transport 可切——
- **远程为主**: streamable HTTP(/SSE) + 鉴权 token + TLS + **AutoDL 自定义服务端口/隧道映射**；
- **保留本地**: 同机 stdio / localhost（开发或同机部署省网络）。

**暴露的 MCP 工具（4 个；输入/输出 schema 明确、结果带溯源）**:
1. `generate_report(image)` → 结构化报告 + 每条结论的证据引用/ROI（端到端）。
2. `detect_lesions(image)` → ROI 列表 + 热图 + 病灶面积带（C 定位头）。
3. `retrieve_evidence(query, top_k, filters)` → Top-K 证据 + 来源/分数（RAG 级联，不生成）。
4. `medical_qa(question)` → 带引用回答（a/b 知识库）。
- 所有工具输出**结构化 + 携带证据/引用/置信**（constitution I，跨项目不丢溯源）；**无证据按 FR-003 在结果里标拒答/不确定**，不编造。

**服务层架构**: `src/serve/` —— FastMCP server 薄层包住 medrag 推理服务（GPU 模型常驻进程 + ChromaDB 连接 + 检测/RAG 管线）；MCP 工具调用映射到管线函数。**MCP 仅是接口层，与本地直调同管线** → 评估 harness 照测管线，归因不受接口影响。

**AutoDL 适配（已定）**: 镜像/conda env 复刻（锁版本）；`/root/autodl-tmp` 数据盘放权重/向量库/影像/产物；学术加速（github/hf 镜像）；多卡 **ZeRO-2** 启动脚本 + 断点续训；服务进程守护（健康检查/重启）+ 端口映射对接 MCP。**资源（已定）**: 预算充足，以"保证完成任务"为准——按 AutoDL 可用选高显存卡（A100/H100 级），必要时多卡上 ZeRO-2；单卡 LoRA 为底线；卡型不预锁、配置代码对卡数自适应。

**安全/合规**: 鉴权 token + 远程 TLS；PHI 不出日志（FR-007）；工具输入校验（图像类型/大小）。

---

## 待补模块（后续讨论后增量并入）

- **数据集分工（✅ 已细化完成）**: a 药品 + b 医学QA → RAG 文本库（去重 + NER 覆盖率/质量筛选见上两节）；c CT-QA → 视觉训练+检索（coreset 见上）。
  不同嵌入空间分别处理、不混聚。0.6B 改写先不做，需消融证明净增益。
- **RAG 链路（✅ 已细化完成）**: ChromaDB collection（药品 / 医学QA / CT-QA；c 因多向量维度不同拆 `c_text`/`c_img_whole`/`c_img_roi` 子库）；父子层级分块 + embedding(a/b + c 多模态) + 级联检索 + Reranker —— 均见上各节。
- **部署与对外服务（✅ 已细化完成）**: AutoDL 托管 GPU+MCP server；FastMCP 双模 transport（远程 HTTP/SSE + 本地 stdio）；暴露 4 工具给 LangGraph client —— 见上节。仅余卡型/预算待定。
- **评估 harness（✅ 已细化完成）**: 配置驱动消融 runner；检测 FROC/sensitivity@FP+mAP、报告 域无关实体F1+关系F1、RAG ragas+recall@k/nDCG/MRR、端到端 溯源率+拒答正确性、bootstrap CI+配对检验、小病灶分层贯穿 —— 见上节。

## Complexity Tracking

| 违例/复杂度 | 为何需要 | 何时才引入 |
|---|---|---|
| 改 Qwen3-VL 源码（B 的 token 融合） | 面试核心创新点，需在 token 层注入 ROI | 多图基线跑通后再上 add/concat 臂 |
| DeepSpeed | 仅多卡/大模型训练才需 | 配置常备、默认关；多卡时切 **ZeRO-2**（不上 ZeRO-3）。基座 30B-A3B MoE 激活小，单/双卡 LoRA 可训 |
| 自定义定位头+损失（C） | 无框定位与抗泄露的核心 | M4，基线与评估 harness 就绪后 |
