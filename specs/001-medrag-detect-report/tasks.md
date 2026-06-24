---
description: "Task list for 医学多模态病灶检测与报告生成系统"
---

# Tasks: 医学多模态病灶检测与报告生成系统

**Input**: Design documents from `specs/001-medrag-detect-report/`
**Prerequisites**: plan.md ✅, spec.md ✅, constitution.md ✅（research/data-model/contracts 未单独生成 → 数据契约在 T0xx 内落为代码 schema）

**Tests**: plan.md 明确 `pytest（逻辑）+ 固定测试集评估 harness`，故含针对性单元/集成测试任务（非全 TDD）。

---

## 📍 实现进度快照（2026-06-24，冷启动看这里；权威叙述见根 CLAUDE.md 看板）

**已完成并测绿**（本地 pytest **94 passed / 12 skipped**；torch 部分在 AutoDL 真跑）：
- 基建/骨架：T001 T002 T003 T004 T005 T006 T007 T011 T013 + 桩端到端骨架 T020–T023（B1/B2 批）。
- **US1 创新 C 定位链**：T024 T025 T026 T027 T028 T029 ✅。
- **US1 创新 B**：T030 ✅ 仅 `models/fusion.py` 模块（三臂、α=0 恒等已测）；⏳ **T012 + 接进 `modeling_qwen3_vl.py` 未做**（AutoDL 交互式做）。
- **US1 报告**：T034 ✅ `models/report.py`（引用标签强制溯源、禁编造、拒答）。
- **US3 评估 harness（全套，归因底座闭合）**：T016 runner/record + T049 检测 + T050 报告F1 + T051 RAG指标 + T052 端到端 + T053 显著性 + T054 消融矩阵 ✅。

**⚠️ 已写代码但本地跑不全、必须 AutoDL 验证**（对应 9 个 torch-skipped 测）：
- T028 `loc_head.py` / T029 `losses.py` / T030 `fusion.py` 的前向/损失/门控测 → AutoDL 装 torch 真跑。
- **T012 + T030 后半（B 融合接线）= 还没写**：AutoDL vendor transformers 5.12.1 的 `modeling_qwen3_vl.py` 交互式插 `DualPathFusion`。
- T011 4B 已 L1 PASS；30B 需 A100-80G。T051 ragas 裁判段需 `ragas`+`DASHSCOPE_API_KEY`。

**下一步建议顺序**：① **T056 质量门 + T055 B消融报告**（纯逻辑、本地可测、依赖已就绪）→ ② US2 文本 RAG 链 T014/T043/T044/T045/T046/T047（结构本地测/功能 AutoDL）→ ③ T012+T030后半 / T031/T032/T033 / T035 训练 / T036 接回管线（均需 AutoDL/GPU）。

**怎么继续**：模块守卫导入（torch/cv2/sklearn/chromadb 缺了也能 import）；纯逻辑本地 `python -m pytest -q` 跑绿，重依赖功能测 `importorskip` 留 AutoDL；改完 commit，用户 `push_github.ps1` 推，AutoDL `git pull` 验。

---

## 排期原则：先总体骨架，再逐个局部（walking skeleton）
1. **接口契约先行**（Foundational）：所有组件 coding 前先定共享数据 schema。
2. **端到端骨架先跑通**（Phase 3，桩实现）：pipeline + 最小 MCP server 用 stub 串通端到端，锁接口/集成/服务。
3. **逐个组件用真实实现替换骨架桩**（US1/US2/US3）：每个组件做完都插回骨架、对着"活的整体"端到端验证。
4. **方法论底座前置**（constitution III）：固定测试集 + 基线 + 评估骨架在 Foundational 即建；增益不显著不进下一阶段。

## Format: `[ID] [P?] [Story] Description`
- **[P]**: 可并行（不同文件、无未完成依赖）
- **[Story]**: US1/US2/US3（Setup/Foundational/Skeleton/Deploy/Polish 无标签）

---

## Phase 1: Setup（共享基建）

- [x] T001 按 plan.md 结构创建模块骨架 `src/{config,contracts,data,models,train,rag,eval,serve}/` 与 `tests/{unit,integration}/`（含 `__init__.py`）✅
- [x] T002 在 `pyproject.toml` 声明并锁版本依赖（torch、transformers、LLaMA-Factory/ms-swift、peft、deepspeed、llama-index、chromadb、rank_bm25、datasketch、ragas、scispacy/medspacy、mcp[fastmcp]、scipy、scikit-learn、opencv-python）✅
- [x] T003 [P] 全局配置加载 + 可复现（固定 seed、确定性开关）于 `src/config/config.py`、`src/config/seed.py` ✅
- [x] T004 [P] run-record/版本化日志工具（config→metrics、模型/数据版本留痕）于 `src/config/run_record.py` ✅
- [x] T005 [P] AutoDL 脚手架（conda env 复刻、`/root/autodl-tmp` 数据盘布局、学术加速）于 `scripts/autodl_setup.sh` ✅
- [x] T006 [P] 多卡启动 + 断点续训（DeepSpeed ZeRO-2、对卡数自适应、默认可关）于 `scripts/launch_train.sh`、`scripts/resume.sh` ✅

---

## Phase 2: Foundational（阻塞所有后续；含接口契约 + 归因底座）

**⚠️ CRITICAL**: 未完成前，骨架与 user story 均不得开工

- [x] T007 **定义共享数据契约/接口 schema**（`ROI`/`DetectionResult`(热图/框/面积带/置信)、`EvidenceItem`(source_id/score/citation/modality)、`RetrievalResult`(含 abstain)、`ReportResult`(结构化+每条结论证据链+不确定标注)、MCP `ToolIO`、`EvalRecord`、`KnowledgeNode`/`CTSample` 元数据）于 `src/contracts/schemas.py` ✅
- [ ] T008 实现 a/b/c 原始数据 ingestion 加载器于 `src/data/ingest.py`
- [ ] T009 [P] PHI 去标识化（FR-007）于 `src/data/deid.py`
- [ ] T010 [P] 病灶面积分层工具（`<2%/2–5%/>5%`）于 `src/data/stratify.py`
- [x] T011 Qwen3-VL 基座封装/加载器（processor、LoRA 挂载、共享 ViT 取特征）于 `src/models/qwen3vl.py` ✅（4B 已 AutoDL L1 PASS；30B 需 A100-80G）
- [ ] T012 [P] vendor 并预备 patch `modeling_qwen3_vl.py`（标注 merger 后视觉 token 融合注入点）于 `src/models/modeling_qwen3_vl.py`
- [x] T013 ChromaDB store + collection 管理（a_drug/b_medqa/c_text/c_img_whole/c_img_roi，cosine，metadata where）于 `src/rag/store.py` ✅
- [x] T014 [P] Qwen3-Embedding 4B 文本嵌入服务（非对称 query 指令、归一化、训推一致）于 `src/rag/embed_text.py` ✅（`format_query`(Instruct前缀)/`format_document`(不加)/`l2_normalize` 纯逻辑本地测；`TextEmbedder` 可注入 `encode_fn`，模型后端守卫导入待 AutoDL）
- [ ] T015 固定 held-out 测试集加载器（面积分层、never-touched 守卫）于 `src/eval/dataset.py`
- [x] T016 配置驱动评估 runner 骨架 + 评估记录 schema（归因底座，依赖 T007）于 `src/eval/runner.py`、`src/eval/record.py` ✅（runner 吃 `predict_fn`+`metric_fn` 与模型解耦；record 落盘/读取/拍平 EvalRecord）
- [ ] T017 [P] vanilla 零样本基线 runner 于 `src/train/baseline_vanilla.py`
- [ ] T018 [P] 朴素 LoRA 基线训练配置于 `src/train/configs/baseline_lora.yaml`
- [ ] T019 LLaMA-Factory/ms-swift 集成（LoRA+多图+ZeRO-2、不手写循环）于 `src/train/framework.py`

**Checkpoint**: 接口契约 + 基座/向量库/嵌入/评估骨架/基线就绪

---

## Phase 3: 端到端骨架（Walking Skeleton，桩实现）🧱

**Purpose**: 在深做组件**之前**，用 stub 把端到端 + MCP 服务串通跑起来，锁定接口/集成/服务边界。后续 US 全部往这具骨架里"填真肉"。

- [x] T020 桩组件实现契约：`stub_detect`(返回假 ROI)、`stub_retrieve`(返回空/假证据)、`stub_report`(模板报告) 于 `src/serve/stubs.py`（依赖 T007）✅
- [x] T021 串通端到端管线 CT→detect→retrieve→report（基于 stub，走 T007 契约）于 `src/serve/pipeline.py`（依赖 T020）✅（真实组件接回归 T036）
- [x] T022 最小 FastMCP server 暴露 4 工具（generate_report/detect_lesions/retrieve_evidence/medical_qa，桩管线驱动，本地 stdio）于 `src/serve/mcp_server.py`（依赖 T021）✅
- [x] T023 [P] 骨架冒烟测试：端到端跑通 + 4 个 MCP 工具可调 + LangGraph(langchain-mcp-adapters) 连通 于 `tests/integration/test_skeleton.py` ✅

**Checkpoint**: 活的整体已就位——接口/数据流/服务已验证；之后每个组件替换桩后都能对骨架端到端验证

---

## Phase 4: User Story 1 - 上传 CT 得到可溯源的检测+报告（Priority: P1）🎯 MVP

**Goal**: 用真实 C 定位 + B 融合 + c 检索 + 报告生成，替换骨架的 detect/retrieve(视觉)/report 桩。

**Independent Test**: 无框 CT 端到端带源报告，每条结论可点开证据；小病灶(<2%)召回显著高于单路 global-only 基线。

- [x] T024 [P] [US1] 绿框提取（`cv2.inRange`→坐标）于 `src/data/ct_box.py` ✅
- [x] T025 [US1] 框→宽高斯热图标签于 `src/data/ct_label.py`（依赖 T024）✅
- [x] T026 [US1] inpaint 抹框 + 二阶随机区域 inpaint（抗泄露、训推同分布，FR-004）于 `src/data/ct_inpaint.py`（依赖 T024）✅
- [x] T027 [P] [US1] CT coreset 选样（面积分层 + 冻结编码器 K-means + 三档 原型/中位距/FPS + 去噪 + √簇配额）于 `src/data/coreset.py` ✅
- [x] T028 [P] [US1] ViT 辅助定位头（推理无框可用）于 `src/models/loc_head.py`（实现 `DetectionResult` 契约）✅
- [x] T029 [P] [US1] CenterNet focal + Dice + σ(t) 退火损失于 `src/models/losses.py` ✅
- [~] T030 [US1] B 双路融合（门控残差相加 α0/拼接/多图 三臂，merger 后 token 层）patch `src/models/modeling_qwen3_vl.py` + `src/models/fusion.py`（依赖 T011, T012）— **`fusion.py` 模块 ✅；T012 vendor+接进 `modeling_qwen3_vl.py` ⏳ 待 AutoDL**
- [ ] T031 [US1] C 图像嵌入（全图+ROI 双向量，独立冻结编码器）于 `src/rag/embed_image.py`（依赖 T013）
- [ ] T032 [US1] c 病例多向量入库（case_id 串 报告文本+全图+ROI）于 `src/rag/index_ct.py`（依赖 T014, T031, T013）
- [ ] T033 [US1] c 检索 query-adaptive 级联（检测头草稿桥接 + 图像主通道兜底，实现 `RetrievalResult`）于 `src/rag/cascade_visual.py`（依赖 T032）
- [x] T034 [US1] **真实报告生成替换 stub_report**（结构化 + 每条结论挂证据/ROI + 无证据拒答，FR-002/003，实现 `ReportResult`）于 `src/models/report.py` ✅（引用标签 [S*]/[ROI*] 强制溯源；接回管线归 T036）
- [ ] T035 [US1] C+B 训练脚本（LoRA、σ 退火课程、各创新点消融开关）于 `src/train/train_cb.py`（依赖 T019, T028, T029, T030）
- [ ] T036 [US1] **把真实 detect/visual-retrieve/report 接回骨架管线**（替换 T020 桩）于 `src/serve/pipeline.py`（依赖 T028, T033, T034）
- [ ] T037 [P] [US1] 集成测试：干净 CT → 端到端带来源报告（对骨架）+ **断言检测草稿空/错时图像兜底通道触发（FR-005）** 于 `tests/integration/test_e2e_report.py`

**Checkpoint**: US1 端到端真实可演示（MVP）

---

## Phase 5: User Story 2 - 基于知识库的可溯源医学问答（Priority: P2）

**Goal**: 用真实 a/b RAG 替换骨架的 retrieve(文本)/medical_qa 桩。

**Independent Test**: 一组医学/药品问题，回答可溯源到来源且 faithfulness 达标。

- [ ] T038 [P] [US2] 文本归一化（NFKC、样板剔除、**医学数值/单位/剂量不归一**）于 `src/data/normalize.py`
- [ ] T039 [US2] 精确(SHA1)+MinHash/LSH 近重复去重（保守阈值、来源 ID 并入溯源）于 `src/data/dedup.py`（依赖 T038）
- [ ] T040 [US2] QA 同问不同答 LLM-judge 四档 + LLM-merge（`llm_merged`、保双来源、不加新事实、可消融）于 `src/data/qa_conflict.py`（依赖 T039, T009）；**送外部 LLM-judge 前 MUST 经去标识、无 PHI 外发（FR-007 / constitution 隐私）**
- [ ] T041 [P] [US2] 双语医学 NER（英 scispaCy/medspaCy + 中文 CMeEE/CCKS BERT-NER，统一 schema）于 `src/data/ner.py`
- [ ] T042 [US2] NER 覆盖率+多信号质量筛选（密度/可链接率/稀有命中；零实体硬丢、低覆盖软降权；基准不含评估集）+ **去重/筛选前后实体集对比覆盖护栏（断言未删独有实体，FR-012）** 于 `src/data/ner_quality.py`（依赖 T041）
- [x] T043 [P] [US2] 父子层级分块（AutoMerging、结构感知+尺寸、短 QA 单节点/长答案父子、子块前置 question）于 `src/rag/chunk.py` ✅（QA 路 `chunk_qa` 纯逻辑本地测：短答案单叶/长答案父子、子块 embed 前置 question 不污染展示文本、`to_knowledge_nodes` 映射；文档路 `HierarchicalNodeParser` 守卫导入待 AutoDL）
- [x] T044 [US2] a/b 叶块入库 + docstore 承载 AutoMerging 于 `src/rag/index_text.py`（依赖 T043, T014, T013）✅（叶块嵌入入 chroma collection + 全节点入 `DocStore`(父块仅 docstore，承 AutoMerging 上浮)；chromadb 本地已装→入库端到端**本地功能测**(内存库+桩嵌入)，真实嵌入模型待 AutoDL）
- [x] T045 [US2] hybrid 文本检索（`BM25Retriever`+dense+`QueryFusionRetriever` RRF；top-N/归一化分过滤）于 `src/rag/retrieve_text.py`（依赖 T044）✅（RRF/AutoMerging 上浮纯逻辑本地测；dense 路内存 chroma 端到端测；BM25 用 `rank_bm25` 守卫；**架构对齐自建栈，RRF 直接实现而非套 LlamaIndex `QueryFusionRetriever`**；拒答阈值留 T046，仅无候选时 NO_EVIDENCE）
- [x] T046 [US2] Qwen3-VL-Reranker cross-encoder Top-5（先 AutoMerge 再 rerank）+ 拒答门 于 `src/rag/rerank.py`（依赖 T045）✅（重排/Top-K/sigmoid 统一分/拒答门(top<min_score→LOW_CONFIDENCE)纯逻辑本地测；上游已拒答则透传；`score_fn` 可注入，CrossEncoder 后端守卫待 AutoDL）
- [x] T047 [US2] **真实 medical_qa 接回骨架/MCP**（带引用，替换桩）于 `src/serve/qa.py`（依赖 T046）✅（`MedicalQA`：检索(T045)→精排+拒答(T046)→**复用 `assemble_report`** 出带 [S*] 引用回答、无据标 uncertain/低置信拒答；`Pipeline(qa_service=)` 路由→MCP `medical_qa` 自动走真实路，默认仍桩向后兼容；检索/精排/草稿全可注入故本地端到端测）
- [ ] T048 [P] [US2] 集成测试：药品/医学问题 → 带引用且不超出来源 于 `tests/integration/test_qa_cited.py`

**Checkpoint**: US1 与 US2 各自独立可用

---

## Phase 6: User Story 3 - 基线+消融的可归因评估（Priority: P3）

**Goal**: 固定测试集上对各模块开关消融，量化增益 + 显著性 + 小病灶分层。

**Independent Test**: 单开关一个改动 → 相对基线 delta + 分层 + bootstrap 显著性。

- [x] T049 [P] [US3] 检测指标 FROC/sensitivity@FP（主）+ mAP（辅），面积分层 于 `src/eval/metrics_detection.py` ✅（贪心 IoU 匹配；分层复用 `ROI.area_band`，未另建 stratify.py）
- [x] T050 [P] [US3] 报告指标 域无关 实体F1+关系F1（复用 T041 NER）于 `src/eval/metrics_report.py` ✅（可注入 `ner_fn`/`rel_fn` 解耦 T041；纯 PRF 本地测绿，缺注入抛清晰错）
- [x] T051 [P] [US3] RAG 指标 ragas(faithfulness/context P-R，固定 judge temp0)+recall@k/nDCG/MRR 于 `src/eval/metrics_rag.py` ✅（检索指标纯逻辑本地测；ragas 裁判走 DashScope/qwen-max temp0，key 从 `DASHSCOPE_API_KEY` env 读、不入库，守卫导入）
- [x] T052 [P] [US3] 端到端指标 证据可溯源率 + 拒答正确性(abstention P/R) 于 `src/eval/metrics_e2e.py` ✅
- [x] T053 [P] [US3] bootstrap CI + 配对显著性检验 于 `src/eval/stats.py` ✅（bootstrap CI / 配对 delta CI / 置换检验 / McNemar，固定 seed）
- [x] T054 [US3] 消融矩阵 runner（一次一变量开关、相对基线 delta、小病灶分层）于 `src/eval/ablation.py`（依赖 T016, T049-T053）✅（`build_variants` 单变量派生；逐样本配对 delta CI + 置换检验；按 AreaBand 分层验"增益集中小病灶"）
- [x] T055 [US3] B 三臂消融报告（相加/拼接/多图，证明增益集中 <2%）于 `src/eval/ablation_b.py`（依赖 T054, T030）✅（**框架完成、桩数据测绿**：强制基线 global-only(fusion=off)，三臂逐样本配对 delta CI+分层；判定每臂"增益是否集中小病灶"+ 区分"整体最强 vs 集中小病灶"；接 T056 gate。⏳ 真数据待 T030 后半 B 接线在 AutoDL 跑出预测）
- [x] T056 [US3] 质量门（增益不显著→阻断进入下一阶段）织入 runner 于 `src/eval/runner.py`（依赖 T053, T054）✅（`QualityGate`：配对 delta CI 方向+置换检验 p+min_delta 三判据；`check_scores`/`assert_pass`；接进 `ablation.run_ablation(gate=)` 给每变体裁决 + `ablation.blocked()`）
- [ ] T057 [P] [US3] 集成测试：开关单改动→分层+显著性 delta 于 `tests/integration/test_ablation.py`

**Checkpoint**: 所有创新点有可归因消融增益

---

## Phase 7: 部署生产化加固（MCP + AutoDL）— FR-013

**Purpose**: 骨架的最小 MCP（T022）已在跑；本阶段加固为生产远程服务

- [ ] T058 medrag 推理服务（GPU 模型常驻 + ChromaDB 连接 + 真实管线复用）于 `src/serve/service.py`（依赖 T036, T047）
- [ ] T059 MCP server 升级：双模 transport（远程 streamable HTTP/SSE + 本地 stdio）+ 鉴权 token/TLS 于 `src/serve/mcp_transport.py`（依赖 T022, T058）
- [ ] T060 [P] 工具输入校验 + PHI 安全输出/日志（FR-007）于 `src/serve/mcp_tools.py`
- [ ] T061 AutoDL 部署：端口映射/隧道、进程守护（健康检查/重启）、权重与向量库落数据盘 于 `scripts/deploy_autodl.sh`（依赖 T059）
- [ ] T062 [P] LangGraph client 生产联通测试（远程 transport + 鉴权 + 每工具）于 `tests/integration/test_mcp_langgraph.py`

**Checkpoint**: 外部 LangGraph 多Agent 项目经远程 MCP 调用 medrag 全部能力，溯源不丢

---

## Phase 8: Polish & Cross-Cutting

- [ ] T063 [P] 数据管线单测（dedup/ner_quality/ct_label/inpaint 抗泄露）于 `tests/unit/test_data.py`
- [ ] T064 [P] 模型单测（损失形状、B 门控 α 初始0 等价预训练分布、三臂 token 数）于 `tests/unit/test_models.py`
- [ ] T065 [P] 检索单测（hybrid RRF、级联、rerank、拒答门）于 `tests/unit/test_rag.py`
- [ ] T066 [P] 编写 quickstart（env→建库→训练→评估→服务）于 `specs/001-medrag-detect-report/quickstart.md`
- [ ] T067 安全加固：贯查无 PHI 入日志/向量库/MCP 传输（FR-007）于 `src/`
- [ ] T068 [P] 可复现验证：固定 seed + 锁版本复现评估记录 于 `tests/integration/test_repro.py`
- [ ] T069 Constitution 门复核：溯源(I)/防泄露·训推一致(II)/基线·一次一变量消融(III)

---

## Dependencies & Execution Order

### Phase Dependencies
- Setup → Foundational（含契约+基线+评估骨架，**阻塞**）→ **Walking Skeleton（桩端到端，先跑通）** → US1/US2/US3（填真肉）→ 部署加固 → Polish
- **关键**：Skeleton(Phase 3) 完成后，US1/US2 的组件做完即可"替桩→对骨架端到端验证"
- 部署加固(Phase 7) 依赖 US1(T036)+US2(T047) 真实管线；但最小 MCP 在 Skeleton 即存在

### User Story Dependencies
- US1/US2/US3 均依赖 Skeleton；之后可并行（US1 视觉路 / US2 文本路 / US3 评估）
- US3 消融需 US1/US2 产物才有可评对象，但 harness 可独立先建（Foundational T015/T016 已起）

### Parallel Opportunities
- Setup：T003-T006 [P]
- Foundational：T009/T010/T012/T014 与 T017/T018 [P]
- US1 起步：T024/T027/T028/T029 [P]
- US2 起步：T038/T041/T043 [P]
- US3：T049-T053 五指标全 [P]

---

## Implementation Strategy

### MVP First
1. Setup → 2. Foundational（接口契约 + 基线 + 评估骨架）→ 3. **Walking Skeleton（桩端到端跑通 + MCP 可调）** → 4. US1（替桩为真实 C+B+报告）
5. **STOP & VALIDATE**：US1 端到端 vs 单路 global-only 基线（小病灶分层）→ 可演示

### Incremental Delivery
骨架活 → US1 真实报告（MVP）→ US2 医学QA → US3 完整消融/显著性 → 部署加固接 LangGraph。每步都对**已跑通的骨架**增量替换，集成风险前置。

### 关键约束
- 每 milestone 跑固定测试集 vs 上一基线；**增益不显著不进下一阶段**
- 一次只改一个变量、逐项消融
- AutoDL 预算充足、以"保证完成任务"为准、对卡数自适应；ZeRO-2 备而默认关

---

## Notes
- [P] = 不同文件、无未完成依赖
- 待实测（非阻塞）：c 图像编码器（BiomedCLIP/PMC-CLIP vs 通用 VL）、多模态 VL 嵌入型号 → 影响 T031/T032
- 未经用户同意不开始写代码/训练
