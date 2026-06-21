<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

---

# 📌 项目状态看板（冷启动先读这里）

> 给新会话：本节是项目的"你在哪"。读完即可继续工作。事实源文件见末尾"指针"。
> **最后更新：2026-06-21（数据/部署 5 项决策已敲定）**

## 这是什么项目
求职面试项目：基于 **RAG + Qwen3-VL** 的"**病灶检测 + 报告生成**"医学多模态垂直系统，
针对通用 VLM 的**微小病灶特征湮灭**与**幻觉**两大痛点。开发用 **Spec-Kit** 框架驱动。

## 整体进度（阶段）
- [x] **环境搭建**：conda env `F:\miniconda\envs\medrag`（Py3.11，已装 git+uv）；Spec-Kit v0.11.3 已 init（claude/ps）；git 仓库已建、首提交完成。
- [x] **Spec-Kit 框架落档（部分）**：constitution v1.0.0 已写；feature `001-medrag-detect-report` 的 spec.md（WHAT/WHY）+ plan.md（HOW，仅 B/C）已写。
- [x] **核心创新 B、C 设计敲定**（详见 plan.md）。
- [x] **数据集分工细化**（a/b/c）— 已细化并入 plan.md：a/b 去重(精确+MinHash,QA冲突 LLM 判别)、NER 覆盖率+多信号质量分(中英分治)、c CT coreset(面积分层+K-means+三档选样)。
- [x] **RAG 链路实现细节** — 已细化并入 plan.md：父子层级 AutoMerging 分块、embedding(a/b Qwen3-Embedding 4B + c 多向量 全图/ROI/报告文本)、级联(BM25+dense hybrid→RRF→文本筛50→图像→二次RRF→Top-10~20)、Qwen3-VL-Reranker(cross-encoder) Top-5 + 拒答门。
- [x] **部署与对外服务（AutoDL + MCP）** — 已细化并入 plan.md：AutoDL 托管 GPU+MCP server；medrag 作 MCP server（FastMCP，远程 HTTP/SSE+本地 stdio 双模），暴露 generate_report/detect_lesions/retrieve_evidence/medical_qa 给**另一个多Agent 医疗项目（LangGraph client）**；工具输出带溯源。仅余卡型/预算待定。
- [x] **评估 harness（设计）** — 已细化并入 plan.md：配置驱动消融 runner；检测 FROC/sensitivity@FP+mAP、报告 域无关实体F1+关系F1、RAG ragas+recall@k/nDCG/MRR、端到端 溯源率+拒答正确性、bootstrap CI+配对检验、小病灶分层贯穿。（实现待 coding 阶段）
- [ ] **写代码 / 训练** — 尚未开工（用户明确要求先把计划落框架，暂不 coding）。

**当前位置**：设计与规格阶段。B/C 已定并入 plan；其余模块在 plan.md 里是 TODO，等逐块讨论后增量补。

## 已敲定的核心决策（不要推翻，除非用户改主意）
- **B 双路融合**：解决病灶特征被背景稀释。全局图 + ROI 放缩图在 merger 后的视觉 token 层融合。
  做三臂消融：`相加(门控残差 V_g+tanh(α)·g(V_roi)，α 初始0)` vs `拼接` vs `多图输入(最强基线)`。
  评估按病灶面积分层，证明增益集中在小病灶(<2%)。需改 `modeling_qwen3_vl.py`。
- **C 高斯退火定位**：绿框**只当标签不进输入**——cv2 提框坐标→生成宽高斯热图标签→inpaint 抹掉
  绿框喂干净图→ViT 上辅助定位头预测热图（即"检测器"，推理无框可用）。损失用 CenterNet
  penalty-reduced focal + Dice；退火 = target 高斯 σ(t) 由大到小（粗→细）。二阶防泄露：随机区域同样 inpaint。
- **方法论铁律**（constitution 不可协商项）：① 证据可溯源、不编造；② 防标签泄露/训推一致；
  ③ 固定基线(vanilla 零样本 + 朴素 LoRA) + 一次只改一个变量 + 逐项消融。
- **技术栈**：训练走 LLaMA-Factory 或 ms-swift（不手写循环），基座 Qwen3-VL 30B-A3B (MoE)；
  RAG 用 LlamaIndex(父子层级 AutoMerging) + Qwen3-Embedding 4B(文本a/b)/多模态VL嵌入(c) + Qwen3-VL-Reranker(Top-5,cross-encoder) + ChromaDB(按数据集分 collection)；
  评估 ragas + 临床指标(实体F1/RadGraph) + 检测 mAP + 小病灶分层。
  DeepSpeed 配置常备、默认关，多卡时切 ZeRO-2（不上 ZeRO-3）。

## ✅ 已解决的开放问题（2026-06-21 敲定）
1. **绿框样式** → 细边框（病灶可见、inpaint 可行，C 成立）。
2. **数据集分工** → a 药品 / b 医学QA / c CT-QA 均入 ChromaDB 按类型分 collection；c CT-QA 同时进视觉训练。
3. **基座 + DeepSpeed** → 基座 Qwen3-VL 30B-A3B (MoE)（8B 太小）；DeepSpeed ZeRO-2 配好、可关。
4. **推理输入** → 恒为处理后无框图像（C 是真贡献，非退化为 cv2 取框）。

## ✅ 资源决策（2026-06-21 敲定）
- **AutoDL 卡型/预算**：预算充足，以"**保证完成任务**"为准——按可用选高显存卡（A100/H100 级），必要时多卡上 ZeRO-2；单卡 LoRA 为底线；**卡型不预锁、配置对卡数自适应**。
- 仅余"待实测"项（非阻塞）：c 图像编码器（BiomedCLIP vs 通用 VL）、多模态 VL 嵌入确切型号。

## 🔧 本机环境约定（执行命令前必看，否则会报错）
- **不要用 `conda run`** 跑有 Unicode 输出的命令（GBK 编码会崩）。改为：把 medrag 环境加进 PATH + 切 UTF-8：
  ```powershell
  chcp 65001 > $null
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  $env:PYTHONUTF8 = "1"
  $r = "F:\miniconda\envs\medrag"
  $env:PATH = "$r;$r\Library\mingw-w64\bin;$r\Library\usr\bin;$r\Library\bin;$r\Scripts;" + $env:PATH
  ```
- git 在 F 盘需 `safe.directory`（已全局加过）。spec-kit 的 pwsh 脚本本机用 `powershell`（无 pwsh.exe）跑。
- GitHub 访问不稳（有代理），克隆/装包失败先重试再判断。
- 激活环境：`conda activate F:\miniconda\envs\medrag`。

## 📂 事实源指针（详情看这些文件，别凭本看板记忆下结论）
- 设计/规格：[constitution](.specify/memory/constitution.md)、[spec.md](specs/001-medrag-detect-report/spec.md)、[plan.md](specs/001-medrag-detect-report/plan.md)、[tasks.md](specs/001-medrag-detect-report/tasks.md)
- 完整技术评审与路线：`C:\Users\virgi\.claude\plans\rag-qwen3-vl-deepspeed-lora-cuddly-tome.md`
- 记忆：项目背景 `project-medrag`、环境坑 `env-tooling-gotchas`（在 memory 目录）

## ▶️ 下一步（新会话可直接接手）
**设计阶段全部模块已细化并入 plan.md**（数据集分工 / RAG 链路 / 评估 harness / 部署与对外服务 AutoDL+MCP）；
AutoDL 卡型/预算已收口（预算充足、保证完成任务、对卡数自适应）；**`tasks.md` 已生成**（66 任务，US1/US2/US3 + 部署）。
仅余"待实测"非阻塞项：c 图像编码器选型、多模态 VL 嵌入型号。
**下一步：经用户同意后进入实现**——MVP = Setup + Foundational + US1（端到端检测+报告+溯源）。
当前仍在"设计落框架"阶段，**未经用户同意不要开始写代码/训练**。
