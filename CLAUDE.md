<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

---

# 📌 项目状态看板（冷启动先读这里）

> 给新会话：本节是项目的"你在哪"。读完即可继续工作。事实源文件见末尾"指针"。
> **最后更新：2026-06-21**

## 这是什么项目
求职面试项目：基于 **RAG + Qwen3-VL** 的"**病灶检测 + 报告生成**"医学多模态垂直系统，
针对通用 VLM 的**微小病灶特征湮灭**与**幻觉**两大痛点。开发用 **Spec-Kit** 框架驱动。

## 整体进度（阶段）
- [x] **环境搭建**：conda env `F:\miniconda\envs\medrag`（Py3.11，已装 git+uv）；Spec-Kit v0.11.3 已 init（claude/ps）；git 仓库已建、首提交完成。
- [x] **Spec-Kit 框架落档（部分）**：constitution v1.0.0 已写；feature `001-medrag-detect-report` 的 spec.md（WHAT/WHY）+ plan.md（HOW，仅 B/C）已写。
- [x] **核心创新 B、C 设计敲定**（详见 plan.md）。
- [ ] **数据集分工细化**（a/b/c）— 待讨论。
- [ ] **RAG 链路实现细节** — 待讨论。
- [ ] **AutoDL 云环境适配** — 待讨论。
- [ ] **评估 harness** — 待实现。
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
- **技术栈**：训练走 LLaMA-Factory 或 ms-swift（不手写循环），基座 Qwen3-VL 4B 起步；
  RAG 用 LlamaIndex(父子层级) + Qwen3-VL-Embedding + Qwen3-VL-Reranker(Top-5) + Qdrant/Milvus；
  评估 ragas + 临床指标(实体F1/RadGraph) + 检测 mAP + 小病灶分层。
  DeepSpeed 非默认，仅多卡/大模型时切 ZeRO。

## ⚠️ 待用户确认的开放问题（卡住部分推进）
1. **绿框样式**：细边框 还是 填充块？（填充块盖住病灶则 inpainting 不可行，需换数据策略）
2. **数据集分工**：a 药品 / b 医学QA / c CT-QA —— 各自进 RAG 库还是进训练？是否补充其它数据集。
3. **AutoDL 配置**：卡型 / 数量 / 预算 → 决定基座 4B 还是 8B、要不要多卡+DeepSpeed。
4. 推理时输入是否始终带框（决定 C 是真贡献还是退化为 cv2 取框）。

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
- 设计/规格：[constitution](.specify/memory/constitution.md)、[spec.md](specs/001-medrag-detect-report/spec.md)、[plan.md](specs/001-medrag-detect-report/plan.md)
- 完整技术评审与路线：`C:\Users\virgi\.claude\plans\rag-qwen3-vl-deepspeed-lora-cuddly-tome.md`
- 记忆：项目背景 `project-medrag`、环境坑 `env-tooling-gotchas`（在 memory 目录）

## ▶️ 下一步（新会话可直接接手）
继续逐块讨论并增量补进 plan.md：**数据集分工 → RAG 链路 → AutoDL 适配 → 评估 harness**。
当前仍在"设计落框架"阶段，**未经用户同意不要开始写代码/训练**。
