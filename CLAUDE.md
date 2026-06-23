<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

---

# 📌 项目状态看板（冷启动先读这里）

> 给新会话：本节是项目的"你在哪"。读完即可继续工作。事实源文件见末尾"指针"。
> **最后更新：2026-06-23（里程碑达成：L1 PASS，环境+模型冒烟跑通；下一步进 US1 MVP）**

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
- [~] **写代码 / 训练（实现已开工，3 批已落地，本地 git 已提交）**：
  - **B1 walking skeleton**（commit `841766e`）：`src/` 8 模块 + `contracts/schemas.py`（共享数据契约，含"禁编造"校验）+ `config/`(config/seed/run_record) + `serve/`(stubs→pipeline→FastMCP server 暴露 4 工具)；12 冒烟测试绿；pipeline 按 `AppConfig.flags` 选实现（现全 `stub`），真实组件逐个替桩。
  - **B2 AutoDL bring-up + T011**（commits `897ae04`/`5a5f4d4`/`968f0fb`/`844268e`）：`models/qwen3vl.py` 基座加载器（`AutoModelForImageTextToText` 自动派发 dense 4B/8B + MoE 30B）；`scripts/`(smoke_gpu L0/L1 分级、autodl_setup、download_assets、push_github)、`autodl_runbook.md`、`test_models.py`；修了 `.gitignore` 把 `src/models`/`src/data` 误忽略的坑（已锚定为 `/models/`）。
  - **AutoDL 实跑通过**：torch 2.12.1+cu130 / CUDA13 / **RTX 4090 D 24GB** / 数据盘 50GB / 裸镜像(清华源装 torch)。**L0 PASS**。
- [x] **L1 PASS（2026-06-23 达成「环境+模型冒烟」里程碑）**：加载器把本地 4B 自动派发为 dense `Qwen3VLForConditionalGeneration`，推理语义正确（准确描述灰阶渐变），峰值显存 8.9GB。
  - **HF 联网崩（httpx `client closed`）的解法 = 全本地离线**：ModelScope 拉全 4B 到 `/root/autodl-tmp/weights/Qwen3-VL-4B-Instruct`，再 `unset *_proxy; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MEDRAG_BASE_MODEL=<该目录>; python scripts/smoke_gpu.py --with-model`。**教训：AutoDL 冒烟优先走"本地目录+离线"，别靠在线 HF 下载。**（崩因：`smoke_gpu.py` 用 `setdefault` 设 `HF_ENDPOINT` 没覆盖到→打到真 huggingface.co 触发 Errno99→hf_hub 的 httpx 重试复用了已关闭的 client。）

**当前位置**：实现阶段·"环境+模型冒烟"批**已完成**。下一步进 **US1 MVP**。详见末尾"▶️ 下一步"。

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
- **AutoDL（云 GPU 端，已在用）**：conda env 也叫 `medrag`（Py3.11，miniconda3）；项目在 `/root/medrag-1/rag_workspace_claude`（**用户手拷，非 git clone**）；数据盘 `/root/autodl-tmp`(50G)；`HF_HOME=/root/autodl-tmp/.cache/huggingface`、`HF_ENDPOINT=https://hf-mirror.com`。学术加速**按渠道**开关：清华源/hf-mirror/ModelScope 直连(关)，github/hf官网/kaggle 才 `turbo_on`(=`source /etc/network_turbo`)。本地↔AutoDL 目前靠**手拷**同步（GitHub 未推通），改了脚本要重拷。

## 📂 事实源指针（详情看这些文件，别凭本看板记忆下结论）
- 设计/规格：[constitution](.specify/memory/constitution.md)、[spec.md](specs/001-medrag-detect-report/spec.md)、[plan.md](specs/001-medrag-detect-report/plan.md)、[tasks.md](specs/001-medrag-detect-report/tasks.md)
- 完整技术评审与路线：`C:\Users\virgi\.claude\plans\rag-qwen3-vl-deepspeed-lora-cuddly-tome.md`
- 记忆：项目背景 `project-medrag`、环境坑 `env-tooling-gotchas`（在 memory 目录）

## ▶️ 下一步（新会话可直接接手）

**① ✅ 已完成：`L1 PASS`（环境+模型冒烟里程碑达成）。** 复现方式（AutoDL bash，`conda activate medrag`，`cd /root/medrag-1/rag_workspace_claude`）——**走本地离线，别在线下**：
```bash
MEDRAG_BASE_MODEL_MS=Qwen/Qwen3-VL-4B-Instruct bash scripts/download_assets.sh weights   # 一次性，ModelScope 拉全 4B
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MEDRAG_BASE_MODEL=/root/autodl-tmp/weights/Qwen3-VL-4B-Instruct
python scripts/smoke_gpu.py --with-model   # → L1 PASS，峰值显存 8.9GB
```

**② 马上要做：US1 MVP**（真实 C 定位 + B 融合 + CT 检索 + 报告，替换骨架桩）。
✅ **前置验证已完成（2026-06-23，`scripts/probe_dataset_c.py`）**：c(`MedTrinity-25M` 25M_demo) 图上**确实画了绿色规整矩形 ROI 框**（每图 ~0.5–0.75% 绿像素，已肉眼确认），caption 是多粒度文本、位置只用文字描述、**无数字坐标**——所以**绿框就是唯一定位信号**，C 的 `cv2.inRange 提框` 原设计成立、T024 不改。caption 同时是报告生成的现成监督。
开工：实现 T024 `src/data/ct_box.py`（cv2.inRange 绿阈值按实图调）→ T025 高斯热图标签 → T026 inpaint 抹框+防泄露。

**③ 资源**：4090/24GB/50GB **只够 4B 验证代码路径**；真跑 30B 基座需升级显卡(A100 80GB 级)+扩容数据盘(≥120GB)，
用户已确认后续可升级。届时同样命令把 `--model-id` 换 `Qwen/Qwen3-VL-30B-A3B-Instruct`（显存紧加 `--quant 4bit`）。

**两个待办（非阻塞）：**
- **GitHub 未推通**：`origin` 已配(`940573605altoria-png/medrag`)，但本助手沙箱无终端弹不出登录窗 + 国内网络需走 7890 代理。
  待用户在**自己终端**跑 `scripts\push_github.ps1 -Proxy http://127.0.0.1:7890`（弹窗登录一次）。推通后 AutoDL 可改 `git pull` 同步。
- **本地有未提交脚本改动**：`scripts/autodl_setup.sh`(清华源+学术加速按渠道 turbo)、`scripts/push_github.ps1`(ASCII+代理)、`scripts/smoke_gpu.py`(HF 超时默认)。需提交。
