<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

---

# 📌 项目状态看板（冷启动先读这里）

> 给新会话：本节是项目的"你在哪"。读完即可继续工作。事实源文件见末尾"指针"。
> **最后更新：2026-06-24（US3 评估 harness 全套落地并本地测绿：T016 runner/record + T049–T054 指标/消融；归因底座闭合。US1 创新 C/报告/RAG 库已就位。⚠️ 本地领先 origin/main 4 个提交待 push）**

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
- [~] **US1 实现批（2026-06-23，本地 main 已提交，AutoDL 已验证 torch 部分）**：
  - **创新 C 定位链** ✅：T024 `data/ct_box.py`(cv2.inRange 绿框→ROI) / T025 `data/ct_label.py`(ROI→宽高斯热图+σ退火) / T026 `data/ct_inpaint.py`(inpaint 抹框+二阶随机诱饵防泄露) / T027 `data/coreset.py`(面积分层+KMeans+√配额+三档选样) / T028 `models/loc_head.py`(ViT 定位头+热图→DetectionResult 解码) / T029 `models/losses.py`(CenterNet penalty-reduced focal+Dice+`make_annealed_target` 接 T025 退火)。
  - **创新 B 融合**：T030 `models/fusion.py` ✅ 模块（三臂 add/concat/multi_image；门控 α=0 初始恒等已测）。⏳ **半成品**：接进基座前向（vendor+patch `modeling_qwen3_vl.py`，T012）**未做**，留 AutoDL 对着 transformers 5.12.1 交互式做。
  - **报告生成** T034 `models/report.py` ✅：引用标签 `[S*]/[ROI*]` 强制每条结论锚证据/ROI，无据标 uncertain 或拒答（禁编造）；VLM 草稿可注入 `draft_fn` 测试。
  - **RAG 地基** T013 `rag/store.py` ✅：ChromaDB 5 collection(`a_drug/b_medqa/c_text/c_img_whole/c_img_roi`)、cosine、where 过滤、内存/持久双模。
- [x] **US3 评估 harness 实现批（2026-06-24，本地 main 已提交、纯逻辑全测绿）——归因底座闭合**：
  - **指标五件套**（纯 numpy，本地全测）：T049 `eval/metrics_detection.py`(贪心 IoU 匹配→FROC/sensitivity@FP+VOC mAP，按 `ROI.area_band` 分层，**未另建 stratify.py**) / T050 `eval/metrics_report.py`(域无关 micro 实体F1+关系F1，**可注入 `ner_fn`/`rel_fn` 解耦未建的 T041**) / T051 `eval/metrics_rag.py`(检索 recall@k/nDCG/MRR 纯逻辑 + ragas faithfulness/context **裁判走 DashScope qwen-max temp0，key 仅从 `DASHSCOPE_API_KEY` env 读、守卫导入**) / T052 `eval/metrics_e2e.py`(证据可溯源率+拒答 P/R) / T053 `eval/stats.py`(bootstrap CI/配对 delta CI/置换检验/McNemar，固定 seed)。
  - **runner/消融**：T016 `eval/runner.py`(配置驱动 `run_eval(predict_fn,metric_fn)`→EvalRecord，**与模型解耦故本地可测**)+`eval/record.py`(落盘/读取/拍平) / T054 `eval/ablation.py`(`build_variants` **强制一次只改一个 flag**；逐样本配对 delta CI+置换检验；按 AreaBand 分层验"增益集中 <2% 小病灶")。
  - **意义**：尺子已就位——任何创新一旦能跑出预测，就能立刻量化"增益+显著性+小病灶分层"。
  - **测试**：本地全量 **94 passed / 12 skipped**（skip 详见下）。提交 `0e84a63`(T049/50) `4cf7e51`(docs) `4e34483`(T051/52/53) `2519852`(T016/T054)，**4 个待 push**。

**当前位置**：US3 评估 harness 全套就位、归因底座闭合；US1 创新 C 定位链+报告+RAG 库就位。**B 融合接线(T012+T030 后半)是 US1 MVP 最后硬骨头，待 AutoDL。下一步见末尾"▶️ 下一步"。**

## ⚠️ 已写代码但本地跑不全 → 必须 AutoDL 收尾/验证（冷启动重点看这里）
> 这些**逻辑已落、本地能 import**，但功能验证缺本地依赖（对应 9 个 torch-skipped 测试）。**别误判为"没做"，也别误判为"全验过"。**
- **T028 `models/loc_head.py` / T029 `models/losses.py` / T030 `models/fusion.py`**：纯逻辑测过；前向/损失/门控 α=0 恒等测 **skip(本地无 torch)** → AutoDL 装 torch 真跑过那 9 个 skip。
- **T012 + T030 后半（B 融合接线）= 还没写**：需 AutoDL vendor transformers 的 `modeling_qwen3_vl.py`，merger 输出处插 `DualPathFusion`（跑两遍视觉塔：全局+ROI）。**版本强相关，必须对着 AutoDL 实装的 transformers 5.12.1 交互式做。**
- **T011 `models/qwen3vl.py`**：4B 已 `L1 PASS`；30B-A3B 需 A100-80G 级 + 扩盘。
- **T013/T024–T027**：本地依赖(cv2/sklearn/chromadb)已装、测过；真实 CT 图/真实建库的功能性跑仍在 AutoDL。
- **T051 ragas 裁判段**：检索指标本地测过；ragas 真打分需 `pip install ragas langchain-openai` + `DASHSCOPE_API_KEY`（联网，本地或 AutoDL 皆可）。

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
- git 在 F 盘需 `safe.directory`（已全局加过）；**git 在 medrag 环境里**（`F:\miniconda\envs\medrag\Library\bin\git.exe`），上面那段 PATH 设置已含。spec-kit 的 pwsh 脚本本机用 `powershell`（无 pwsh.exe）跑。
- 激活环境：`conda activate F:\miniconda\envs\medrag`。
- **本机装重依赖的坑**：默认/清华镜像源会 `SSL UNEXPECTED_EOF`（本机 Clash TUN 全局拦截，`unset *_proxy` 也绕不开）。**正解**：保留代理走官方源 `pip install <pkg> -i https://pypi.org/simple`（环境变量带 `http(s)_proxy=http://127.0.0.1:7890`）。已这样装好本机：`opencv-python-headless`、`scikit-learn`、`chromadb`（仅本机验证用；AutoDL 靠 `autodl_setup.sh`）。
- **写代码方法论（沿用）**：模块用**守卫导入**（torch/cv2/sklearn/chromadb 缺了也能 `import`，调用才报清晰错）；**纯逻辑/numpy 部分本地单测跑绿**，重依赖功能测用 `pytest.importorskip` + 在 AutoDL 真跑。本机跑测：上面 PATH 段 + `python -m pytest -q`。
- **GitHub 同步已打通（2026-06-23）**：本地 `main` 为权威源。本地→远程：用户在**自己终端**跑 `scripts\push_github.ps1 -Proxy http://127.0.0.1:7890`（`--force-with-lease` 推 `main`，首次弹窗登录）。本助手沙箱推不了（无 TTY），所以**改完代码先 commit，由用户 push**。
- **AutoDL（云 GPU 端，已在用）**：conda env 也叫 `medrag`（Py3.11）；**干净仓在 `/root/medrag-clean`（git clone，已弃用手拷的乱仓 `/root/medrag-1`）**；同步靠 `cd /root/medrag-clean && git pull`。数据盘 `/root/autodl-tmp`(50G)；`HF_HOME=/root/autodl-tmp/.cache/huggingface`、`HF_ENDPOINT=https://hf-mirror.com`。学术加速**按渠道**开关：清华源/hf-mirror/ModelScope 直连(关)，github/hf官网/kaggle 才 `turbo_on`(=`source /etc/network_turbo`)。装依赖：`bash scripts/autodl_setup.sh`（已含 torch 校验 + opencv/sklearn/pytest/chromadb）。

## 📂 事实源指针（详情看这些文件，别凭本看板记忆下结论）
- 设计/规格：[constitution](.specify/memory/constitution.md)、[spec.md](specs/001-medrag-detect-report/spec.md)、[plan.md](specs/001-medrag-detect-report/plan.md)、[tasks.md](specs/001-medrag-detect-report/tasks.md)
- 完整技术评审与路线：`C:\Users\virgi\.claude\plans\rag-qwen3-vl-deepspeed-lora-cuddly-tome.md`
- 记忆：项目背景 `project-medrag`、环境坑 `env-tooling-gotchas`（在 memory 目录）

## ▶️ 下一步（新会话可直接接手）

**里程碑已达**：① `L1 PASS`（环境+模型冒烟，4B）；② US1 创新 C 定位链(T024–T029)+报告(T034)+RAG 库(T013) 落地、本地测绿；③ **US3 评估 harness 全套(T016+T049–T054)落地、归因底座闭合**。任务勾选见 [tasks.md](specs/001-medrag-detect-report/tasks.md)（注：T001–T023 基建/骨架实际已完成，旧勾选框可能未同步）。

**① 先做（纯逻辑、本地可全测、依赖已就绪）—— US3 收尾 T056 + T055**：
- **T056 质量门**：把"配对检验不显著 → 阻断进下一阶段"织进 `eval/runner.py`（复用 T053/T054）。让 harness 自己会拦不靠谱改动，constitution III 彻底闭环。
- **T055 B 三臂消融报告** `eval/ablation_b.py`：add/concat/multiimg 专项对比、出"增益集中 <2%"报告；B 真数据没出来前先用桩数据跑通框架。

**② RAG 文本链 US2（结构本地测/功能 AutoDL 验）**：T014 文本嵌入(Qwen3-Embedding 4B) → T043 父子分块 → T044 入库(用 T013 store) → T045 hybrid 检索 → T046 reranker → T047 medical_qa 接回。守卫导入、纯逻辑本地测。

**③ 收口与训练（需 AutoDL/GPU，见上"⚠️ 必须 AutoDL"节）**：
- **T012 + T030 后半**：vendor transformers 的 `modeling_qwen3_vl.py`，merger 输出处插 `DualPathFusion`（跑两遍视觉塔：全局+ROI）。**对着 AutoDL 的 transformers 5.12.1 交互式做。** 同时过掉 T028/T029/T030 的 9 个 torch-skipped 测。
- **T031/T032/T033**：C 图像嵌入(全图+ROI 双向量) → c 多向量入库 → 视觉级联检索。
- **T035**：C+B 训练脚本（LoRA + σ 退火课程 + 各创新消融开关），走 LLaMA-Factory/ms-swift。
- **T036**：把真实 detect/visual-retrieve/report 接回 `serve/pipeline.py`（改 `flags` 分支，签名不变），对骨架端到端验证。

**④ 数据/资源**：真实跑训练前需 `bash scripts/download_assets.sh data`（c=MedTrinity-25M 25M_demo 落 `/root/autodl-tmp/raw/c_ct`，**gated 需 `export HF_TOKEN=<read token>`**）。4090/24GB 只够 4B 验证代码路径；真跑 30B 需 A100-80GB 级 + 扩盘(≥120GB)，用户已确认后续升级。

**待办（非阻塞）**：本地领先 origin/main **4 个提交待用户 push**（`0e84a63` T049/50 / `4cf7e51` docs / `4e34483` T051/52/53 / `2519852` T016/T054）；AutoDL `git pull` 后确保 `pip install scikit-learn chromadb`（旧 setup 没装全），ragas 真跑再 `pip install ragas langchain-openai`，然后 `python -m pytest tests/unit/ -q` 复验。
