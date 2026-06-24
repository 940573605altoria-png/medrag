# Quickstart（T066）—— env → 建库 → 训练 → 评估 → 服务

> 约定：🟢 = 本地纯逻辑可跑（无需 GPU）；🟡 = 需重依赖/模型/GPU，**在 AutoDL 跑**。
> 冷启动先读项目根 `CLAUDE.md` 的"项目状态看板"。

## 0. 环境

**本地（Windows，仅验证代码路径）** 🟢
```powershell
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$r = "F:\miniconda\envs\medrag"
$env:PATH = "$r;$r\Library\mingw-w64\bin;$r\Library\usr\bin;$r\Library\bin;$r\Scripts;" + $env:PATH
python -m pytest -q          # 全量纯逻辑/numpy 测试（重依赖测自动 skip）
```

**AutoDL（云 GPU，真实功能）** 🟡
```bash
cd /root/medrag-clean && git pull
bash scripts/autodl_setup.sh           # torch 校验 + opencv/sklearn/pytest/chromadb
pip install scikit-learn chromadb sentence-transformers rank-bm25 datasketch \
            llama-index-core ragas langchain-openai   # 按需装重依赖
python -m pytest tests/unit -q         # 含 torch/模型功能测
```

## 1. 建知识库（a 药品 / b 医学QA → ChromaDB）

**数据 ingestion 清洗管线** 🟢（结构本地测）/ 🟡（真实 NER 需模型）
- a/通用：`src.data.ingest.ingest_documents(nodes, config, ner=...)`
- b 医学QA：`src.data.ingest.ingest_qa(items, config, judge_fn, merge_fn, ner=...)`
- 链路：归一化(T038)→去标识(T009,FR-007)→去重(T039)→(b)QA冲突(T040)→NER 质量筛(T041/T042)；
  全程覆盖护栏(FR-012)。NER/judge 后端 🟡（scispaCy / CMeEE BERT / DashScope）。

**分块 + 嵌入 + 入库** 🟢（结构/内存库本地测）/ 🟡（真实嵌入模型）
- 分块：`src.rag.chunk.chunk_qa` / `hierarchical_chunk_documents`(🟡 llama-index)
- 嵌入：`src.rag.embed_text.TextEmbedder`（🟡 Qwen3-Embedding 4B；本地用注入 `encode_fn` 桩）
- 入库：`src.rag.index_text.index_qa/index_chunks`（叶块→chroma，全节点→DocStore 承 AutoMerging）
- 真实建库（AutoDL）：`bash scripts/download_assets.sh data`（c 数据集 gated → `export HF_TOKEN`）

## 2. 训练（C 定位 + B 融合 + LoRA）🟡 AutoDL

- 基座 `Qwen3-VL`（4B 验证代码路径已 `L1 PASS`；30B-A3B 需 A100-80G 级 + 扩盘）。
- **B 融合接线（T012 + T030 后半，未完成）**：vendor `transformers` 的 `modeling_qwen3_vl.py`，
  在 merger 输出处插 `DualPathFusion`（跑两遍视觉塔：全局+ROI）。**对着 AutoDL 实装的 transformers
  5.12.1 交互式做。**
- 训练脚本 T035（LoRA + σ 退火课程 + 各创新消融开关），走 LLaMA-Factory/ms-swift；多卡 ZeRO-2（备而默认关）。

## 3. 评估（归因 harness，constitution III）🟢 本地可跑（桩数据）

```python
from src.eval.ablation import run_ablation, blocked
from src.eval.runner import QualityGate, GateConfig
# 一次一变量消融 + 分层 + 配对显著性 + 质量门
cells = run_ablation(base_config, {"fusion": ["add","concat","multiimg"]},
                     score_fn, bands=area_bands, gate=QualityGate(GateConfig(min_delta=0.0)))
assert blocked(cells) == []        # 不显著的改动会被门拦
```
- 指标：检测 FROC/sens@FP+mAP(T049)、报告实体/关系F1(T050)、RAG recall/nDCG/MRR+ragas(T051)、
  端到端溯源率+拒答(T052)、bootstrap/配对检验(T053)。
- B 三臂报告：`src.eval.ablation_b.b_ablation_report(...)`（证明增益集中 <2% 小病灶）。
- ragas 裁判 🟡：`DASHSCOPE_API_KEY` + `pip install ragas langchain-openai`（DashScope qwen-max temp0）。

## 4. 服务（MCP，对 LangGraph）

**本地 stdio**（桩或注入真实组件）🟢
```python
from src.serve.pipeline import Pipeline
from src.serve.qa import MedicalQA
pipe = Pipeline(qa_service=MedicalQA(retriever, reranker, draft_fn=...))  # 注入真实 QA
pipe.answer("阿司匹林的作用?")        # 带引用回答；无据/低置信拒答
```
- MCP server：`python -m src.serve.mcp_server`（暴露 detect_lesions/retrieve_evidence/medical_qa/generate_report）。
- 生产远程 transport + 鉴权（T058–T062）🟡 AutoDL。

## 5. 同步与提交

- 改完先 commit；**推送在你自己终端**：
  `cd f:\vscode\rag_workspace_claude; powershell -ExecutionPolicy Bypass -File scripts\push_github.ps1 -Proxy http://127.0.0.1:7890`
- AutoDL：`cd /root/medrag-clean && git pull`。
