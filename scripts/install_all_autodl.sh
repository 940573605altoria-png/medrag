#!/usr/bin/env bash
# ============================================================================
# medrag 一键依赖安装（AutoDL）—— 装齐截至目前所有模块的守卫依赖。
#
# 适用：AutoDL 实例、已 `conda activate medrag`、已 `cd /root/medrag-clean`。
# 跑法：bash scripts/install_all_autodl.sh
#
# 设计：
#   - **不用 set -e**：可选/易碎包（scispaCy 模型、ms-swift、deepspeed 等）失败不该
#     中断整体——这些模块都是守卫导入，缺了不影响其余功能；脚本末尾统一体检。
#   - 镜像/清华源直连（turbo OFF）；仅 s3/github 渠道（scispaCy 模型）临时 turbo ON。
#   - 幂等：重复跑安全；已装的 pip 会跳过。
#
# 模型**权重**不在此脚本（太大、另走）：见末尾提示 + scripts/download_assets.sh。
# 可调：PIP_INDEX / HF_ENDPOINT / DATA_ROOT / SKIP_TORCH=1（镜像已带 torch 时跳过）
# ============================================================================
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

turbo_on()  { [[ -f /etc/network_turbo ]] && { set +u; source /etc/network_turbo; set -u; echo "  [turbo] ON"; } || echo "  [turbo] 无 network_turbo，跳过"; }
turbo_off() { unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true; echo "  [turbo] OFF（镜像直连）"; }

pipi()  { echo "  pip: $*"; pip install -i "$PIP_INDEX" "$@" || echo "  [WARN] 安装失败（守卫依赖，可后补）: $*"; }
preqi() { echo "  pip(必需): $*"; pip install -i "$PIP_INDEX" "$@" || { echo "  [ERROR] 必需依赖装失败: $*"; FAILED_REQ=1; }; }
FAILED_REQ=0

echo "==> [0/11] 网络 + 数据盘"
turbo_off
for d in raw weights chroma runs; do mkdir -p "$DATA_ROOT/$d"; done
export HF_HOME="${HF_HOME:-$DATA_ROOT/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$DATA_ROOT/.cache/modelscope}"
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE"
python -m pip install -U pip -i "$PIP_INDEX" >/dev/null 2>&1 || true
echo "  PIP_INDEX=$PIP_INDEX  HF_ENDPOINT=$HF_ENDPOINT  DATA_ROOT=$DATA_ROOT"

echo "==> [1/11] torch（校验/安装；AutoDL 镜像通常自带）"
if [[ "${SKIP_TORCH:-0}" != "1" ]] && ! python -c "import torch" 2>/dev/null; then
  pipi torch torchvision
  echo "  若 torch.cuda 不可用：turbo_on 后 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
fi

echo "==> [2/11] 核心运行（契约 + MCP 服务 + 配置）"
preqi -e . --no-deps        # medrag 包本体
preqi pydantic "mcp[cli]" numpy pyyaml   # pyyaml: config.py 读 yaml 配置

echo "==> [3/11] 模型/基座（Qwen3-VL 加载 + 训练栈）"
pipi -U transformers        # Qwen3-VL 需较新版（B 接线对照实装版本 vendor modeling_qwen3_vl.py）
pipi accelerate peft modelscope pillow
pipi bitsandbytes           # 4bit 量化（qwen3vl BitsAndBytesConfig；显存紧时用，易碎）
pipi deepspeed              # 多卡 ZeRO-2（易碎，缺了单卡也能跑）

echo "==> [4/11] RAG 检索栈（文本 + 视觉）"
pipi chromadb               # T013 向量库
pipi llama-index-core       # T043 父子分块 HierarchicalNodeParser
pipi rank-bm25              # T045 BM25 通道
pipi sentence-transformers  # T014 文本嵌入 / T046 CrossEncoder 精排
pipi open_clip_torch        # T031 图像嵌入（BiomedCLIP 经 open_clip 加载）

echo "==> [5/11] 数据预处理"
pipi opencv-python-headless # T024/T026 绿框提取 + inpaint（cv2）
pipi scikit-learn           # T027 coreset KMeans / silhouette
pipi datasketch             # T039 MinHash/LSH 近重复
pipi datasets kagglehub     # 数据下载（b HF / a kaggle）

echo "==> [6/11] 评估 harness"
pipi ragas langchain-openai # T051 ragas 裁判（DashScope qwen-max，需 DASHSCOPE_API_KEY）
pipi scipy                  # 显著性辅助

echo "==> [7/11] 双语医学 NER（易碎，warn-continue）"
pipi spacy scispacy         # T041 英文 NER（scispaCy 对 spaCy 版本敏感）
echo "  装 scispaCy 模型 en_ner_bc5cdr_md（走 s3，临时 turbo ON）"
turbo_on
pip install -i "$PIP_INDEX" "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz" \
  || echo "  [WARN] scispaCy 模型装失败；可后补或换版本（中文 NER 走 transformers 不受影响）"
turbo_off
# 中文医学 NER（CMeEE/CCKS BERT）= transformers token-classification，权重运行时拉，无需额外 pip。

echo "==> [8/11] 训练框架（可选，T019/T035 用；易碎）"
pipi ms-swift               # 或 LLaMA-Factory：pip install llamafactory（baseline_lora.yaml 为 LF 格式）

echo "==> [9/11] 开发/测试"
pipi pytest langchain-mcp-adapters

echo "==> [10/11] 体检：各守卫导入是否就位"
python - <<'PY'
mods = [
    ("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),
    ("peft","peft"),("deepspeed","deepspeed"),("bitsandbytes","bitsandbytes"),
    ("chromadb","chromadb"),("llama_index.core","llama-index-core"),
    ("rank_bm25","rank-bm25"),("sentence_transformers","sentence-transformers"),
    ("open_clip","open_clip_torch"),
    ("cv2","opencv-python-headless"),("sklearn","scikit-learn"),("datasketch","datasketch"),
    ("datasets","datasets"),("kagglehub","kagglehub"),
    ("ragas","ragas"),("langchain_openai","langchain-openai"),("scipy","scipy"),
    ("spacy","spacy"),("scispacy","scispacy"),
    ("mcp","mcp"),("pydantic","pydantic"),("numpy","numpy"),("PIL","pillow"),("yaml","pyyaml"),
    ("pytest","pytest"),("langchain_mcp_adapters","langchain-mcp-adapters"),
]
ok = miss = 0
for imp, pkg in mods:
    try:
        __import__(imp); print(f"  [ok ] {pkg}"); ok += 1
    except Exception as e:
        print(f"  [MISS] {pkg}  ({type(e).__name__})"); miss += 1
# scispaCy 模型单独探一下
try:
    import spacy; spacy.load("en_ner_bc5cdr_md"); print("  [ok ] scispaCy:en_ner_bc5cdr_md")
except Exception:
    print("  [MISS] scispaCy:en_ner_bc5cdr_md（模型未装，可后补）")
print(f"\n  汇总：{ok} 就位 / {miss} 缺失（缺失项多为易碎/可选，守卫导入容忍）")
PY

echo "==> [11/11] 跑单测（含先前 skip 的 torch/重依赖功能测）"
python -m pytest tests/ -q || echo "  [WARN] 部分测试失败/未过，按上方输出排查"

cat <<EOF

============================================================================
依赖安装完成（装在当前 conda 环境）。还需单独下载的**模型权重**（非 pip）：
  - 基座 Qwen3-VL（4B 验证 / 30B 真跑）  → ModelScope，见 scripts/download_assets.sh
  - Qwen3-Embedding-4B（文本嵌入 T014）  → HF: Qwen/Qwen3-Embedding-4B
  - Qwen3-Reranker（精排 T046）           → HF: Qwen/Qwen3-Reranker-*
  - BiomedCLIP（图像嵌入 T031）           → open_clip 首次用时自动从 HF 拉
  - 数据集：bash scripts/download_assets.sh data （c 为 gated，需 export HF_TOKEN）
联网评估（ragas）：export DASHSCOPE_API_KEY=<你的百炼key>
============================================================================
EOF

[[ "$FAILED_REQ" == "1" ]] && { echo "!! 有**必需**依赖安装失败，请回看 [ERROR] 行"; exit 1; } || exit 0
