#!/usr/bin/env bash
# 下载权重 + 数据集到数据盘（与 autodl_setup.sh 配套）。
#
# 权重走 ModelScope（国内快）；数据集走 HuggingFace datasets（hf-mirror 兜底）。
# 用法:
#   bash scripts/download_assets.sh weights    # 仅 Qwen3-VL 30B 权重
#   bash scripts/download_assets.sh data       # 仅 a/b/c 数据集
#   bash scripts/download_assets.sh all        # 两者
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
WEIGHTS_DIR="$DATA_ROOT/weights"
RAW_DIR="$DATA_ROOT/raw"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$DATA_ROOT/.cache/modelscope}"

# 默认 30B（真跑）；4090 冒烟时覆盖： MEDRAG_BASE_MODEL_MS=Qwen/Qwen3-VL-4B-Instruct
MODEL_ID="${MEDRAG_BASE_MODEL_MS:-Qwen/Qwen3-VL-30B-A3B-Instruct}"

download_weights() {
  local name; name="$(basename "$MODEL_ID")"   # 本地目录名随模型走，避免小/大模型互相覆盖
  local dest="$WEIGHTS_DIR/$name"
  echo "==> 权重：$MODEL_ID → $dest（ModelScope；30B 约 60GB，4B 约 8GB）"
  mkdir -p "$WEIGHTS_DIR"
  modelscope download --model "$MODEL_ID" --local_dir "$dest"
  echo "  完成。冒烟时： export MEDRAG_BASE_MODEL=$dest"
}

download_data() {
  echo "==> 数据集 a/b/c → $RAW_DIR（HF datasets；a 需 kaggle token）"
  mkdir -p "$RAW_DIR"
  python - <<'PY'
import os
raw = os.path.join(os.environ.get("DATA_ROOT", "/root/autodl-tmp"), "raw")

# b 英文医学QA
from datasets import load_dataset
print("[b] lavita/medical-qa-datasets (all-processed) ...")
b = load_dataset("lavita/medical-qa-datasets", "all-processed")
b.save_to_disk(os.path.join(raw, "b_medqa"))
print("    saved ->", os.path.join(raw, "b_medqa"))

# c 脑CT图文（demo 子集）
print("[c] UCSC-VLAA/MedTrinity-25M (25M_demo) ...")
c = load_dataset("UCSC-VLAA/MedTrinity-25M", "25M_demo")
c.save_to_disk(os.path.join(raw, "c_ct"))
print("    saved ->", os.path.join(raw, "c_ct"))
PY
  echo "  [a] 药品信息走 kagglehub（需 ~/.kaggle/kaggle.json token）："
  python - <<'PY'
import os
try:
    import kagglehub
    p = kagglehub.dataset_download("imtkaggleteam/medical-information-dataset")
    print("    [a] downloaded ->", p)
except Exception as e:  # noqa: BLE001
    print("    [a] 跳过（需配置 kaggle token）：", repr(e))
PY
}

cmd="${1:-all}"
case "$cmd" in
  weights) download_weights ;;
  data)    download_data ;;
  all)     download_weights; download_data ;;
  *) echo "用法: bash scripts/download_assets.sh [weights|data|all]"; exit 2 ;;
esac
