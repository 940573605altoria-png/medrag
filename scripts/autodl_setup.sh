#!/usr/bin/env bash
# AutoDL 环境 bring-up（环境+模型冒烟批次）。
#
# 假设：用 AutoDL 的 PyTorch 官方镜像（torch + CUDA 已随镜像装好）。
# 本脚本**不碰 torch**——只补我们的包依赖 + 推理所需库（transformers/accelerate/peft/
# modelscope/pillow），避免重装 torch 触发版本churn。若你的镜像没有 torch，先按 runbook
# 用匹配 CUDA 的 index-url 单独装 torch，再跑本脚本。
#
# 用法（在 AutoDL 实例，项目已 clone 到当前目录）:
#   bash scripts/autodl_setup.sh
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"

echo "==> [1/5] 学术加速（AutoDL 自带；存在则启用）"
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo && echo "  network_turbo on" || echo "  network_turbo 跳过"
fi
# HF 镜像兜底（modelscope 为主时可不依赖）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "  HF_ENDPOINT=$HF_ENDPOINT"

echo "==> [2/5] 数据盘布局：$DATA_ROOT/{raw,weights,chroma,runs}"
for d in raw weights chroma runs; do mkdir -p "$DATA_ROOT/$d"; done
# 把数据/权重/缓存都引到数据盘，免撑爆系统盘
export HF_HOME="${HF_HOME:-$DATA_ROOT/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$DATA_ROOT/.cache/modelscope}"
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE"

echo "==> [3/5] 校验 torch / CUDA（镜像应已装）"
python - <<'PY'
import sys
try:
    import torch
    print(f"  torch {torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()} n={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("  !! CUDA 不可用——检查镜像/驱动后再继续", file=sys.stderr)
except ImportError:
    print("  !! 未检测到 torch——请先按 runbook 装匹配 CUDA 的 torch", file=sys.stderr)
    sys.exit(1)
PY

echo "==> [4/5] 安装 medrag 包（轻量）+ 推理依赖（不动 torch）"
pip install -e . --no-deps        # 我们的包本体
pip install pydantic "mcp[cli]" numpy   # 基础运行依赖
# 推理/训练库：不带 torch，沿用镜像里的 torch
pip install "transformers>=4.46" accelerate peft modelscope pillow

echo "==> [5/5] L0 冒烟（环境 + 骨架，不下大模型）"
python scripts/smoke_gpu.py

cat <<EOF

环境就绪。下一步（按需）：
  - 下权重 + 数据：    bash scripts/download_assets.sh weights      # Qwen3-VL 30B（约 60GB）
                       bash scripts/download_assets.sh data         # a/b/c 数据集
  - 模型冒烟(L1)：     python scripts/smoke_gpu.py --with-model
                       # 显存紧：  python scripts/smoke_gpu.py --with-model --quant 4bit
导出的环境变量（HF_HOME/MODELSCOPE_CACHE/HF_ENDPOINT）仅本 shell 生效；
长期生效请写入 ~/.bashrc（见 runbook）。
EOF
