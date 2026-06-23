#!/usr/bin/env bash
# AutoDL 环境 bring-up（环境+模型冒烟批次）。
#
# 适用：AutoDL 实例上、已激活目标 conda 环境（如 `conda activate medrag`）。
#
# 学术加速策略（按下载渠道开关，非一刀切）：
#   - 清华源 / hf-mirror / ModelScope 等镜像 → **直连，关加速**（turbo_off）。
#   - GitHub / huggingface 官网 / kaggle 等   → **开加速**（turbo_on）。
#   每步按其渠道设状态；turbo_on/off 幂等，相邻同状态不重复切换、变了才切。
#   *本脚本的下载全走镜像/清华源，故全程 turbo OFF；函数留给需要官网/GitHub 的步骤用。*
#
# 可调环境变量：
#   PIP_INDEX   pip 源（默认清华）
#   TORCH_CUDA  官方 torch 轮子 CUDA tag（仅走 pytorch.org 兜底时用，默认 cu124）
#   DATA_ROOT   数据盘根（默认 /root/autodl-tmp）
set -euo pipefail
trap 'echo "  !! 某步失败。镜像源装失败可重试；若改用官网/GitHub 记得 turbo_on" >&2' ERR

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# ── 学术加速开关（按渠道调用）──────────────────────────────
turbo_on() {
  if [[ -f /etc/network_turbo ]]; then
    set +u; source /etc/network_turbo; set -u   # 该脚本仅 export 代理变量
    echo "  [turbo] ON（github/huggingface官网/kaggle 用）"
  else
    echo "  [turbo] /etc/network_turbo 不存在，跳过（非 AutoDL？）"
  fi
}
turbo_off() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
  echo "  [turbo] OFF（清华源/镜像直连）"
}

echo "==> [1/5] 网络与镜像"
turbo_off                                        # 起步即关，后续按渠道再开
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # 模型走 HF 镜像
echo "  pip 源 = $PIP_INDEX ; HF_ENDPOINT = $HF_ENDPOINT"

echo "==> [2/5] 数据盘布局：$DATA_ROOT/{raw,weights,chroma,runs}"
for d in raw weights chroma runs; do mkdir -p "$DATA_ROOT/$d"; done
export HF_HOME="${HF_HOME:-$DATA_ROOT/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$DATA_ROOT/.cache/modelscope}"
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE"

echo "==> [3/5] 校验/安装 torch（清华源，直连，关加速）"
turbo_off
if ! python -c "import torch" 2>/dev/null; then
  echo "  未检测到 torch → 从清华源装 torch+torchvision（PyPI 默认 CUDA 构建）"
  pip install torch torchvision -i "$PIP_INDEX"
  # 若默认构建 CUDA 不可用（torch.cuda.is_available()=False），改用官方 cu 轮子：
  #   该渠道是 pytorch.org 官网 → 先 turbo_on 再装：
  #   turbo_on; TORCH_CUDA="${TORCH_CUDA:-cu124}"; \
  #     pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"; turbo_off
fi
python - <<'PY'
import sys
try:
    import torch
    print(f"  torch {torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()} n={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("  !! CUDA 不可用——见脚本注释改用官方 cu 轮子（turbo_on）后重试", file=sys.stderr)
        sys.exit(1)
except ImportError:
    print("  !! torch 安装失败——重试，或改官方轮子（turbo_on）后重跑", file=sys.stderr)
    sys.exit(1)
PY

echo "==> [4/5] 安装 medrag 包 + 推理依赖（清华源，直连，关加速）"
turbo_off
pip install -e . --no-deps -i "$PIP_INDEX"            # 我们的包本体
pip install pydantic "mcp[cli]" numpy -i "$PIP_INDEX" # 基础运行依赖
pip install "transformers>=4.46" accelerate peft modelscope pillow -i "$PIP_INDEX"
pip install opencv-python-headless scikit-learn pytest -i "$PIP_INDEX"   # US1 数据路：T024/T026 绿框+inpaint(cv2)、T027 coreset(sklearn KMeans)；pytest 跑单测

echo "==> [5/5] L0 冒烟（环境 + 骨架，不下大模型）"
python scripts/smoke_gpu.py

cat <<EOF

环境就绪（装在当前激活的 conda 环境里；全程走镜像/清华源，未开加速）。下一步：
  - 模型冒烟(L1, 当前 4090 用 4B；走 hf-mirror，仍不需加速)：
      export HF_ENDPOINT=$HF_ENDPOINT
      python scripts/smoke_gpu.py --with-model --model-id Qwen/Qwen3-VL-4B-Instruct
  - 升级实例后真跑 30B： python scripts/smoke_gpu.py --with-model   # 显存紧加 --quant 4bit
导出的环境变量仅本 shell 生效；长期生效写入 ~/.bashrc（见 runbook）。
EOF
