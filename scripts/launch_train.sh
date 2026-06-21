#!/usr/bin/env bash
# T006 — 多卡启动（占位框架；US1 训练时实跑）
#
# 对卡数自适应（plan.md 资源决策）：
#   - 单卡        → 直接 LoRA（DeepSpeed 关），底线可训。
#   - 多卡(>=2)   → 切 DeepSpeed ZeRO-2（不上 ZeRO-3）。
# 训练不手写循环——走 LLaMA-Factory / ms-swift（见 src/train/framework.py，US1 接入）。
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
CONFIG="${1:-src/train/configs/baseline_lora.yaml}"   # US1 才会存在
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)}"

echo "==> 检测到 GPU 数: $NUM_GPUS ; 训练配置: $CONFIG"

if [[ "$NUM_GPUS" -ge 2 ]]; then
  echo "==> 多卡：DeepSpeed ZeRO-2"
  DS_ARGS=(--deepspeed src/train/configs/ds_zero2.json)   # 配置常备、默认关
else
  echo "==> 单卡：LoRA（DeepSpeed 关）"
  DS_ARGS=()
fi

CMD=(llamafactory-cli train "$CONFIG" "${DS_ARGS[@]}")
echo "+ ${CMD[*]}"
if [[ "$DRY_RUN" == "0" ]]; then "${CMD[@]}"; fi

echo "完成（DRY_RUN=$DRY_RUN）。实跑：DRY_RUN=0 bash scripts/launch_train.sh <config>"
