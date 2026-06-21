#!/usr/bin/env bash
# T006 — 断点续训（占位框架；US1 训练时实跑）
#
# 从最近 checkpoint 恢复。AutoDL 抢占/重启后保训练进度。
# 复用 launch_train.sh 的卡数自适应逻辑，仅追加 --resume_from_checkpoint。
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
CKPT_DIR="${CKPT_DIR:-/root/autodl-tmp/runs}"

LATEST="$(ls -dt "$CKPT_DIR"/checkpoint-* 2>/dev/null | head -n1 || true)"
if [[ -z "$LATEST" ]]; then
  echo "未找到 checkpoint（$CKPT_DIR/checkpoint-*）；改用 scripts/launch_train.sh 从头训。"
  exit 0
fi

echo "==> 从 $LATEST 续训"
export RESUME_FROM="$LATEST"
DRY_RUN="$DRY_RUN" bash "$(dirname "$0")/launch_train.sh" "${1:-}"
