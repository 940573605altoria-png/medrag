#!/usr/bin/env bash
# T005 — AutoDL 脚手架（占位框架；US1+ 上 GPU 时再实跑）
#
# 职责：在 AutoDL 实例上复刻本机 conda env、布置数据盘、开学术加速。
# 当前为占位：结构齐全、步骤注释完整，但默认 dry-run，不实际下载/安装。
# 实跑前去掉 DRY_RUN，并补全锁版本依赖与数据集下载。
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
DATA_ROOT="/root/autodl-tmp"          # AutoDL 数据盘（plan.md 部署节约定）
ENV_NAME="${ENV_NAME:-medrag}"

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" == "0" ]]; then "$@"; fi
}

echo "==> [1/5] 学术加速（AutoDL 自带；github/hf 走镜像）"
# run source /etc/network_turbo

echo "==> [2/5] conda env 复刻（Py3.11，锁版本）"
# run conda create -y -n "$ENV_NAME" python=3.11
# run conda activate "$ENV_NAME"
# run pip install -e ".[train,rag,eval,data]"   # 锁版本见 pyproject.toml

echo "==> [3/5] 数据盘布局：$DATA_ROOT/{raw,weights,chroma,runs}"
for d in raw weights chroma runs; do
  run mkdir -p "$DATA_ROOT/$d"
done

echo "==> [4/5] 数据集下载到 $DATA_ROOT/raw（数据到位后启用）"
# a 药品: kagglehub imtkaggleteam/medical-information-dataset
# b QA  : HF lavita/medical-qa-datasets (all-processed)
# c CT  : HF UCSC-VLAA/MedTrinity-25M (25M_demo)

echo "==> [5/5] 权重下载到 $DATA_ROOT/weights（Qwen3-VL / embedding / reranker）"

echo "完成（DRY_RUN=$DRY_RUN）。实跑：DRY_RUN=0 bash scripts/autodl_setup.sh"
