"""评估记录持久化（T016 的一半）—— EvalRecord 落盘/读取/汇总。

EvalRecord 的**结构**已在 [contracts.schemas] 定死（config→metrics 的归因单元）；本模块只管它的
**生命周期**：把一次 run 的记录写进 `runs_dir`、读回、拍平成可比对的表行。消融 runner（T054）和
质量门（T056）都消费这里的产物。

为什么落盘：constitution III 要求"可复现 + 可归因"——每次 run 的 flags+指标存成 json，事后能复盘
"哪套开关→什么指标"，也能跨会话比对基线。纯标准库，无重依赖。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.contracts.schemas import EvalRecord


def save_record(record: EvalRecord, runs_dir: str | Path) -> Path:
    """把一条 EvalRecord 写成 `<runs_dir>/<run_id>.json`，返回路径。"""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.run_id}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_record(path: str | Path) -> EvalRecord:
    """从 json 读回一条 EvalRecord。"""
    text = Path(path).read_text(encoding="utf-8")
    return EvalRecord.model_validate_json(text)


def load_records(runs_dir: str | Path) -> list[EvalRecord]:
    """读 `runs_dir` 下所有 `*.json` 记录（按 run_id 排序）。"""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    out = [load_record(p) for p in sorted(runs_dir.glob("*.json"))]
    return sorted(out, key=lambda r: r.run_id)


def flatten(record: EvalRecord) -> dict[str, float]:
    """拍平成单层表行：整体指标 + `band/metric` 分层指标，便于并排比对。"""
    row: dict[str, float] = dict(record.metrics)
    for band, metrics in record.stratified.items():
        for k, v in metrics.items():
            row[f"{band}/{k}"] = v
    return row


def delta_row(record: EvalRecord, baseline: EvalRecord) -> dict[str, float]:
    """逐指标算 record 相对 baseline 的 delta（只对两边都有的键）。"""
    a, b = flatten(record), flatten(baseline)
    return {k: a[k] - b[k] for k in a if k in b}


__all__ = [
    "save_record",
    "load_record",
    "load_records",
    "flatten",
    "delta_row",
]
