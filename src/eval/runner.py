"""配置驱动评估 runner 骨架（T016 的另一半）—— 归因底座的执行器。

**为什么"配置驱动"**：constitution III 的"一次只改一个变量"靠的就是——评估的全部行为由
[config.AppConfig] 的 `flags` 决定，runner 只负责"按这套 flags 跑系统 → 算指标 → 记成
EvalRecord"。消融（T054）就是反复用不同的单变量 config 调本 runner。

runner 刻意**与模型/数据解耦**（骨架可本地测、不拉 GPU）：
- `predict_fn(config)`：按 config 产出 metric_fn 需要的中间物（预测、检索结果、报告……）。
- `metric_fn(outputs)`：返回 `{"metrics": {...}, "stratified": {...}}`——正好是 metrics_* 模块
  的统一输出形态（detection_metrics / report_metrics / e2e_metrics / retrieval_metrics）。

真实组件接进来时只换 predict_fn 的实现，runner 与记录格式不变（接口先行，集成风险前置）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.config.config import AppConfig
from src.contracts.schemas import EvalRecord

# 系统执行：吃 config（含 flags 开关）→ 产出待评中间物。
PredictFn = Callable[[AppConfig], Any]
# 指标计算：吃中间物 → {"metrics": {...}, "stratified": {...}}。
MetricFn = Callable[[Any], dict]


@dataclass
class EvalSpec:
    """一次评估的"做什么"：run 标识 + 系统执行 + 指标计算。"""

    run_id: str
    predict_fn: PredictFn
    metric_fn: MetricFn


def run_eval(
    config: AppConfig, spec: EvalSpec, *, baseline_id: str | None = None
) -> EvalRecord:
    """按 config 跑一次评估，落成 EvalRecord（含 flags 快照、整体+分层指标）。"""
    outputs = spec.predict_fn(config)
    result = spec.metric_fn(outputs)
    if "metrics" not in result:
        raise ValueError("metric_fn 必须返回含 'metrics' 键的 dict（见 metrics_* 模块约定）")
    return EvalRecord(
        run_id=spec.run_id,
        flags=dict(config.flags),
        metrics={k: float(v) for k, v in result["metrics"].items()},
        stratified={
            band: {k: float(v) for k, v in m.items()}
            for band, m in result.get("stratified", {}).items()
        },
        baseline_id=baseline_id,
    )


__all__ = ["PredictFn", "MetricFn", "EvalSpec", "run_eval"]
