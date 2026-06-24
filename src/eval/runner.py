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

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from src.config.config import AppConfig
from src.contracts.schemas import EvalRecord
from src.eval import stats as st

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


# ════════════════════════════════════════════════════════════════════
# 质量门（T056）—— 增益不显著 → 阻断进入下一阶段
# ════════════════════════════════════════════════════════════════════

class QualityGateError(RuntimeError):
    """质量门阻断：候选改动相对基线的增益不达标（不显著/方向反/效应过小）。"""


@dataclass
class GateConfig:
    """质量门判据（constitution III：一次一变量 + 增益须真实显著）。

    - `require_significant_ci`：配对 delta CI 必须**在更优方向上不含 0**（噪声排除）。
    - `max_p_value`：配对置换检验 p 的上限。
    - `min_delta`：最小**有意义增益**（效应量下限，挡住"统计显著但实务无意义"的微小改动）。
    - `higher_is_better`：多数指标越高越好；FP/图、拒答误伤等越低越好时置 False。
    """

    require_significant_ci: bool = True
    max_p_value: float = 0.05
    min_delta: float = 0.0
    higher_is_better: bool = True


@dataclass
class GateResult:
    """质量门裁决：过/挡 + 原因 + 复核用的统计量。"""

    passed: bool
    reason: str
    delta: float
    p_value: float
    ci_low: float
    ci_high: float


@dataclass
class QualityGate:
    """把"增益是否真实显著"固化成一道门：不达标就挡住、可硬抛异常阻断流水线。

    既能吃**已算好的统计量**（`check`，复用 ablation 的 Comparison），也能吃**逐样本配对得分**
    （`check_scores`，自带 bootstrap CI + 置换检验，复用 [stats]）。
    """

    config: GateConfig = field(default_factory=GateConfig)
    n_resamples: int = 2000
    level: float = 0.95
    seed: int | None = 0

    def check(
        self, *, delta: float, p_value: float, ci_low: float, ci_high: float
    ) -> GateResult:
        """对已有统计量裁决（delta = 变体−基线，方向由 config 决定）。"""
        cfg = self.config
        improvement = delta if cfg.higher_is_better else -delta
        direction_ok = (ci_low > 0.0) if cfg.higher_is_better else (ci_high < 0.0)

        reasons: list[str] = []
        if cfg.require_significant_ci and not direction_ok:
            reasons.append(
                f"配对 delta CI 在更优方向上含 0（[{ci_low:.4f}, {ci_high:.4f}]）"
            )
        if p_value > cfg.max_p_value:
            reasons.append(f"p={p_value:.4f} > {cfg.max_p_value}")
        if improvement < cfg.min_delta:
            reasons.append(f"增益 {improvement:.4f} < 最小阈值 {cfg.min_delta}")

        passed = not reasons
        return GateResult(
            passed=passed,
            reason="通过（增益显著且达效应量阈值）" if passed else "；".join(reasons),
            delta=delta, p_value=p_value, ci_low=ci_low, ci_high=ci_high,
        )

    def check_scores(
        self, baseline: Sequence[float], variant: Sequence[float]
    ) -> GateResult:
        """从逐样本配对得分裁决：先算 bootstrap delta CI + 置换检验 p，再走 `check`。"""
        ci = st.paired_bootstrap_delta_ci(
            variant, baseline, n_resamples=self.n_resamples, level=self.level, seed=self.seed
        )
        p = st.paired_permutation_test(
            variant, baseline, n_resamples=self.n_resamples, seed=self.seed
        )
        return self.check(delta=ci.point, p_value=p, ci_low=ci.low, ci_high=ci.high)

    def assert_pass(self, result: GateResult) -> GateResult:
        """门没过就抛 QualityGateError（用于"不显著则阻断进入下一阶段"的硬卡点）。"""
        if not result.passed:
            raise QualityGateError(result.reason)
        return result


__all__ = [
    "PredictFn",
    "MetricFn",
    "EvalSpec",
    "run_eval",
    "QualityGateError",
    "GateConfig",
    "GateResult",
    "QualityGate",
]
