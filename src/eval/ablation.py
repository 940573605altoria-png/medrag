"""消融矩阵 runner（T054）—— 一次一变量、相对基线 delta、小病灶分层、配对显著性。

这是归因底座的**总装**：把 [config.AppConfig] 的开关、metrics_* 的尺子、[stats] 的显著性串成
一句可证伪的话——"打开开关 X，指标在小病灶上相对基线涨了 Δ，且 p<0.05"。

铁律落地（constitution III）：
- **一次只改一个变量**：`build_variants` 从基线 config 出发，每个变体只覆盖**一个** flag，
  绝不同时动两个——否则增益归因不清。
- **相对基线 + 配对显著性**：在**同一批样本**上取基线与变体的逐样本得分，配对算 delta CI
  （[stats.paired_bootstrap_delta_ci]）+ 置换检验 p（[stats.paired_permutation_test]），消去样本
  难度方差，比独立比较更敏感。
- **小病灶分层贯穿**：给每个样本一个 [schemas.AreaBand] 标签，在每个带内单独复算 delta+显著性，
  专门验证"增益是否集中在 <2% 小病灶"（本项目核心命题）。

逐样本得分 `score_fn(config) -> np.ndarray`（长度/顺序在各 config 间一致）由调用方提供：可以是
每病例的检测 sensitivity@固定FP、每查询 recall@k、每报告实体F1……粒度对齐配对检验即可。纯 numpy，
本地全量单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from src.config.config import AppConfig
from src.contracts.schemas import AreaBand
from src.eval import stats as st
from src.eval.runner import GateResult, QualityGate

# 逐样本得分：吃 config（含 flags）→ 每样本一个标量得分（顺序与基线对齐）。
ScoreFn = Callable[[AppConfig], Sequence[float]]


@dataclass
class Comparison:
    """变体 vs 基线在某一(子)样本集上的配对比较结果。"""

    baseline_mean: float
    variant_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    n: int


@dataclass
class AblationCell:
    """消融矩阵的一格：一个单变量变体的整体 + 分层比较（+ 可选质量门裁决）。"""

    name: str            # 如 "fusion=add"
    flag: str            # 被改的开关
    value: Any           # 该开关取值
    overall: Comparison
    by_band: dict[str, Comparison] = field(default_factory=dict)
    gate: GateResult | None = None   # 给了质量门时，整体增益是否过门


def build_variants(
    base: AppConfig, sweep: dict[str, Sequence[Any]]
) -> list[tuple[str, str, Any, AppConfig]]:
    """从基线派生"只改一个 flag"的变体列表，返回 (name, flag, value, config)。

    跳过与基线同值的取值（那等于基线、无对比意义）。基线本身不在返回列表里。
    """
    variants: list[tuple[str, str, Any, AppConfig]] = []
    for flag, values in sweep.items():
        for v in values:
            if base.flags.get(flag) == v:
                continue
            cfg = base.with_overrides(flags={flag: v})
            variants.append((f"{flag}={v}", flag, v, cfg))
    return variants


def _compare(
    baseline: np.ndarray,
    variant: np.ndarray,
    *,
    n_resamples: int,
    level: float,
    seed: int | None,
) -> Comparison:
    """配对比较：均值 delta + bootstrap delta CI + 置换检验 p + 是否显著。"""
    ci = st.paired_bootstrap_delta_ci(
        variant, baseline, n_resamples=n_resamples, level=level, seed=seed
    )
    p = st.paired_permutation_test(variant, baseline, n_resamples=n_resamples, seed=seed)
    return Comparison(
        baseline_mean=float(baseline.mean()),
        variant_mean=float(variant.mean()),
        delta=ci.point,
        ci_low=ci.low,
        ci_high=ci.high,
        p_value=p,
        significant=st.is_significant(ci),
        n=int(baseline.size),
    )


def run_ablation(
    base: AppConfig,
    sweep: dict[str, Sequence[Any]],
    score_fn: ScoreFn,
    *,
    bands: Sequence[AreaBand] | None = None,
    gate: QualityGate | None = None,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int | None = 0,
) -> list[AblationCell]:
    """对每个单变量变体跑配对消融，返回消融矩阵（含按面积带分层）。

    `bands` 若给出（与样本等长），则在每个 AreaBand 子集内复算 delta+显著性——验证增益是否集中
    在小病灶。`gate` 若给出，则对每个变体的整体增益做质量门裁决（T056），过不了的变体应被阻断进入
    下一阶段（见 `blocked`）。
    """
    base_scores = np.asarray(score_fn(base), dtype=float)
    bands_arr = np.asarray([b.value for b in bands]) if bands is not None else None
    if bands_arr is not None and bands_arr.shape[0] != base_scores.shape[0]:
        raise ValueError("bands 长度须与逐样本得分一致")

    cells: list[AblationCell] = []
    for name, flag, value, cfg in build_variants(base, sweep):
        v_scores = np.asarray(score_fn(cfg), dtype=float)
        if v_scores.shape != base_scores.shape:
            raise ValueError(f"变体 {name} 得分形状与基线不一致（配对检验要求同批同序）")
        overall = _compare(
            base_scores, v_scores, n_resamples=n_resamples, level=level, seed=seed
        )
        by_band: dict[str, Comparison] = {}
        if bands_arr is not None:
            for band in AreaBand:
                mask = bands_arr == band.value
                if not mask.any():
                    continue
                by_band[band.value] = _compare(
                    base_scores[mask], v_scores[mask],
                    n_resamples=n_resamples, level=level, seed=seed,
                )
        gate_result = (
            gate.check(delta=overall.delta, p_value=overall.p_value,
                       ci_low=overall.ci_low, ci_high=overall.ci_high)
            if gate is not None else None
        )
        cells.append(AblationCell(name=name, flag=flag, value=value,
                                  overall=overall, by_band=by_band, gate=gate_result))
    return cells


def blocked(cells: Sequence[AblationCell]) -> list[AblationCell]:
    """返回被质量门阻断的变体（gate 已算且未通过）——这些不该进入下一阶段。"""
    return [c for c in cells if c.gate is not None and not c.gate.passed]


def to_table(cells: Sequence[AblationCell]) -> list[dict[str, Any]]:
    """把消融矩阵拍平成可打印/可存表的行（每变体一行，含整体 delta/p + 各带 delta）。"""
    rows: list[dict[str, Any]] = []
    for c in cells:
        row: dict[str, Any] = {
            "variant": c.name,
            "delta": c.overall.delta,
            "ci_low": c.overall.ci_low,
            "ci_high": c.overall.ci_high,
            "p_value": c.overall.p_value,
            "significant": c.overall.significant,
        }
        for band, comp in c.by_band.items():
            row[f"{band}/delta"] = comp.delta
            row[f"{band}/significant"] = comp.significant
        if c.gate is not None:
            row["gate_passed"] = c.gate.passed
            row["gate_reason"] = c.gate.reason
        rows.append(row)
    return rows


__all__ = [
    "ScoreFn",
    "Comparison",
    "AblationCell",
    "build_variants",
    "run_ablation",
    "blocked",
    "to_table",
]
