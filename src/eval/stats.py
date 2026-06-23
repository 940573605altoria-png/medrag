"""显著性与置信区间（T053）—— bootstrap CI + 配对检验。

**为什么这是归因底座的收口**：消融跑出"创新比基线高 1.8 个点"还不够——constitution III
要求回答"这 1.8 是真增益还是噪声"。本模块提供：
- **bootstrap 百分位 CI**：对单系统指标给区间，看它稳不稳。
- **配对 bootstrap delta CI**：同一批样本上 A vs B 的差值区间（配对消去样本难度方差，更敏感）。
- **配对置换检验**：对差值做符号翻转置换，给两侧 p 值（假设极弱，适合小评估集）。
- **McNemar 检验**：拒答正确性/溯源命中这类**逐样本二值**结果的配对显著性。

全部 numpy、可固定 seed（复现铁律）；纯逻辑，本地全量单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# 统计量默认取均值；可换中位数等。
Statistic = Callable[[np.ndarray], float]


@dataclass
class CI:
    """点估计 + 双侧置信区间。"""

    point: float
    low: float
    high: float
    level: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "ci_low": self.low, "ci_high": self.high}


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Statistic = lambda x: float(np.mean(x)),
    n_resamples: int = 10000,
    level: float = 0.95,
    seed: int | None = 0,
) -> CI:
    """对单组样本做有放回重采样，取统计量分布的百分位作为 CI。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci 需要非空样本")
    rng = _rng(seed)
    n = arr.size
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        stats[i] = statistic(arr[rng.integers(0, n, n)])
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return CI(point=statistic(arr), low=float(lo), high=float(hi), level=level)


def paired_bootstrap_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10000,
    level: float = 0.95,
    seed: int | None = 0,
) -> CI:
    """配对差值 (a-b) 的均值 bootstrap CI。a/b 为同一批样本上两系统的逐样本指标。

    CI 不含 0 → 在该置信水平下差异显著。
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"配对样本形状不一致：{a_arr.shape} vs {b_arr.shape}")
    if a_arr.size == 0:
        raise ValueError("paired_bootstrap_delta_ci 需要非空样本")
    diff = a_arr - b_arr
    rng = _rng(seed)
    n = diff.size
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        means[i] = diff[rng.integers(0, n, n)].mean()
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return CI(point=float(diff.mean()), low=float(lo), high=float(hi), level=level)


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int | None = 0,
) -> float:
    """配对符号翻转置换检验，返回两侧 p 值。

    H0：a 与 b 的配对差值对称分布于 0。随机给每个差值乘 ±1 模拟 H0，统计 |均值| 不小于
    观测的比例。用 (count+1)/(n+1) 修正，避免 p=0。
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"配对样本形状不一致：{a_arr.shape} vs {b_arr.shape}")
    diff = a_arr - b_arr
    if diff.size == 0:
        raise ValueError("paired_permutation_test 需要非空样本")
    observed = abs(diff.mean())
    rng = _rng(seed)
    n = diff.size
    count = 0
    for _ in range(n_resamples):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs((diff * signs).mean()) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_resamples + 1)


def mcnemar_test(
    a_correct: Sequence[bool], b_correct: Sequence[bool], *, exact_threshold: int = 25
) -> dict[str, float]:
    """配对二值结果的 McNemar 检验（如拒答是否正确、溯源是否命中）。

    只看分歧对：b01 = a 错 b 对、b10 = a 对 b 错。小样本走精确二项，否则连续性校正卡方。
    返回 p 值与不一致计数。
    """
    a_arr = np.asarray(a_correct, dtype=bool)
    b_arr = np.asarray(b_correct, dtype=bool)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"配对样本形状不一致：{a_arr.shape} vs {b_arr.shape}")
    b01 = int(np.sum(~a_arr & b_arr))   # a 错 b 对
    b10 = int(np.sum(a_arr & ~b_arr))   # a 对 b 错
    n = b01 + b10
    if n == 0:
        p = 1.0
    elif n < exact_threshold:
        # 精确：H0 下 min(b01,b10) ~ Binom(n, 0.5)，双侧
        from math import comb

        k = min(b01, b10)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
        p = min(1.0, 2.0 * tail)
    else:
        chi2 = (abs(b01 - b10) - 1.0) ** 2 / n  # 连续性校正
        p = float(np.exp(-chi2 / 2.0))  # 卡方(df=1) 生存函数 = exp(-x/2)
    return {"p_value": p, "b01": float(b01), "b10": float(b10), "n_discordant": float(n)}


def is_significant(ci: CI) -> bool:
    """配对 delta CI 是否不含 0（该水平下显著）。"""
    return ci.low > 0.0 or ci.high < 0.0


__all__ = [
    "CI",
    "Statistic",
    "bootstrap_ci",
    "paired_bootstrap_delta_ci",
    "paired_permutation_test",
    "mcnemar_test",
    "is_significant",
]
