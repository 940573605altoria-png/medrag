"""病灶面积分层工具（T010）—— <2% / 2–5% / >5% 的分组与分布。

分层主轴在 [contracts.schemas.AreaBand]（唯一事实源，避免重复实现）；本模块只提供**按带分组/计分布**
的便捷函数，给评估分层（T049/T054）与采样/统计复用。纯逻辑。
"""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

from src.contracts.schemas import AreaBand

T = TypeVar("T")


def band_of(area_fraction: float) -> AreaBand:
    """面积占比 → 面积带（薄包 `AreaBand.from_fraction`）。"""
    return AreaBand.from_fraction(area_fraction)


def group_by_band(
    items: Sequence[T], *, fraction_fn: Callable[[T], float]
) -> dict[AreaBand, list[T]]:
    """按每个元素的面积占比分到三带。"""
    out: dict[AreaBand, list[T]] = {b: [] for b in AreaBand}
    for it in items:
        out[band_of(fraction_fn(it))].append(it)
    return out


def band_distribution(
    items: Sequence[T], *, fraction_fn: Callable[[T], float]
) -> dict[str, int]:
    """各带样本计数（`{band_value: n}`）。"""
    return {b.value: len(v) for b, v in group_by_band(items, fraction_fn=fraction_fn).items()}


__all__ = ["band_of", "group_by_band", "band_distribution"]
