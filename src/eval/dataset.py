"""固定 held-out 测试集加载器（T015）—— 面积分层 + never-touched 守卫。

归因评估（constitution III）要可信，前提是**评估集固定且从未被训练/选样/去重碰过**。本模块：
- 加载固定测试集（`CTSample`，`in_eval_set=True`）；
- 按 [AreaBand] 分层访问（小病灶命题需分层指标）；
- **never-touched 守卫**：断言评估集 id 与 train/coreset/dedup 用到的 id **不相交**——防泄露（铁律 II）。

纯逻辑，本地全测；真实数据文件在 AutoDL。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.contracts.schemas import AreaBand, CTSample
from src.data.ingest import load_jsonl, ct_records_to_samples


def _sample_fraction(s: CTSample) -> float:
    """病例的代表面积占比：优先 area_band 已定，否则取最大 ROI 占比。"""
    if s.rois:
        return max(r.area_fraction for r in s.rois)
    return 0.0


def _sample_band(s: CTSample) -> AreaBand:
    return s.area_band or AreaBand.from_fraction(_sample_fraction(s))


@dataclass
class HeldoutSet:
    """固定评估集：分层访问 + 守卫。"""

    samples: list[CTSample]

    @property
    def ids(self) -> set[str]:
        return {s.case_id for s in self.samples}

    def by_band(self) -> dict[AreaBand, list[CTSample]]:
        out: dict[AreaBand, list[CTSample]] = {b: [] for b in AreaBand}
        for s in self.samples:
            out[_sample_band(s)].append(s)
        return out

    def band_counts(self) -> dict[str, int]:
        return {b.value: len(v) for b, v in self.by_band().items()}

    def assert_never_touched(self, used_ids: Sequence[str]) -> None:
        """断言评估集与 train/coreset/dedup 用到的 id 不相交（防泄露）。"""
        overlap = self.ids & set(used_ids)
        if overlap:
            raise ValueError(
                f"评估集泄露：{sorted(overlap)} 同时出现在训练/选样集（违反铁律 II）"
            )


def load_heldout(path: str | Path) -> HeldoutSet:
    """从 jsonl 加载固定测试集；只收 `in_eval_set` 为真的样本（守卫源）。"""
    samples = ct_records_to_samples(load_jsonl(path))
    for s in samples:
        s.in_eval_set = True
    return HeldoutSet(samples=samples)


def from_samples(samples: Sequence[CTSample]) -> HeldoutSet:
    """从内存样本构造（测试/已加载）。"""
    return HeldoutSet(samples=[s for s in samples if s.in_eval_set])


__all__ = ["HeldoutSet", "load_heldout", "from_samples"]
