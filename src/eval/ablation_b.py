"""创新 B 三臂消融专项报告（T055）—— add / concat / multiimg vs global-only 基线。

**为什么要专门一个报告**：B 的论点不是"融合能涨点"，而是一句更尖锐的话——"**门控残差相加
(add) 用极小代价，把增益精准砸在 <2% 小病灶上**"。多图输入(multiimg)是最强但最贵的基线，拼接
(concat) 居中。泛泛的整体 delta 会把这个故事抹平，所以本模块在 T054 通用消融之上，专门：

1. 固定基线 = `fusion=off`（全局图单路，global-only），三臂只各改 `fusion` 这一个 flag（铁律：
   一次一变量）；
2. 逐臂拆出**按面积带的 delta**，判定每臂的增益是否**集中在小病灶**（small 带显著 且 small 的
   delta 严格大于其它各带）；
3. 给出三个 headline：整体最强臂、小病灶最强臂、"增益集中小病灶"的臂——B 的卖点正是后者可能
   ≠ 前者（multiimg 整体最强，但 add 才是把好处集中在小病灶上的那个）。

复用 [ablation.run_ablation]（配对 delta CI + 置换检验 + 分层）与 [runner.QualityGate]；纯逻辑，
B 真数据出来前可用桩 `score_fn` 跑通整套框架，本地全量单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.config.config import AppConfig
from src.contracts.schemas import AreaBand
from src.eval.ablation import AblationCell, ScoreFn, run_ablation
from src.eval.runner import QualityGate

FUSION_FLAG = "fusion"
BASELINE_ARM = "off"                       # global-only 单路
B_ARMS: tuple[str, ...] = ("add", "concat", "multiimg")


@dataclass
class ArmSummary:
    """单臂相对 global-only 基线的战报。"""

    arm: str
    overall_delta: float
    overall_significant: bool
    gate_passed: bool | None               # 给了质量门时的整体裁决
    band_delta: dict[str, float]           # band -> delta
    band_significant: dict[str, bool]      # band -> 是否显著
    small_concentrated: bool               # 小病灶带显著 且 delta 严格大于其它各带


@dataclass
class BReport:
    """B 三臂消融汇总：每臂战报 + 三个 headline。"""

    arms: list[ArmSummary]
    best_overall: str | None               # 整体 delta 最大且显著的臂
    best_small: str | None                 # 小病灶 delta 最大且显著的臂
    concentrated_arms: list[str] = field(default_factory=list)  # 增益集中小病灶的臂

    def to_table(self) -> list[dict]:
        """每臂一行：整体 + 各带 delta + 是否集中小病灶 + 过门。"""
        rows: list[dict] = []
        for a in self.arms:
            row: dict = {
                "arm": a.arm,
                "overall_delta": a.overall_delta,
                "overall_significant": a.overall_significant,
                "small_concentrated": a.small_concentrated,
            }
            if a.gate_passed is not None:
                row["gate_passed"] = a.gate_passed
            for band, d in a.band_delta.items():
                row[f"{band}/delta"] = d
            rows.append(row)
        return rows

    def summary_lines(self) -> list[str]:
        """人读小结（可打印进评估日志）。"""
        lines = [
            f"B 三臂 vs global-only：整体最强={self.best_overall or '无显著臂'}"
            f"，小病灶最强={self.best_small or '无显著臂'}",
            f"增益集中在 <2% 小病灶的臂：{', '.join(self.concentrated_arms) or '无'}",
        ]
        for a in self.arms:
            small = AreaBand.SMALL.value
            lines.append(
                f"  - {a.arm}: 整体Δ={a.overall_delta:+.4f}"
                f"（{'显著' if a.overall_significant else '不显著'}）"
                f"，小病灶Δ={a.band_delta.get(small, float('nan')):+.4f}"
                f"，集中小病灶={'是' if a.small_concentrated else '否'}"
            )
        return lines


def run_b_ablation(
    base: AppConfig,
    score_fn: ScoreFn,
    bands: Sequence[AreaBand],
    *,
    arms: Sequence[str] = B_ARMS,
    gate: QualityGate | None = None,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int | None = 0,
) -> list[AblationCell]:
    """跑 B 三臂配对消融：**强制基线为 global-only(fusion=off)**，只 sweep `fusion`。

    `bands` 必给（B 的命题就是分层验小病灶）。返回 T054 的 AblationCell 列表。
    """
    base = base.with_overrides(flags={FUSION_FLAG: BASELINE_ARM})
    return run_ablation(
        base, {FUSION_FLAG: list(arms)}, score_fn,
        bands=bands, gate=gate, n_resamples=n_resamples, level=level, seed=seed,
    )


def build_report(cells: Sequence[AblationCell]) -> BReport:
    """把消融格子汇总成 B 报告（判定每臂是否"增益集中小病灶" + 三个 headline）。"""
    small = AreaBand.SMALL.value
    summaries: list[ArmSummary] = []
    for c in cells:
        band_delta = {b: comp.delta for b, comp in c.by_band.items()}
        band_sig = {b: comp.significant for b, comp in c.by_band.items()}
        others = [d for b, d in band_delta.items() if b != small]
        small_conc = (
            band_sig.get(small, False)
            and small in band_delta
            and (not others or band_delta[small] > max(others))
        )
        summaries.append(ArmSummary(
            arm=str(c.value),
            overall_delta=c.overall.delta,
            overall_significant=c.overall.significant,
            gate_passed=(c.gate.passed if c.gate is not None else None),
            band_delta=band_delta,
            band_significant=band_sig,
            small_concentrated=small_conc,
        ))

    sig = [a for a in summaries if a.overall_significant]
    best_overall = max(sig, key=lambda a: a.overall_delta).arm if sig else None
    small_sig = [a for a in summaries if a.band_significant.get(small, False)]
    best_small = max(small_sig, key=lambda a: a.band_delta[small]).arm if small_sig else None
    concentrated = [a.arm for a in summaries if a.small_concentrated]
    return BReport(arms=summaries, best_overall=best_overall,
                   best_small=best_small, concentrated_arms=concentrated)


def b_ablation_report(
    base: AppConfig,
    score_fn: ScoreFn,
    bands: Sequence[AreaBand],
    *,
    arms: Sequence[str] = B_ARMS,
    gate: QualityGate | None = None,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int | None = 0,
) -> BReport:
    """一站式：跑 B 三臂消融 → 汇总成报告。"""
    cells = run_b_ablation(base, score_fn, bands, arms=arms, gate=gate,
                           n_resamples=n_resamples, level=level, seed=seed)
    return build_report(cells)


__all__ = [
    "FUSION_FLAG",
    "BASELINE_ARM",
    "B_ARMS",
    "ArmSummary",
    "BReport",
    "run_b_ablation",
    "build_report",
    "b_ablation_report",
]
