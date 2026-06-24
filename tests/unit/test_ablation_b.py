"""B 三臂消融报告单测（T055）—— 桩 score_fn、纯逻辑。

桩设定刻意还原 B 的故事：multiimg 整体最强（最贵基线），但 add(门控残差) 才把增益**集中**在
<2% 小病灶上——报告须把"整体最强"与"增益集中小病灶"区分开。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.config import AppConfig
from src.contracts.schemas import AreaBand
from src.eval import ablation_b as bab
from src.eval import runner as run

N = 20
BANDS = [AreaBand.SMALL] * 10 + [AreaBand.LARGE] * 10
_SMALL = np.array([b is AreaBand.SMALL for b in BANDS])


def _b_score_fn(cfg: AppConfig) -> np.ndarray:
    """global-only=0.5；add 整体+0.3 且小病灶再+0.1；concat+0.15；multiimg+0.5（整体最强）。"""
    s = np.full(N, 0.5)
    f = cfg.flags.get("fusion")
    if f == "add":
        s = s + 0.3
        s[_SMALL] += 0.1            # small Δ=0.4, large Δ=0.3 → 集中小病灶
    elif f == "concat":
        s = s + 0.15               # 各带均匀，不集中
    elif f == "multiimg":
        s = s + 0.5                # 整体最强，但各带均匀、不集中
    return s


def test_run_b_ablation_forces_global_only_baseline():
    # 即便传入的 base 已开 fusion=add，也应被重置为 off，使三臂都作为变体出现
    base = AppConfig().with_overrides(flags={"fusion": "add"})
    cells = bab.run_b_ablation(base, _b_score_fn, BANDS, n_resamples=300, seed=0)
    assert {c.value for c in cells} == {"add", "concat", "multiimg"}


def test_b_report_separates_best_overall_from_concentrated():
    report = bab.b_ablation_report(AppConfig(), _b_score_fn, BANDS,
                                   n_resamples=800, seed=0)
    by_arm = {a.arm: a for a in report.arms}
    # 整体最强 = multiimg(+0.5)；小病灶 delta 也是 multiimg 最大(+0.5)
    assert report.best_overall == "multiimg"
    assert report.best_small == "multiimg"
    # 但"增益集中在小病灶"的只有 add（small 0.4 > large 0.3）
    assert report.concentrated_arms == ["add"]
    assert by_arm["add"].small_concentrated
    assert not by_arm["concat"].small_concentrated      # 各带相等，不算集中
    assert not by_arm["multiimg"].small_concentrated


def test_b_report_band_deltas_values():
    report = bab.b_ablation_report(AppConfig(), _b_score_fn, BANDS,
                                   n_resamples=500, seed=0)
    add = next(a for a in report.arms if a.arm == "add")
    assert add.band_delta["small"] == pytest.approx(0.4)
    assert add.band_delta["large"] == pytest.approx(0.3)
    assert add.overall_delta == pytest.approx(0.35)


def test_b_report_with_gate_all_arms_pass():
    gate = run.QualityGate(n_resamples=500, seed=0)
    report = bab.b_ablation_report(AppConfig(), _b_score_fn, BANDS,
                                   gate=gate, n_resamples=500, seed=0)
    assert all(a.gate_passed for a in report.arms)   # 三臂相对基线都显著为正


def test_b_report_table_and_summary():
    report = bab.b_ablation_report(AppConfig(), _b_score_fn, BANDS,
                                   n_resamples=300, seed=0)
    rows = report.to_table()
    assert {r["arm"] for r in rows} == {"add", "concat", "multiimg"}
    assert "small/delta" in rows[0]
    lines = report.summary_lines()
    assert any("增益集中" in ln for ln in lines)
