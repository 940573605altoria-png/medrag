"""US3 消融集成测试（T057）—— 单开关改动 → 分层 + 显著性 delta，串通整套 harness。

把 runner(T016) + 消融矩阵(T054) + 质量门(T056) + B 三臂报告(T055) + 记录持久化(record) 串成一次
完整"评估 campaign"，端到端验证 constitution III：一次只改一个变量、相对基线、分层、显著性。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.config import AppConfig
from src.contracts.schemas import AreaBand
from src.eval import ablation as ab
from src.eval import ablation_b as bab
from src.eval import record as rec
from src.eval import runner as run
from src.eval import stats as st
from src.eval.runner import GateConfig, QualityGate

N = 20
BANDS = [AreaBand.SMALL] * 10 + [AreaBand.LARGE] * 10
_SMALL = np.array([b is AreaBand.SMALL for b in BANDS])


def _score_fn(cfg: AppConfig) -> np.ndarray:
    """确定性逐样本得分：fusion=add 整体+0.3、小病灶再+0.1；concat+0.15；multiimg+0.5。"""
    s = np.full(N, 0.5)
    f = cfg.flags.get("fusion")
    if f == "add":
        s = s + 0.3
        s[_SMALL] += 0.1
    elif f == "concat":
        s = s + 0.15
    elif f == "multiimg":
        s = s + 0.5
    return s


def test_b_ablation_campaign_with_gate():
    gate = QualityGate(GateConfig(min_delta=0.05), n_resamples=800, seed=0)
    cells = bab.run_b_ablation(AppConfig(), _score_fn, BANDS, gate=gate,
                               n_resamples=800, seed=0)
    report = bab.build_report(cells)
    # 一次一变量：每个变体只改了 fusion
    assert {c.flag for c in cells} == {"fusion"}
    # 增益集中在小病灶的只有 add；三臂都显著为正 → 无被门拦
    assert report.concentrated_arms == ["add"]
    assert ab.blocked(cells) == []


def test_single_change_stratified_delta_and_significance():
    base = AppConfig()
    add = base.with_overrides(flags={"fusion": "add"})
    base_scores, add_scores = _score_fn(base), _score_fn(add)
    # 分层 delta：小病灶(0.4) > 大病灶(0.3)
    small_delta = (add_scores[_SMALL] - base_scores[_SMALL]).mean()
    large_delta = (add_scores[~_SMALL] - base_scores[~_SMALL]).mean()
    assert small_delta == pytest.approx(0.4) and large_delta == pytest.approx(0.3)
    # 配对显著性
    ci = st.paired_bootstrap_delta_ci(add_scores, base_scores, n_resamples=800, seed=0)
    assert st.is_significant(ci)


def test_runner_records_persist_and_delta(tmp_path):
    base = AppConfig()

    def metric_fn(cfg):
        scores = _score_fn(cfg)
        return {
            "metrics": {"mean": float(scores.mean())},
            "stratified": {
                AreaBand.SMALL.value: {"mean": float(scores[_SMALL].mean())},
                AreaBand.LARGE.value: {"mean": float(scores[~_SMALL].mean())},
            },
        }

    rec_base = run.run_eval(base, run.EvalSpec("baseline", lambda c: c, metric_fn))
    rec_add = run.run_eval(base.with_overrides(flags={"fusion": "add"}),
                           run.EvalSpec("fusion_add", lambda c: c, metric_fn),
                           baseline_id="baseline")
    rec.save_record(rec_base, tmp_path)
    rec.save_record(rec_add, tmp_path)

    loaded = {r.run_id: r for r in rec.load_records(tmp_path)}
    d = rec.delta_row(loaded["fusion_add"], loaded["baseline"])
    assert d["mean"] == pytest.approx(0.35)                       # 整体增益
    assert d[f"{AreaBand.SMALL.value}/mean"] == pytest.approx(0.4)  # 小病灶增益更大
