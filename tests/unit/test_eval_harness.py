"""评估 harness 单测（T016 runner/record + T054 ablation）—— 纯逻辑、无模型。

用确定性的 score_fn（得分只由 config.flags 决定）驱动消融,手算 delta/分层,验证"一次一变量 +
相对基线 + 小病灶分层 + 配对显著性"的闭环口径。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.config import AppConfig
from src.contracts.schemas import AreaBand, EvalRecord
from src.eval import ablation as ab
from src.eval import record as rec
from src.eval import runner as run


# ── T016 record：落盘/读取/拍平 ───────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    r = EvalRecord(run_id="r1", flags={"fusion": "add"},
                   metrics={"acc": 0.8}, stratified={"small": {"acc": 0.7}})
    path = rec.save_record(r, tmp_path)
    back = rec.load_record(path)
    assert back.run_id == "r1"
    assert back.metrics["acc"] == pytest.approx(0.8)
    assert back.stratified["small"]["acc"] == pytest.approx(0.7)


def test_flatten_and_delta_row():
    a = EvalRecord(run_id="a", metrics={"acc": 0.8}, stratified={"small": {"acc": 0.7}})
    b = EvalRecord(run_id="b", metrics={"acc": 0.5}, stratified={"small": {"acc": 0.4}})
    flat = rec.flatten(a)
    assert flat["acc"] == 0.8 and flat["small/acc"] == 0.7
    d = rec.delta_row(a, b)
    assert d["acc"] == pytest.approx(0.3)
    assert d["small/acc"] == pytest.approx(0.3)


def test_load_records_sorted(tmp_path):
    rec.save_record(EvalRecord(run_id="b"), tmp_path)
    rec.save_record(EvalRecord(run_id="a"), tmp_path)
    ids = [r.run_id for r in rec.load_records(tmp_path)]
    assert ids == ["a", "b"]


# ── T016 runner：config 驱动 → EvalRecord ─────────────────────────

def test_run_eval_captures_flags_and_metrics():
    cfg = AppConfig().with_overrides(flags={"fusion": "add"})
    spec = run.EvalSpec(
        run_id="rx",
        predict_fn=lambda c: c.flags["fusion"],          # 中间物 = 开关值
        metric_fn=lambda out: {"metrics": {"acc": 0.9},
                               "stratified": {"small": {"acc": 0.6}}},
    )
    record = run.run_eval(cfg, spec, baseline_id="base")
    assert record.flags["fusion"] == "add"
    assert record.metrics["acc"] == pytest.approx(0.9)
    assert record.stratified["small"]["acc"] == pytest.approx(0.6)
    assert record.baseline_id == "base"


def test_run_eval_requires_metrics_key():
    spec = run.EvalSpec(run_id="r", predict_fn=lambda c: None, metric_fn=lambda o: {})
    with pytest.raises(ValueError):
        run.run_eval(AppConfig(), spec)


# ── T054 ablation：一次一变量 + 分层 + 显著性 ─────────────────────

N = 20
BANDS = [AreaBand.SMALL] * 10 + [AreaBand.LARGE] * 10


def _score_fn(cfg: AppConfig) -> np.ndarray:
    """确定性逐样本得分：fusion=add 整体 +0.3、小病灶再 +0.1；concat +0.15；其余基线 0.5。"""
    s = np.full(N, 0.5)
    f = cfg.flags.get("fusion")
    if f == "add":
        s = s + 0.3
        for i in range(N):
            if BANDS[i] is AreaBand.SMALL:
                s[i] += 0.1
    elif f == "concat":
        s = s + 0.15
    return s


def test_build_variants_one_change_and_skip_equal():
    base = AppConfig()  # fusion 默认 "off"
    variants = ab.build_variants(base, {"fusion": ["off", "add", "concat"]})
    names = [v[0] for v in variants]
    assert "fusion=off" not in names          # 与基线同值 → 跳过
    assert names == ["fusion=add", "fusion=concat"]
    # 每个变体只改了 fusion 这一个开关
    for _, flag, _, cfg in variants:
        changed = [k for k in cfg.flags if cfg.flags[k] != base.flags[k]]
        assert changed == ["fusion"]


def test_ablation_gain_concentrated_in_small_lesions():
    base = AppConfig()
    cells = ab.run_ablation(base, {"fusion": ["add", "concat"]}, _score_fn,
                            bands=BANDS, n_resamples=1000, seed=0)
    by_name = {c.name: c for c in cells}
    add = by_name["fusion=add"]
    # 整体 delta = (0.4*10 + 0.3*10)/20 = 0.35，显著
    assert add.overall.delta == pytest.approx(0.35, abs=1e-9)
    assert add.overall.significant
    # 增益集中在小病灶：SMALL delta(0.4) > LARGE delta(0.3)
    assert add.by_band["small"].delta == pytest.approx(0.4)
    assert add.by_band["large"].delta == pytest.approx(0.3)
    assert add.by_band["small"].delta > add.by_band["large"].delta
    # concat 整体 +0.15，无分层差异
    assert by_name["fusion=concat"].overall.delta == pytest.approx(0.15)


def test_ablation_noop_variant_not_significant():
    base = AppConfig()  # reranker 默认 False；score_fn 不看 reranker
    cells = ab.run_ablation(base, {"reranker": [True]}, _score_fn,
                            n_resamples=500, seed=0)
    cell = cells[0]
    assert cell.overall.delta == pytest.approx(0.0)
    assert not cell.overall.significant
    assert cell.overall.p_value == pytest.approx(1.0)


def test_ablation_bands_length_mismatch_raises():
    with pytest.raises(ValueError):
        ab.run_ablation(AppConfig(), {"fusion": ["add"]}, _score_fn,
                        bands=[AreaBand.SMALL])  # 长度 1 ≠ N


def test_to_table_shape():
    cells = ab.run_ablation(AppConfig(), {"fusion": ["add"]}, _score_fn,
                            bands=BANDS, n_resamples=300, seed=0)
    rows = ab.to_table(cells)
    assert rows[0]["variant"] == "fusion=add"
    assert "small/delta" in rows[0] and "p_value" in rows[0]


# ── T056 质量门：增益不显著 → 阻断 ────────────────────────────────

def test_gate_passes_significant_gain():
    gate = run.QualityGate(n_resamples=500, seed=0)
    res = gate.check_scores([0.5] * 20, [0.85] * 20)  # 配对差恒 +0.35
    assert res.passed
    assert res.delta == pytest.approx(0.35)


def test_gate_blocks_noop():
    gate = run.QualityGate(n_resamples=500, seed=0)
    res = gate.check_scores([0.5] * 20, [0.5] * 20)   # 无差异
    assert not res.passed
    assert "CI" in res.reason and res.p_value == pytest.approx(1.0)


def test_gate_blocks_significant_but_tiny_delta():
    # 统计显著但效应量太小：min_delta 挡掉
    gate = run.QualityGate(config=run.GateConfig(min_delta=0.05), n_resamples=500, seed=0)
    res = gate.check_scores([0.5] * 20, [0.52] * 20)  # 恒 +0.02，显著但 <0.05
    assert not res.passed
    assert "最小阈值" in res.reason


def test_gate_lower_is_better_direction():
    # FP/图 越低越好：变体更低 → 过门
    gate = run.QualityGate(config=run.GateConfig(higher_is_better=False),
                           n_resamples=500, seed=0)
    res = gate.check_scores([2.0] * 10, [1.0] * 10)   # delta=-1（更优）
    assert res.passed


def test_gate_assert_pass_raises_on_block():
    gate = run.QualityGate(n_resamples=300, seed=0)
    blocked = gate.check_scores([0.5] * 10, [0.5] * 10)
    with pytest.raises(run.QualityGateError):
        gate.assert_pass(blocked)


def test_ablation_with_gate_attaches_verdict():
    gate = run.QualityGate(n_resamples=500, seed=0)
    cells = ab.run_ablation(AppConfig(), {"fusion": ["add", "concat"]}, _score_fn,
                            bands=BANDS, gate=gate, n_resamples=500, seed=0)
    by_name = {c.name: c for c in cells}
    assert by_name["fusion=add"].gate is not None
    assert by_name["fusion=add"].gate.passed       # +0.35 显著 → 过门
    assert ab.blocked(cells) == []                 # 两臂都是真增益


def test_ablation_gate_blocks_noop_variant():
    gate = run.QualityGate(n_resamples=400, seed=0)
    cells = ab.run_ablation(AppConfig(), {"reranker": [True]}, _score_fn,
                            gate=gate, n_resamples=400, seed=0)
    assert [c.name for c in ab.blocked(cells)] == ["reranker=True"]
