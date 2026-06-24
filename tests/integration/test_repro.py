"""可复现验证（T068）—— 固定 seed 复现评估记录，确定性管线两跑一致。

constitution III 的"可复现"铁律：同 seed + 同输入 → 同评估数字。覆盖 bootstrap/置换检验的随机性、
run_eval 的确定性、EvalRecord 落盘读回的一致性、消融两跑一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.config import AppConfig
from src.contracts.schemas import AreaBand
from src.eval import ablation as ab
from src.eval import record as rec
from src.eval import runner as run
from src.eval import stats as st

N = 16
BANDS = [AreaBand.SMALL] * 8 + [AreaBand.LARGE] * 8
_SMALL = np.array([b is AreaBand.SMALL for b in BANDS])


def _score_fn(cfg: AppConfig) -> np.ndarray:
    s = np.full(N, 0.5)
    if cfg.flags.get("fusion") == "add":
        s = s + 0.3
        s[_SMALL] += 0.1
    return s


def test_bootstrap_and_permutation_deterministic():
    a = [0.8, 0.9, 0.85, 0.95, 0.7, 0.88]
    b = [0.5, 0.55, 0.52, 0.6, 0.5, 0.58]
    ci1 = st.paired_bootstrap_delta_ci(a, b, n_resamples=500, seed=0)
    ci2 = st.paired_bootstrap_delta_ci(a, b, n_resamples=500, seed=0)
    assert (ci1.point, ci1.low, ci1.high) == (ci2.point, ci2.low, ci2.high)
    p1 = st.paired_permutation_test(a, b, n_resamples=500, seed=0)
    p2 = st.paired_permutation_test(a, b, n_resamples=500, seed=0)
    assert p1 == p2
    # 不同 seed 仍是确定的（各自可复现）
    assert st.bootstrap_ci(a, seed=1).point == st.bootstrap_ci(a, seed=1).point


def test_run_eval_reproducible():
    base = AppConfig()
    spec = run.EvalSpec("r", lambda c: c,
                        lambda c: {"metrics": {"mean": float(_score_fn(c).mean())}})
    r1 = run.run_eval(base, spec)
    r2 = run.run_eval(base, spec)
    assert r1.metrics == r2.metrics


def test_eval_record_roundtrip_identical(tmp_path):
    r = run.run_eval(
        AppConfig().with_overrides(flags={"fusion": "add"}),
        run.EvalSpec("add", lambda c: c,
                     lambda c: {"metrics": {"mean": float(_score_fn(c).mean())},
                                "stratified": {AreaBand.SMALL.value: {"mean": 0.9}}}),
    )
    path = rec.save_record(r, tmp_path)
    back = rec.load_record(path)
    assert rec.flatten(back) == rec.flatten(r)
    assert back.flags == r.flags


def test_ablation_reproducible():
    base = AppConfig()
    kw = dict(bands=BANDS, n_resamples=400, seed=0)
    c1 = ab.run_ablation(base, {"fusion": ["add"]}, _score_fn, **kw)
    c2 = ab.run_ablation(base, {"fusion": ["add"]}, _score_fn, **kw)
    assert c1[0].overall.delta == c2[0].overall.delta
    assert c1[0].overall.p_value == c2[0].overall.p_value
    assert c1[0].by_band["small"].delta == c2[0].by_band["small"].delta
