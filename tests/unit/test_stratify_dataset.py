"""面积分层 + held-out 加载单测（T010/T015）—— 纯逻辑本地测。"""

from __future__ import annotations

import json

import pytest

from src.contracts.schemas import AreaBand, BBox, CTSample, ROI
from src.data import stratify as sf
from src.eval import dataset as ds


def _roi(frac):
    return ROI(bbox=BBox(x1=0, y1=0, x2=10, y2=10), area_fraction=frac)


def _case(cid, frac, in_eval=True):
    return CTSample(case_id=cid, image_path=f"{cid}.png", rois=[_roi(frac)],
                    in_eval_set=in_eval)


# ── T010 分层工具 ─────────────────────────────────────────────────

def test_band_of_thresholds():
    assert sf.band_of(0.01) is AreaBand.SMALL
    assert sf.band_of(0.03) is AreaBand.MEDIUM
    assert sf.band_of(0.10) is AreaBand.LARGE


def test_group_and_distribution():
    items = [0.01, 0.015, 0.03, 0.2]
    dist = sf.band_distribution(items, fraction_fn=lambda x: x)
    assert dist == {"small": 2, "medium": 1, "large": 1}


# ── T015 held-out 加载 + 守卫 ─────────────────────────────────────

def test_heldout_stratify_and_ids():
    hs = ds.from_samples([_case("a", 0.01), _case("b", 0.03), _case("c", 0.2)])
    assert hs.band_counts() == {"small": 1, "medium": 1, "large": 1}
    assert hs.ids == {"a", "b", "c"}


def test_from_samples_filters_non_eval():
    hs = ds.from_samples([_case("a", 0.01, in_eval=True), _case("b", 0.01, in_eval=False)])
    assert hs.ids == {"a"}


def test_never_touched_guard():
    hs = ds.from_samples([_case("a", 0.01), _case("b", 0.03)])
    hs.assert_never_touched(["x", "y"])                 # 不相交 → 不抛
    with pytest.raises(ValueError):
        hs.assert_never_touched(["a", "z"])             # 评估集泄露 → 抛


def test_load_heldout_jsonl(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text(json.dumps({"id": "c1", "answer": "报告"}) + "\n", encoding="utf-8")
    hs = ds.load_heldout(p)
    assert hs.ids == {"c1"} and all(s.in_eval_set for s in hs.samples)
