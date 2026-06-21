"""模型层单测 —— CPU 可跑，不下载 30B 权重、不需要 torch。

只验证加载器的"无重依赖也能 import + 配置逻辑正确 + 缺依赖给清晰错误"，
真正的加载/推理在 AutoDL 上由 `scripts/smoke_gpu.py --with-model` 覆盖。
"""

from __future__ import annotations

import importlib

import pytest

from src.models import qwen3vl


def test_module_imports_without_torch():
    """守卫导入：本机无 torch/transformers 也能 import 模块（供契约/单测引用）。"""
    importlib.reload(qwen3vl)
    assert qwen3vl.DEFAULT_MODEL_ID == "Qwen/Qwen3-VL-30B-A3B-Instruct"


def test_resolve_model_id_precedence(monkeypatch):
    # 显式参数优先
    assert qwen3vl.resolve_model_id("foo/bar") == "foo/bar"
    # 环境变量次之
    monkeypatch.setenv("MEDRAG_BASE_MODEL", "env/model")
    assert qwen3vl.resolve_model_id() == "env/model"
    # 默认兜底
    monkeypatch.delenv("MEDRAG_BASE_MODEL", raising=False)
    assert qwen3vl.resolve_model_id() == qwen3vl.DEFAULT_MODEL_ID


def test_config_model_kwargs_defaults():
    cfg = qwen3vl.Qwen3VLConfig(model_id="x/y")
    kw = cfg.model_kwargs()
    assert kw["device_map"] == "auto"
    assert kw["dtype"] == "auto"
    assert kw["trust_remote_code"] is True
    # 默认无量化、无 attn_implementation 注入
    assert "quantization_config" not in kw
    assert "attn_implementation" not in kw


def test_config_attn_impl_injected():
    cfg = qwen3vl.Qwen3VLConfig(attn_implementation="flash_attention_2")
    assert cfg.model_kwargs()["attn_implementation"] == "flash_attention_2"


def test_load_base_without_transformers_raises_clear_error():
    """无 transformers 时 load_base 报清晰 RuntimeError，而非晦涩 ImportError。"""
    pytest.importorskip  # noqa: B018 - 占位，确保 pytest 可用
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="transformers"):
            qwen3vl.load_base()
    else:
        pytest.skip("本环境已装 transformers，跳过缺依赖分支")
