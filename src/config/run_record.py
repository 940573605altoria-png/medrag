"""run-record：把 `config → metrics` 连同模型/数据/代码版本一并留痕。

为什么需要（constitution III 可归因 + SC-005 可复现）：每次评估/训练 run 都必须能
回答"这个数字是哪套配置、哪版数据、哪个 commit 跑出来的"。评估 harness 的消融 runner
（`src/eval/runner.py`）将复用本模块写记录；骨架阶段先把 schema 与落盘逻辑定下来。

记录落 `runs_dir/<run_id>/run_record.json`，纯标准库实现，无重依赖。
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    """当前 commit 短哈希；非 git 环境或失败返回 None。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _git_dirty() -> bool | None:
    """工作区是否有未提交改动（脏 = 复现性风险，须留痕）。"""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


@dataclass
class RunRecord:
    """一次 run 的完整留痕。

    Attributes:
        run_id: 唯一标识（默认时间戳）。
        config: 本次 run 的配置快照（AppConfig 序列化或任意 dict）。
        metrics: 产出指标（评估完成后 `add_metrics` 填）。
        data_versions: 数据集版本/标识（如 a/b/c 的 HF/kaggle id + config + 切分哈希）。
        model_versions: 模型/checkpoint 版本（基座、LoRA、reranker、judge 等）。
        env: 运行环境（python/platform/git）。
        notes: 自由备注。
    """

    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    data_versions: dict[str, Any] = field(default_factory=dict)
    model_versions: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def create(
        cls,
        config: Any = None,
        *,
        run_id: str | None = None,
        notes: str = "",
    ) -> "RunRecord":
        """新建记录并自动采集环境/git 信息。

        `config` 可为 dataclass、pydantic model 或 dict；统一转 dict 存储。
        """
        return cls(
            run_id=run_id or time.strftime("%Y%m%d-%H%M%S"),
            config=_to_dict(config),
            env=_collect_env(),
            notes=notes,
        )

    def add_metrics(self, **metrics: Any) -> "RunRecord":
        self.metrics.update(metrics)
        return self

    def add_data_version(self, name: str, version: Any) -> "RunRecord":
        self.data_versions[name] = version
        return self

    def add_model_version(self, name: str, version: Any) -> "RunRecord":
        self.model_versions[name] = version
        return self

    def save(self, runs_dir: str | Path) -> Path:
        """落盘到 `runs_dir/<run_id>/run_record.json`，返回文件路径。"""
        out_dir = Path(runs_dir) / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "run_record.json"
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunRecord":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _collect_env() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    # pydantic v2
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    # dataclass
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    return {"repr": repr(config)}
