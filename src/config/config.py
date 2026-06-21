"""全局配置加载。

设计目标：**配置驱动**（评估 harness 的消融 runner 靠它把"一次只改一个变量"
落成开关）。骨架阶段提供一个最小但可扩展的 `AppConfig`：
- 路径布局（本机 vs AutoDL `/root/autodl-tmp` 数据盘）；
- 可复现 seed；
- 组件开关 `flags`（B/C/RAG/reranker/dedup/coreset… 后续逐个接入，骨架默认 stub）。

加载优先级：默认值 < 配置文件（json/yaml）< 环境变量 `MEDRAG_*`。
yaml 为可选依赖；未安装时仅支持 json，不报错（骨架不强依赖 yaml）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from src.config.seed import DEFAULT_SEED

# AutoDL 数据盘约定（plan.md 部署节）：权重/向量库/影像/产物落此处。
_AUTODL_DATA_ROOT = Path("/root/autodl-tmp")


def _default_data_root() -> Path:
    """AutoDL 上用数据盘，本机用仓库内 ./data_root。"""
    if _AUTODL_DATA_ROOT.exists():
        return _AUTODL_DATA_ROOT
    return Path(__file__).resolve().parents[2] / "data_root"


@dataclass(frozen=True)
class Paths:
    """路径布局，对本机/AutoDL 自适应。"""

    data_root: Path = field(default_factory=_default_data_root)

    @property
    def chroma_dir(self) -> Path:
        return self.data_root / "chroma"

    @property
    def weights_dir(self) -> Path:
        return self.data_root / "weights"

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"


# 组件开关：骨架阶段全部走 stub（实现为 False / "stub"），真实组件逐批替换时打开。
# 评估 harness 的消融矩阵直接消费这张表（constitution III：一次一变量）。
_DEFAULT_FLAGS: dict[str, Any] = {
    "detect": "stub",      # stub | lochead   (C 定位头)
    "retrieve": "stub",    # stub | cascade   (RAG 级联)
    "report": "stub",      # stub | qwen3vl   (报告生成)
    "fusion": "off",       # off | add | concat | multiimg  (B 三臂)
    "reranker": False,
    "dedup": False,
    "coreset": False,
}


@dataclass(frozen=True)
class AppConfig:
    """应用级配置（不可变；用 `with_overrides` 派生变体做消融）。"""

    seed: int = DEFAULT_SEED
    deterministic: bool = True
    paths: Paths = field(default_factory=Paths)
    flags: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_FLAGS))

    def with_overrides(self, **changes: Any) -> "AppConfig":
        """派生一个改了若干字段的新配置（消融 runner 用）。

        `flags=` 会与现有 flags 合并而非整体替换，避免漏掉默认开关。
        """
        if "flags" in changes:
            merged = {**self.flags, **(changes.pop("flags") or {})}
            changes["flags"] = merged
        return replace(self, **changes)


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """加载配置：默认值 < 文件 < 环境变量。

    Args:
        path: json 或 yaml 配置文件路径；None 时仅用默认值 + 环境变量。
    """
    cfg = AppConfig()

    if path is not None:
        cfg = _apply_mapping(cfg, _read_file(Path(path)))

    cfg = _apply_env(cfg)
    return cfg


def _read_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:  # 骨架不强依赖 yaml
            raise RuntimeError(
                "读取 yaml 配置需要 pyyaml；请 `pip install pyyaml` 或改用 json。"
            ) from exc
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def _apply_mapping(cfg: AppConfig, data: dict[str, Any]) -> AppConfig:
    changes: dict[str, Any] = {}
    if "seed" in data:
        changes["seed"] = int(data["seed"])
    if "deterministic" in data:
        changes["deterministic"] = bool(data["deterministic"])
    if "data_root" in data:
        changes["paths"] = Paths(data_root=Path(data["data_root"]))
    if "flags" in data:
        changes["flags"] = dict(data["flags"])
    return cfg.with_overrides(**changes)


def _apply_env(cfg: AppConfig) -> AppConfig:
    """环境变量覆盖：MEDRAG_SEED / MEDRAG_DATA_ROOT。"""
    changes: dict[str, Any] = {}
    if (raw := os.environ.get("MEDRAG_SEED")) is not None:
        changes["seed"] = int(raw)
    if (raw := os.environ.get("MEDRAG_DATA_ROOT")) is not None:
        changes["paths"] = Paths(data_root=Path(raw))
    return cfg.with_overrides(**changes) if changes else cfg
