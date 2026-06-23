"""B 双路融合（T030）—— 全局图 + ROI 放缩图在视觉 token 层的融合，含三臂消融。

创新 B 的痛点：通用 VLM 把小病灶的特征在全局下采样里稀释掉了。对策：除全局图外，再把
ROI（C 定位出的病灶区）放缩成一张图过同一视觉塔，得到"放大特征"，在 **merger 之后的视觉
token 层**与全局 token 融合，再喂给 LLM。

**三臂消融**（一次只改这一个开关，对照评估，constitution III）：
- **add（门控残差，我们的主张）**：`V = V_g + tanh(α)·g(V_roi)`，α 初始 0 → tanh(0)=0 →
  初始化时融合**恒等**（输出==全局 token），**不破坏基座预训练分布**，再让模型学着把 ROI
  特征加进来。这是 B 的核心设计。
- **concat（拼接）**：把两组 token 沿序列拼接，交给 LLM 自己注意。更简单的对照。
- **multi_image（多图输入，最强基线）**：把全局图与 ROI 当两张图喂模型原生多图通路；token
  层就是两组拼接。作为"不加任何新机制"的强基线。

本模块是可独立替换的组件，操作 token 张量，与基座建模文件解耦 → 合成张量即可单测。接进基座
前向（merger 后注入）是 AutoDL 上 vendor/patch `modeling_qwen3_vl.py` 的集成步骤（T012）。

守卫导入 torch：无 torch 也能 import 本模块（配置/工厂），构建/前向才需 torch。
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 环境相关
    _HAS_TORCH = False


FUSION_MODES = ("add", "concat", "multi_image")


@dataclass
class FusionConfig:
    """融合配置。`mode` 即三臂开关，消融时只改这一项。"""

    mode: str = "add"          # add | concat | multi_image
    dim: int = 1152            # 视觉 token 特征维（按基座 vision 隐藏维调）
    alpha_init: float = 0.0    # 门控初值；0 → tanh(0)=0 → 初始恒等（等价预训练分布）


if _HAS_TORCH:

    class DualPathFusion(nn.Module):
        """全局 token + ROI token → 融合 token。三臂由 `FusionConfig.mode` 选择。"""

        def __init__(self, config: FusionConfig | None = None):
            super().__init__()
            cfg = config or FusionConfig()
            if cfg.mode not in FUSION_MODES:
                raise ValueError(f"未知 fusion mode: {cfg.mode}，应属 {FUSION_MODES}")
            self.mode = cfg.mode
            self.dim = cfg.dim
            if self.mode == "add":
                # g(·)：对 ROI token 的可学习投影；α：门控标量（初始 0 → 残差为 0）
                self.proj = nn.Linear(cfg.dim, cfg.dim)
                self.alpha = nn.Parameter(torch.tensor(float(cfg.alpha_init)))

        def forward(self, global_tokens, roi_tokens):
            """global_tokens (B,Ng,D) + roi_tokens (B,Nr,D) → 融合 token。

            - add：要求 Ng==Nr（逐 token 残差），输出 (B,Ng,D)；
            - concat / multi_image：沿序列拼接，输出 (B,Ng+Nr,D)。
            """
            if self.mode == "add":
                if global_tokens.shape != roi_tokens.shape:
                    raise ValueError(
                        f"add 模式要求全局/ROI token 同形，得 {tuple(global_tokens.shape)} "
                        f"vs {tuple(roi_tokens.shape)}"
                    )
                gated = torch.tanh(self.alpha) * self.proj(roi_tokens)
                return global_tokens + gated
            # concat / multi_image：token 层都是序列拼接（multi_image 的"多图"语义在上游
            # 输入构造端体现，到 token 层等价拼接，作最强基线）
            return torch.cat([global_tokens, roi_tokens], dim=1)


def build_fusion(config: FusionConfig | None = None):
    """构建融合模块；无 torch 时给清晰错误。"""
    if not _HAS_TORCH:  # pragma: no cover - 环境相关
        raise RuntimeError("构建 DualPathFusion 需要 torch")
    return DualPathFusion(config)


__all__ = ["FUSION_MODES", "FusionConfig", "build_fusion"]
