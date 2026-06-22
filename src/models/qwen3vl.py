"""Qwen3-VL 基座封装/加载器（T011）。

基座 = `Qwen/Qwen3-VL-30B-A3B-Instruct`（MoE，30B 总参/约 3B 激活）。
加载类是 MoE 专用的 `Qwen3VLMoeForConditionalGeneration` + `AutoProcessor`。

设计：
- **守卫导入**：torch/transformers 仅在真正加载时才 import。这样本文件在无 GPU/无
  这些重依赖的本机也能 `import`（供单测/契约引用），只有调用 `load_base` 才需环境。
- **对显存自适应**：`quant` 支持 None(bf16) / "fp8" / "4bit"（bitsandbytes），由探测到的
  VRAM 决定；`device_map="auto"` 多卡自动分片（呼应"对卡数自适应"）。
- US1 再扩：LoRA 挂载、共享 ViT 取特征（B 融合注入点）——此处先留接口位，冒烟只需加载+推理。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# 显存紧张时的官方量化镜像（runbook 里按 VRAM 切换）：
FP8_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
AWQ_MODEL_ID = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"


def resolve_model_id(model_id: str | None = None) -> str:
    """模型 id 解析：显式参数 > 环境变量 MEDRAG_BASE_MODEL > 默认。"""
    return model_id or os.environ.get("MEDRAG_BASE_MODEL") or DEFAULT_MODEL_ID


@dataclass
class Qwen3VLConfig:
    """基座加载配置。"""

    model_id: str = field(default_factory=resolve_model_id)
    dtype: str = "auto"                 # auto -> bf16/fp16 by hardware
    device_map: str = "auto"            # 多卡自动分片；单卡也可
    quant: str | None = None            # None | "fp8" | "4bit"
    attn_implementation: str | None = None  # 如 "flash_attention_2"（装了再开）
    trust_remote_code: bool = True
    max_new_tokens: int = 64

    def model_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "dtype": self.dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.attn_implementation:
            kw["attn_implementation"] = self.attn_implementation
        if self.quant == "4bit":
            kw["quantization_config"] = _bnb_4bit_config()
        return kw


def _bnb_4bit_config():
    """4bit 量化配置（bitsandbytes）。仅在 quant='4bit' 时触发，缺包则报清晰错误。"""
    try:
        import torch  # noqa: PLC0415
        from transformers import BitsAndBytesConfig  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError("4bit 量化需要 transformers + bitsandbytes + torch") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _load_model_class():
    """选加载类。

    **优先 `AutoModelForImageTextToText`**：它按 checkpoint 的 config 自动派发到正确类——
    既能加载 30B MoE（`Qwen3VLMoeForConditionalGeneration`），也能加载 4B/8B dense
    （`Qwen3VLForConditionalGeneration`），避免硬选 MoE 类去加载 dense 权重而 mismatch。
    Auto 类不可用（老 transformers）时再回退到显式类。
    """
    try:
        from transformers import AutoModelForImageTextToText  # noqa: PLC0415

        return AutoModelForImageTextToText
    except ImportError:
        pass

    import transformers  # noqa: PLC0415

    for name in ("Qwen3VLMoeForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "transformers 太旧：既无 AutoModelForImageTextToText 也无 Qwen3VL* 类，请升级"
    )


def load_base(config: Qwen3VLConfig | None = None) -> tuple[Any, Any]:
    """加载基座模型 + processor，返回 (model, processor)。

    需要 GPU 环境（transformers/torch + 已下载权重）。权重路径可用本地目录
    （ModelScope 下载到 /root/autodl-tmp/weights/... 后把 model_id 设为该路径）。
    """
    config = config or Qwen3VLConfig()
    try:
        from transformers import AutoProcessor  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "加载基座需要 transformers（AutoDL：scripts/autodl_setup.sh 安装）"
        ) from exc

    model_cls = _load_model_class()
    model = model_cls.from_pretrained(config.model_id, **config.model_kwargs())
    processor = AutoProcessor.from_pretrained(
        config.model_id, trust_remote_code=config.trust_remote_code
    )
    model.eval()
    return model, processor


def quick_infer(model: Any, processor: Any, image: Any, prompt: str, *, max_new_tokens: int = 64) -> str:
    """对单图 + 文本做一次推理，返回解码文本（冒烟用）。

    `image` 接受 PIL.Image。走 processor 的 chat 模板，与训练对话格式一致。
    """
    import torch  # noqa: PLC0415

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # 仅解码新生成部分
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
