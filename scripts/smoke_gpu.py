"""GPU 冒烟测试（分级）—— 验证 AutoDL 链路通。

为什么分级：下 30B 权重又慢又占盘，不该为"先确认 GPU+我们的管线能跑"而被它阻塞。
所以分两级，按需推进：

  L0（默认，无需权重）: 打印 torch/CUDA/显存信息 + 在本机/实例上跑通骨架 pipeline。
                        几秒钟，先确认环境 + 我们的代码在这台机器上 OK。
  L1（--with-model）   : 加载 Qwen3-VL 基座 + 对合成图做一次推理。需已下权重 + 足够显存。

用法（在 AutoDL 实例）:
    python scripts/smoke_gpu.py                 # 仅 L0
    python scripts/smoke_gpu.py --with-model    # L0 + L1（默认 bf16）
    python scripts/smoke_gpu.py --with-model --quant 4bit
    MEDRAG_BASE_MODEL=/root/autodl-tmp/weights/Qwen3-VL-30B-A3B-Instruct \
        python scripts/smoke_gpu.py --with-model
"""

from __future__ import annotations

import argparse
import sys

from src.config.seed import seed_everything
from src.contracts.schemas import ReportResult
from src.serve.pipeline import Pipeline


def level0() -> None:
    """环境信息 + 骨架管线跑通（不碰大模型）。"""
    print("=== L0: 环境 + 骨架 ===")
    try:
        import torch

        print(f"torch        : {torch.__version__}")
        print(f"cuda built   : {torch.version.cuda}")
        print(f"cuda avail   : {torch.cuda.is_available()}")
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"gpu count    : {n}")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu[{i}]     : {p.name}  {p.total_memory/1e9:.1f} GB")
    except ImportError:
        print("torch        : (未安装；L0 仅跑骨架)")

    seed_everything(42)
    report = Pipeline().generate_report("smoke-ct-001")
    assert isinstance(report, ReportResult)
    assert report.findings and not report.abstain, "骨架管线应产出有据报告"
    print(f"pipeline     : OK（{len(report.findings)} findings, grounded）")
    print("L0 PASS\n")


def level1(quant: str | None, max_new_tokens: int) -> None:
    """加载基座 + 一次推理。"""
    print("=== L1: Qwen3-VL 基座加载 + 推理 ===")
    import torch  # noqa: F401  (确保有 torch 才进 L1)
    from PIL import Image
    import numpy as np

    from src.models.qwen3vl import Qwen3VLConfig, load_base, quick_infer, resolve_model_id

    model_id = resolve_model_id()
    print(f"model_id     : {model_id}  (quant={quant})")

    cfg = Qwen3VLConfig(quant=quant)
    print("loading ...（首次会下载/读取权重，耐心等）")
    model, processor = load_base(cfg)
    print(f"loaded       : {type(model).__name__} on {getattr(model, 'device', '?')}")

    # 合成一张可辨认的渐变灰阶图当假 CT
    arr = (np.tile(np.linspace(0, 255, 224, dtype=np.uint8), (224, 1)))
    img = Image.fromarray(arr).convert("RGB")
    out = quick_infer(model, processor, img, "Briefly describe this image.",
                      max_new_tokens=max_new_tokens)
    print(f"infer output : {out!r}")
    assert out, "推理应返回非空文本"
    if "torch" in sys.modules and torch.cuda.is_available():
        print(f"peak VRAM    : {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print("L1 PASS\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="medrag GPU 冒烟测试（分级）")
    ap.add_argument("--with-model", action="store_true", help="加跑 L1（加载基座+推理）")
    ap.add_argument("--quant", choices=["fp8", "4bit"], default=None, help="L1 量化模式")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    level0()
    if args.with_model:
        level1(args.quant, args.max_new_tokens)
    print("ALL SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
