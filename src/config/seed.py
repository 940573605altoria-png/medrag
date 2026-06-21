"""可复现：固定随机种子 + 确定性开关（constitution III 可复现铁律）。

骨架阶段只依赖标准库 `random`；torch/numpy 等重依赖按可用性**惰性 seed**，
未安装则静默跳过——这样无 GPU 环境也能跑桩管线并保证桩输出确定。
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """统一播种所有可用的随机源，返回实际使用的 seed。

    Args:
        seed: 随机种子。
        deterministic: 为 True 时尽量开启确定性算法（torch 的 deterministic 模式、
            cudnn 关闭 benchmark）。骨架阶段无 torch 时此参数对标准库无影响。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    _seed_numpy(seed)
    _seed_torch(seed, deterministic=deterministic)
    return seed


def _seed_numpy(seed: int) -> None:
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return
    np.random.seed(seed)


def _seed_torch(seed: int, *, deterministic: bool) -> None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 不让任何上游修改静默吞掉确定性失败
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001 - 老版本 torch 无此 API
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
