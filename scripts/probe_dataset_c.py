"""探查数据集 c（MedTrinity-25M / 25M_demo）的标注形态 —— US1 前置验证。

为什么要先跑这个：创新 C 的设计假设 CT 图上画了**绿框**（cv2.inRange 提框 → 高斯热图标签
→ inpaint 抹框）。但 MedTrinity-25M 很可能给的是 **ROI bbox 坐标 + 多粒度文本**，而非像素
上的框。两种形态决定 C 的提框入口（src/data/ct_box.py, T024）怎么写：
  - 像素绿框  → 保留 cv2.inRange 检测；
  - bbox 坐标 → 直接读坐标，inpaint/热图/防泄露逻辑（T025/T026）不变。

本脚本三件事一起验：
  ① 打印样本字段结构（key/类型/示例）；
  ② 在文本字段里搜坐标样式（bbox / [x,y,w,h] / "located"）；
  ③ 把图像扫一遍找强绿色像素（疑似画上去的框），统计占比 + 绿色包围盒，并存前几张图供肉眼看。

用法（AutoDL bash，conda activate medrag）：
    python scripts/probe_dataset_c.py                 # 默认 streaming 拉 5 条，不全量下载
    python scripts/probe_dataset_c.py --n 10
    python scripts/probe_dataset_c.py --from-disk /root/autodl-tmp/raw/c_ct   # 若已 save_to_disk
    python scripts/probe_dataset_c.py --out-dir /root/autodl-tmp/probe_c      # 存图目录
"""

from __future__ import annotations

import argparse
import os
import re

# 与 smoke_gpu 一致：走镜像 + 放大超时（streaming 也会读元数据）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

DATASET_ID = "UCSC-VLAA/MedTrinity-25M"
CONFIG = "25M_demo"

# 文本里坐标的常见样式：bbox 关键字、[x, y, w, h] / (x1,y1,x2,y2)、"located in the ... region"
COORD_PATTERNS = [
    re.compile(r"\bbbox\b", re.I),
    re.compile(r"\bbounding\s*box\b", re.I),
    re.compile(r"[\[\(]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*[\]\)]"),
    re.compile(r"\blocat(?:ed|ion)\b", re.I),
    re.compile(r"\bregion\b", re.I),
    re.compile(r"\bcoordinate", re.I),
]


def _load_samples(args):
    """优先从已存盘读取；否则 streaming 拉前 n 条（不触发全量下载）。"""
    from datasets import load_dataset, load_from_disk

    if args.from_disk:
        print(f"[load] from_disk: {args.from_disk}")
        ds = load_from_disk(args.from_disk)
        split = next(iter(ds.keys())) if hasattr(ds, "keys") else None
        ds = ds[split] if split else ds
        return [ds[i] for i in range(min(args.n, len(ds)))]

    print(f"[load] streaming: {DATASET_ID} ({CONFIG})，拉前 {args.n} 条")
    ds = load_dataset(DATASET_ID, CONFIG, split="train", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        out.append(ex)
    return out


def _describe_value(v):
    """对单个字段值给出紧凑描述（类型 + 形状/长度 + 截断示例）。"""
    from PIL import Image

    if isinstance(v, Image.Image):
        return f"PIL.Image mode={v.mode} size={v.size}"
    if isinstance(v, str):
        s = v.replace("\n", " ")
        return f"str(len={len(v)}): {s[:160]!r}{'…' if len(v) > 160 else ''}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)}): {str(v)[:160]}"
    if isinstance(v, dict):
        return f"dict(keys={list(v.keys())})"
    return f"{type(v).__name__}: {str(v)[:160]}"


def _scan_green(img):
    """扫图找强绿色像素（疑似画上去的框）。返回 (绿占比, 绿包围盒 or None)。

    强绿判据：G 明显高于 R 和 B（典型纯绿框 (0,255,0) 这类）。细框占比会很小但非零，
    且绿像素会聚成矩形边缘 —— 包围盒接近全图四边即是框。
    """
    import numpy as np

    rgb = img.convert("RGB")
    a = np.asarray(rgb).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (g > 100) & (g - r > 40) & (g - b > 40)
    frac = float(mask.mean())
    box = None
    if mask.any():
        ys, xs = np.where(mask)
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return frac, box


def main() -> int:
    ap = argparse.ArgumentParser(description="探查 MedTrinity-25M (25M_demo) 标注形态")
    ap.add_argument("--n", type=int, default=5, help="探查样本数")
    ap.add_argument("--from-disk", default=None, help="已 save_to_disk 的目录（跳过下载）")
    ap.add_argument("--out-dir", default="probe_c_out", help="存图目录（肉眼核验绿框）")
    args = ap.parse_args()

    samples = _load_samples(args)
    if not samples:
        print("!! 没拿到样本")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n=== 字段结构（共 {len(samples)} 条，列首条所有字段）===")
    first = samples[0]
    for k, v in first.items():
        print(f"  {k:18s} -> {_describe_value(v)}")

    from PIL import Image

    verdict_green = 0
    verdict_coords = 0
    print("\n=== 逐条核验 ===")
    for i, ex in enumerate(samples):
        print(f"\n--- sample[{i}] ---")
        # ② 文本里找坐标
        hits = []
        for k, v in ex.items():
            if isinstance(v, str):
                for pat in COORD_PATTERNS:
                    if pat.search(v):
                        hits.append(f"{k}~/{pat.pattern}/")
        if hits:
            verdict_coords += 1
            print(f"  文本坐标线索: {hits}")
        else:
            print("  文本坐标线索: 无")

        # ③ 图像找绿框 + 存图
        img = next((v for v in ex.values() if isinstance(v, Image.Image)), None)
        if img is not None:
            frac, box = _scan_green(img)
            painted = frac > 1e-4  # 细框也有千分之几
            verdict_green += int(painted)
            print(f"  图像绿像素占比: {frac:.5f}  绿包围盒: {box}  -> "
                  f"{'疑似画了绿框' if painted else '基本无绿（非像素框）'}")
            p = os.path.join(args.out_dir, f"sample_{i}.png")
            img.convert("RGB").save(p)
            print(f"  已存图: {p}（请肉眼确认）")
        else:
            print("  本条无 PIL 图像字段")

    # 汇总判定
    print("\n=== 结论（n={}）===".format(len(samples)))
    print(f"  含像素绿框的样本数: {verdict_green}/{len(samples)}")
    print(f"  含文本坐标线索样本数: {verdict_coords}/{len(samples)}")
    if verdict_green == 0 and verdict_coords > 0:
        print("  => 形态判定：**bbox 坐标 + 文本**（非像素绿框）。")
        print("     C 的提框入口(T024)按'直接读坐标'实现；inpaint/热图/防泄露(T025/26)不变。")
    elif verdict_green > 0:
        print("  => 形态判定：**图上画了框**。保留 cv2.inRange 提框路径（注意框颜色未必正绿，按存图调阈值）。")
    else:
        print("  => 两者都没明显命中：请打开 out-dir 的图 + 上面字段结构人工判断。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
