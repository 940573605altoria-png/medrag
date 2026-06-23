"""CT coreset 选样（T027）—— 在有限算力下选出"最该训练"的代表性子集。

目标：从 c（CT 图文）大池子里挑一个小而**有代表性、保多样、护小病灶**的子集去做视觉训练，
而不是随机抽样（随机会让稀有的小病灶被淹没）。四件事：

1. **面积分层**：按病灶 `AreaBand`（small<2% / med / large）分桶，**小病灶可过采样**
   （`small_band_boost`），守住项目核心命题。
2. **冻结编码器 K-means**：每个分层内对（外部传入的、冻结编码器算好的）嵌入聚类，
   抓住数据内部结构。
3. **√簇配额**：每簇名额 ∝ √簇大小 —— 大簇多给但**次线性**，避免稠密区吃掉全部名额、
   保住稀疏样本的代表性。
4. **三档选样 + 去噪**：簇内先丢离群点（距质心 > mean+zσ），再按 **原型(最靠质心)→
   中位距→FPS(最远点采样补多样)** 的优先级填满该簇名额。

本模块是**纯算法**：输入嵌入 + 每样本病灶面积占比 + id，输出选中的 id 列表。嵌入怎么来
（哪种冻结编码器）是 T031 的事，与本模块解耦，故可用合成嵌入完整单测。

依赖：numpy + scikit-learn(KMeans)。守卫导入 sklearn，缺包给清晰错误。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from src.contracts.schemas import AreaBand

if TYPE_CHECKING:
    import numpy as np


@dataclass
class CoresetConfig:
    """coreset 选样配置。"""

    budget: float = 0.3            # >1 视为绝对条数；(0,1] 视为占总量比例
    clusters_per_stratum: Any = "sqrt"  # "sqrt"->round(√n)；或固定整数
    denoise_z: float = 3.0         # 去噪：丢弃簇内距质心 > mean+z·std 的离群点
    small_band_boost: float = 1.5  # 小病灶分层预算加权（>1 过采样小病灶）
    random_state: int = 42


def _require_kmeans():
    try:
        from sklearn.cluster import KMeans  # noqa: PLC0415

        return KMeans
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "coreset 需要 scikit-learn：pip install scikit-learn"
        ) from exc


def _resolve_k(n: int, spec: Any) -> int:
    """决定簇数：'sqrt'→round(√n)，整数→直接用；夹在 [1, n]。"""
    import numpy as np  # noqa: PLC0415

    k = round(float(np.sqrt(n))) if spec == "sqrt" else int(spec)
    return max(1, min(k, n))


def _sqrt_quota(sizes: "np.ndarray", budget: int) -> "np.ndarray":
    """按 √簇大小分配名额，整数化且总和 == budget（余数给小数部分最大的簇）。"""
    import numpy as np  # noqa: PLC0415

    if budget <= 0 or sizes.sum() == 0:
        return np.zeros_like(sizes)
    w = np.sqrt(sizes.astype(float))
    raw = w / w.sum() * budget
    q = np.floor(raw).astype(int)
    q = np.minimum(q, sizes)                      # 名额不超过簇内样本数
    remainder = budget - int(q.sum())
    # 余数按 (小数部分大 & 还有余量) 优先补
    frac = raw - np.floor(raw)
    order = np.argsort(-frac)
    i = 0
    while remainder > 0 and i < len(order) * 4:
        c = order[i % len(order)]
        if q[c] < sizes[c]:
            q[c] += 1
            remainder -= 1
        i += 1
    return q


def _three_tier_order(Xc: "np.ndarray", center: "np.ndarray") -> list[int]:
    """簇内候选顺序：原型(最靠质心) → 中位距 → FPS(最远点采样)补多样。"""
    import numpy as np  # noqa: PLC0415

    d = np.linalg.norm(Xc - center, axis=1)
    order: list[int] = [int(np.argmin(d))]                 # 原型
    med = int(np.argsort(d)[len(d) // 2])                  # 中位距
    if med not in order:
        order.append(med)
    remaining = [i for i in range(len(Xc)) if i not in order]
    while remaining:                                       # FPS：贪心选离已选集最远者
        sel = np.array(order)
        dmat = np.linalg.norm(Xc[remaining][:, None, :] - Xc[sel][None, :, :], axis=2)
        nxt = remaining[int(np.argmax(dmat.min(axis=1)))]
        order.append(nxt)
        remaining.remove(nxt)
    return order


def _allocate_strata(band_sizes: dict, total_budget: int, boost: float) -> dict:
    """把总预算按 (簇大小 × 小病灶加权) 分到各分层。"""
    import numpy as np  # noqa: PLC0415

    bands = list(band_sizes)
    weights = np.array([
        band_sizes[b] * (boost if b is AreaBand.SMALL else 1.0) for b in bands
    ], dtype=float)
    if weights.sum() == 0:
        return {b: 0 for b in bands}
    raw = weights / weights.sum() * total_budget
    alloc = {b: int(np.floor(r)) for b, r in zip(bands, raw)}
    # 名额不超过该层样本数；余数补给小数部分大的层
    for b in bands:
        alloc[b] = min(alloc[b], band_sizes[b])
    remainder = total_budget - sum(alloc.values())
    frac = {b: raw[i] - np.floor(raw[i]) for i, b in enumerate(bands)}
    for b in sorted(bands, key=lambda x: -frac[x]):
        while remainder > 0 and alloc[b] < band_sizes[b]:
            alloc[b] += 1
            remainder -= 1
        if remainder <= 0:
            break
    return alloc


def select_coreset(
    embeddings: "np.ndarray", area_fractions: Sequence[float],
    ids: Sequence[Any] | None = None, *, config: CoresetConfig | None = None,
) -> list[Any]:
    """选 coreset，返回选中的样本 id 列表。

    `embeddings`: (N, D) 冻结编码器嵌入；`area_fractions`: 每样本病灶面积占比(0–1)，
    用于分层；`ids`: 样本标识（缺省 0..N-1）。预算超过 N 时返回全体。
    """
    import numpy as np  # noqa: PLC0415

    KMeans = _require_kmeans()
    cfg = config or CoresetConfig()
    X = np.asarray(embeddings, dtype=float)
    n = len(X)
    ids = list(ids) if ids is not None else list(range(n))
    if n == 0:
        return []

    total_budget = int(cfg.budget) if cfg.budget > 1 else int(round(cfg.budget * n))
    total_budget = max(0, min(total_budget, n))
    if total_budget >= n:
        return list(ids)

    bands = [AreaBand.from_fraction(f) for f in area_fractions]
    strata: dict = {}
    for i, b in enumerate(bands):
        strata.setdefault(b, []).append(i)
    band_alloc = _allocate_strata(
        {b: len(idxs) for b, idxs in strata.items()}, total_budget, cfg.small_band_boost
    )

    selected: list[Any] = []
    for band, idxs in strata.items():
        budget_b = band_alloc.get(band, 0)
        if budget_b <= 0:
            continue
        idxs_arr = np.array(idxs)
        Xb = X[idxs_arr]
        if budget_b >= len(idxs):                          # 该层全要
            selected.extend(ids[i] for i in idxs)
            continue

        k = _resolve_k(len(idxs), cfg.clusters_per_stratum)
        labels = KMeans(n_clusters=k, n_init=10,
                        random_state=cfg.random_state).fit_predict(Xb)

        # 去噪 + 收集每簇（局部）成员
        clusters: list[np.ndarray] = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            center = Xb[members].mean(axis=0)
            d = np.linalg.norm(Xb[members] - center, axis=1)
            if len(members) > 2 and d.std() > 0:           # 样本够多才去噪
                keep = d <= d.mean() + cfg.denoise_z * d.std()
                members = members[keep]
            if len(members) > 0:
                clusters.append(members)

        sizes = np.array([len(m) for m in clusters])
        quota = _sqrt_quota(sizes, budget_b)
        for members, q in zip(clusters, quota):
            if q <= 0:
                continue
            center = Xb[members].mean(axis=0)
            order = _three_tier_order(Xb[members], center)
            for local in order[:int(q)]:
                selected.append(ids[idxs_arr[members[local]]])

    return selected


__all__ = ["CoresetConfig", "select_coreset"]
