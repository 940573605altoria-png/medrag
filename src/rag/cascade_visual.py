"""c 视觉级联检索（T033）—— 检测头草稿桥接 + 图像主通道兜底 → RetrievalResult。

query-adaptive 级联（plan.md）：
- **检测头草稿桥接**：C 定位头先给草稿（病灶带/器官/findings 文本）→ 走 `c_text` 文本通道粗筛病例；
- **图像主通道兜底**：用查询图像（全图 + ROI 向量）查 `c_img_whole`/`c_img_roi`——草稿空/错时由它兜底。
- 两路按 **RRF** 融合到 `case_id` 粒度（复用 [retrieve_text.rrf_fuse]），出 Top-K 病例证据。

证据按 `case_id` 去重聚合（一个病例多向量命中只算一条，取最高 RRF 分），`modality=IMAGE`。无候选 → 拒答。
检索全用**独立冻结编码器**（embed_text/embed_image），不耦合在训模型（铁律）。逻辑纯、可注入桩本地测。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.contracts.schemas import (
    AbstainReason,
    EvidenceItem,
    Modality,
    RetrievalResult,
)
from src.rag.embed_image import ImageEmbedder
from src.rag.embed_text import TextEmbedder
from src.rag.retrieve_text import rrf_fuse
from src.rag.store import VectorStore


def _case_id(raw_id: str) -> str:
    """从向量 id（`<case>:text` / `:whole` / `:roiN`）取回 case_id。"""
    return raw_id.split(":", 1)[0]


def _ranked_cases(result: dict) -> list[str]:
    """chroma 查询结果 → 去重后的 case_id 排名（保序、首次命中为准）。"""
    ids = result.get("ids", [[]])[0] if result.get("ids") else []
    seen: list[str] = []
    for rid in ids:
        cid = _case_id(rid)
        if cid not in seen:
            seen.append(cid)
    return seen


@dataclass
class VisualCascadeConfig:
    top_n: int = 10
    rrf_k: int = 60
    text_top: int = 30
    image_top: int = 30
    normalize_scores: bool = True


@dataclass
class VisualCascade:
    """视觉级联检索器：文本草稿通道 + 图像主通道（全图/ROI）→ RRF → 病例证据。"""

    store: VectorStore
    image_embedder: ImageEmbedder
    text_embedder: TextEmbedder | None = None
    config: VisualCascadeConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = VisualCascadeConfig()

    def retrieve(
        self,
        query_image: np.ndarray | None,
        *,
        draft_text: str | None = None,
        query: str = "",
    ) -> RetrievalResult:
        cfg = self.config
        rankings: list[list[str]] = []

        # 检测头草稿桥接：文本通道粗筛
        if draft_text and self.text_embedder is not None:
            tv = self.text_embedder.encode_queries([draft_text])
            tres = self.store.query("c_text", tv.tolist(), n_results=cfg.text_top)
            rankings.append(_ranked_cases(tres))

        # 图像主通道兜底：全图 + ROI
        if query_image is not None:
            qv = self.image_embedder.encode_whole([query_image])
            wres = self.store.query("c_img_whole", qv.tolist(), n_results=cfg.image_top)
            rankings.append(_ranked_cases(wres))
            rres = self.store.query("c_img_roi", qv.tolist(), n_results=cfg.image_top)
            rankings.append(_ranked_cases(rres))

        fused = rrf_fuse(rankings, k=cfg.rrf_k)[: cfg.top_n]
        if not fused:
            return RetrievalResult(query=query or (draft_text or ""), evidence=[],
                                   abstain=True, abstain_reason=AbstainReason.NO_EVIDENCE)

        max_s = max(s for _, s in fused) or 1.0
        evidence = [
            EvidenceItem(
                source_id=cid, citation=f"case:{cid}",
                score=(s / max_s if cfg.normalize_scores else s),
                modality=Modality.IMAGE, case_id=cid,
            )
            for cid, s in fused
        ]
        return RetrievalResult(query=query or (draft_text or ""),
                               evidence=evidence, abstain=False)


__all__ = ["VisualCascadeConfig", "VisualCascade"]
