"""最小 FastMCP server（T022）—— 对 LangGraph client 暴露 medrag 的 4 个能力。

骨架阶段：本地 stdio transport，工具由桩管线（`pipeline.py`）驱动。
后续（Phase 7 / T059）升级为双模 transport（远程 streamable HTTP/SSE + 鉴权）。

4 工具（plan.md 部署节，输入/输出 schema 明确、结果带溯源）：
    generate_report(image_id)              → 端到端结构化报告 + 每条结论证据/ROI
    detect_lesions(image_id)               → ROI 列表 + 面积带（C 定位头）
    retrieve_evidence(query, top_k)        → Top-K 证据 + 来源/分数（不生成）
    medical_qa(question)                   → 带引用回答（a/b 知识库）

所有工具统一返回 `ToolIO` 包络：拒答（FR-003）经 `status=ABSTAINED` 显式传达，
不编造；跨项目调用溯源不丢（constitution I）。**MCP 仅是接口层**，与本地直调同管线。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.contracts.schemas import ToolIO, ToolStatus
from src.serve.pipeline import Pipeline, get_pipeline

mcp = FastMCP("medrag")


def _wrap(tool: str, result, *, abstain: bool, reason: str = "") -> ToolIO:
    """把管线结果包成 ToolIO，按 abstain 决定 status。"""
    if abstain:
        return ToolIO.abstained(tool, result, message=reason)
    return ToolIO.ok(tool, result)


@mcp.tool()
def detect_lesions(image_id: str = "stub-image") -> ToolIO:
    """检测 CT 病灶，返回 ROI 列表 + 热图 + 面积带（C 定位头）。"""
    pipe = get_pipeline()
    result = pipe.detect(image_id)
    return _wrap("detect_lesions", result, abstain=result.abstained, reason="no confident lesion")


@mcp.tool()
def retrieve_evidence(query: str, top_k: int = 5) -> ToolIO:
    """检索证据，返回 Top-K 证据 + 来源/分数（RAG 级联，不生成）。"""
    pipe = get_pipeline()
    result = pipe.retrieve(query, top_k)
    return _wrap(
        "retrieve_evidence", result,
        abstain=result.abstain, reason=result.abstain_reason.value,
    )


@mcp.tool()
def medical_qa(question: str, top_k: int = 5) -> ToolIO:
    """医学/药品问答，返回带引用回答（a/b 知识库）。"""
    pipe = get_pipeline()
    result = pipe.answer(question, top_k)
    return _wrap(
        "medical_qa", result,
        abstain=result.abstain, reason=result.abstain_reason.value,
    )


@mcp.tool()
def generate_report(image_id: str = "stub-image", top_k: int = 5) -> ToolIO:
    """端到端：CT → 结构化报告 + 每条结论的证据引用/ROI。"""
    pipe = get_pipeline()
    result = pipe.generate_report(image_id, top_k=top_k)
    return _wrap(
        "generate_report", result,
        abstain=result.abstain, reason=result.abstain_reason.value,
    )


def main() -> None:
    """以本地 stdio transport 启动 server。"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
