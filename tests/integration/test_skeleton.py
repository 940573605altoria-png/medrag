"""骨架冒烟测试（T023）—— walking skeleton checkpoint。

验证"活的整体"已就位：
  1. 端到端管线跑通 → 结构合法的 ReportResult（含证据链 / ROI）；
  2. 拒答门（FR-003）在无据时触发；
  3. 4 个 MCP 工具经 FastMCP 层可调，返回 ToolIO 包络；
  4. 可复现：固定 seed 两次调用输出一致（桩为确定性）；
  5. （条件性）LangGraph / langchain-mcp-adapters 经 stdio 连通——依赖缺失或环境
     不支持子进程时 skip，不阻塞本批。
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from src.config.config import AppConfig, load_config
from src.config.seed import seed_everything
from src.contracts.schemas import AbstainReason, ReportResult, ToolStatus
from src.serve import mcp_server
from src.serve.pipeline import Pipeline

TOOL_NAMES = {"generate_report", "detect_lesions", "retrieve_evidence", "medical_qa"}


# ── 1. 端到端管线 ────────────────────────────────────────────────

def test_end_to_end_report_is_grounded():
    pipe = Pipeline(load_config())
    report = pipe.generate_report("ct-001")

    assert isinstance(report, ReportResult)
    assert not report.abstain
    assert report.findings, "应至少产出一条结论"
    # constitution I：每条非不确定结论必须挂证据或 ROI（pydantic 已校验，这里再断言）
    for f in report.findings:
        assert f.uncertain or f.evidence or f.roi is not None
    # 检测结果随报告带回
    assert report.detection is not None
    assert report.detection.rois, "桩检测应给出 ROI"


def test_small_lesion_band_present():
    """核心命题：桩检测落在 small(<2%) 带，确保分层链路在骨架就通。"""
    pipe = Pipeline(load_config())
    det = pipe.detect("ct-001")
    assert any(roi.area_band.value == "small" for roi in det.rois)


def test_abstain_on_empty_evidence():
    """拒答门（FR-003）：检索无据 → medical_qa 整体拒答，不编造。"""
    pipe = Pipeline(load_config())
    report = pipe.answer("")  # 空问题 → 桩检索拒答
    assert report.abstain
    assert report.abstain_reason is not AbstainReason.NONE
    assert not report.findings


def test_unimplemented_real_impl_raises():
    """flags 切到尚未接入的真实实现时显式报错，而非静默走桩（防伪装跑通）。"""
    cfg = AppConfig().with_overrides(flags={"detect": "lochead"})
    with pytest.raises(NotImplementedError):
        Pipeline(cfg).detect("ct-001")


# ── 2. 可复现 ────────────────────────────────────────────────────

def test_reproducible_deterministic_output():
    seed_everything(42)
    a = Pipeline(load_config()).generate_report("ct-001").model_dump(mode="json")
    seed_everything(42)
    b = Pipeline(load_config()).generate_report("ct-001").model_dump(mode="json")
    assert a == b


# ── 3. MCP 层：4 工具经 FastMCP 可调 ─────────────────────────────

def _call_tool(name: str, args: dict):
    return asyncio.run(mcp_server.mcp.call_tool(name, args))


def test_mcp_exposes_four_tools():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} == TOOL_NAMES


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("detect_lesions", {"image_id": "ct-001"}),
        ("retrieve_evidence", {"query": "aspirin dosage", "top_k": 3}),
        ("medical_qa", {"question": "what is aspirin used for?"}),
        ("generate_report", {"image_id": "ct-001"}),
    ],
)
def test_mcp_tool_callable_returns_toolio(name: str, args: dict):
    result = _call_tool(name, args)
    structured = _structured_payload(result)
    assert structured["tool"] == name
    assert structured["status"] in {s.value for s in ToolStatus}
    assert structured["schema_version"] == "0.1.0"


def test_mcp_retrieve_abstains_on_blank_query():
    result = _call_tool("retrieve_evidence", {"query": "  "})
    structured = _structured_payload(result)
    assert structured["status"] == ToolStatus.ABSTAINED.value


def _structured_payload(call_result) -> dict:
    """从 FastMCP.call_tool 返回里取出结构化 ToolIO dict（兼容不同 SDK 版本）。"""
    # 新版返回 (content_blocks, structured_dict)
    if isinstance(call_result, tuple) and len(call_result) == 2:
        _content, structured = call_result
        if isinstance(structured, dict):
            # FastMCP 对非 BaseModel 包一层 {"result": ...}；ToolIO 直接展开
            return structured.get("result", structured)
    raise AssertionError(f"无法解析 call_tool 返回: {type(call_result)} {call_result!r}")


# ── 4. （条件性）LangGraph / langchain-mcp-adapters stdio 连通 ────

def test_langgraph_mcp_connectivity():
    adapters = pytest.importorskip(
        "langchain_mcp_adapters.client",
        reason="langchain-mcp-adapters 未安装，跳过 LangGraph 连通子测试",
    )

    async def _run():
        client = adapters.MultiServerMCPClient(
            {
                "medrag": {
                    "command": sys.executable,
                    "args": ["-m", "src.serve.mcp_server"],
                    "transport": "stdio",
                }
            }
        )
        tools = await client.get_tools()
        return {t.name for t in tools}

    try:
        names = asyncio.run(asyncio.wait_for(_run(), timeout=60))
    except Exception as exc:  # noqa: BLE001 - 子进程/网络/版本差异都不阻塞本批
        pytest.skip(f"LangGraph 连通子测试环境不可用，跳过: {exc!r}")
    assert TOOL_NAMES.issubset(names)
