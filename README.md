# medrag — 医学多模态病灶检测与报告生成系统

基于 **RAG + Qwen3-VL** 的"病灶检测 + 报告生成"医学多模态垂直系统，针对通用 VLM 的
**微小病灶特征湮灭**与**幻觉**两大痛点。详见 [CLAUDE.md](CLAUDE.md) 项目看板与
[specs/001-medrag-detect-report/](specs/001-medrag-detect-report/) 设计规格。

## 当前状态：Walking Skeleton（桩端到端）

第一批落地的是**活骨架**——用桩把 `CT → detect → retrieve → report` 端到端与 4 个 MCP 工具
串通跑起来，锁定接口/集成/服务边界；真实组件（C 定位、B 融合、RAG、报告）后续逐个替桩。

## 快速开始（骨架阶段）

```bash
pip install -e .            # 仅装轻量依赖（pydantic / mcp / numpy）
pip install -e ".[dev]"     # + pytest 等测试依赖
pytest tests/integration/test_skeleton.py -v
```

## 模块结构（`src/`）

| 模块 | 职责 |
|---|---|
| `config/`    | 全局配置、可复现 seed、run-record |
| `contracts/` | 共享数据契约（schema）——桩与真实实现的接口地基 |
| `data/`      | 数据管线（ingestion / 去重 / NER / 绿框→标签 / coreset）|
| `models/`    | Qwen3-VL 封装、双路融合(B)、定位头(C)、损失 |
| `train/`     | LLaMA-Factory/ms-swift 配置、LoRA、DeepSpeed |
| `rag/`       | embedding、父子层级检索、级联、reranker |
| `eval/`      | 配置驱动消融 runner + 指标 + 显著性 |
| `serve/`     | 端到端管线 + FastMCP server（4 工具）|

依赖分组见 [pyproject.toml](pyproject.toml)：`data` / `train` / `rag` / `eval` 按阶段按需安装。
