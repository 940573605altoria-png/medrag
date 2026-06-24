# Constitution 门复核（T069）

> 对照 [constitution](../../.specify/memory/constitution.md) 三条不可协商铁律，逐项映射到已落地代码与
> 测试。🟢 = 已本地强制 + 测试覆盖；🟡 = 机制就位、真实功能待 AutoDL 验。

## I. 证据可溯源、禁编造

| 机制 | 落地 | 测试 |
|---|---|---|
| 每条结论必挂证据/ROI，否则标 uncertain 或拒答 | `contracts.schemas.Finding._must_be_grounded`、`models.report.assemble_report` | `test_report.py`、`test_qa.py` 🟢 |
| 伪造引用不挂凭空来源 | `assemble_report` 丢无效 `[S*]` → uncertain | `test_qa_cited.py::test_fabricated_citation...` 🟢 |
| 检索/精排无据/低置信 → 显式拒答 | `RetrievalResult.abstain`、`rerank` 拒答门 | `test_rerank.py`、`test_qa_cited.py` 🟢 |
| 去重合并保留被并来源 id | `dedup._merge_into`（source_ids 并入） | `test_normalize_dedup.py` 🟢 |
| 工具输出带溯源包络 | `ToolIO`（payload 携证据） | `test_qa_cited.py` 🟢 |

## II. 防标签泄露 / 训推一致

| 机制 | 落地 | 测试 |
|---|---|---|
| 绿框只当标签、推理输入恒为无框图 | `data.ct_inpaint`（抹框 + 二阶随机诱饵） | `test_data.py` 🟢 / 真实 CT 🟡 |
| 索引端/查询端同模型同预处理 | `embed_text.TextEmbedder`（非对称但共用实例/归一化） | `test_rag_text.py` 🟢 |
| 质量筛 df 基准排除评估集 | `ner_quality.document_frequency(eval_mask=)` | `test_ner_quality.py` 🟢 |
| 覆盖护栏：去重/筛选不丢独有实体 | `ner_quality.coverage_guard`、`ingest` 全程护栏 | `test_ner_quality.py`、`test_ingest.py` 🟢 |
| PHI 不入库/不外发（FR-007） | `deid` + ingestion 前置 deidentify + `assert_no_phi` | `test_no_phi_egress.py` 🟢 |
| ROI 字段索引/推理端统一 | `contracts.schemas.ROI`（绿框坐标 / 定位头同字段） | schema 校验 🟢 |

## III. 固定基线 + 一次一变量 + 逐项消融

| 机制 | 落地 | 测试 |
|---|---|---|
| 配置驱动评估、与模型解耦 | `eval.runner.run_eval(predict_fn, metric_fn)` | `test_eval_harness.py` 🟢 |
| 消融强制一次只改一个 flag | `eval.ablation.build_variants` | `test_eval_harness.py` 🟢 |
| 相对基线 delta + 配对显著性 | `ablation` + `stats`（bootstrap/置换/McNemar） | `test_eval_harness.py`、`test_ablation.py` 🟢 |
| 小病灶分层贯穿 | 指标按 `AreaBand` 分层（FROC/消融） | `test_eval.py`、`test_ablation.py` 🟢 |
| 质量门：不显著 → 阻断 | `runner.QualityGate.assert_pass` | `test_eval_harness.py` 🟢 |
| 可复现：固定 seed 两跑一致 | `stats` 固定 seed、`config.seed` | `test_repro.py` 🟢 |
| 固定基线 runner（vanilla/朴素 LoRA） | T017/T018 🟡 待建 | — |

## 结论
- **铁律 I/II/III 的核心机制均已在代码层强制 + 本地测试覆盖**（199 passed / 12 skipped）。
- 待 AutoDL/GPU 收口项（不影响铁律成立，仅功能验证）：B 融合接线(T012+T030后半)、真实模型功能、
  固定基线 runner(T017/T018)、训练(T035)、生产部署(T058–T062)。详见 `CLAUDE.md` 的"⚠️ 必须 AutoDL"节。
