# demo20 Preview 修复与全量验证报告

- 验证日期：2026-08-09
- 验证范围：仅 Preview 放宽；Strict 与全局 Schema 保持原校验语义
- 验证目录：`batch_results/demo20_preview_20260809`

## 结论

- 20/20 文献均生成 `stage4_properties.json`。
- `preview_degraded_empty_shell`：0/20。
- Stage 4 空结果：0/20。
- 完整模型响应归档：20/20；当前无 `stage4_failure.json`。
- Preview salvage 未删除任何 condition、property、unresolved、series、point、evidence 或 coordinate。
- 20/20 均生成 `candidate.json`，`publication.status=complete`，且 `stage_failures=[]`。
- Stage 4/5 到最终 `candidate.json` 的 ID、series 和 points 数量全部精确一致。

本轮可以确认：**20 篇在 Preview 流程上全部跑通，Stage 4 不再因单个格式错误整篇清空，恢复的数据能正确进入最终汇总 JSON。**

不能宣称 Strict 完全合规或所有语义百分之百正确：18 篇带 `preview_semantic_validation_bypassed`，这是本轮按需求允许“内容与原文对应校验暂不阻断”的结果，后续仍需 Strict/人工复核。

## 本轮代码范围

- `extraction/stages/stage4_property.py`
  - Preview 单元素 `PropertySeriesCoordinate.evidence` 数组确定性解包；Strict 保持报错。
  - 配合此前 Preview 修复：evidence 字段别名/缺失句补齐、condition quantity 清理、series 非法字段删除、多主体 series salvage、raw response 安全持久化等。
- `extraction/tests/test_stage4_property.py`
  - 新增 Preview 解包与 Strict 不解包测试。

## 测试

- Stage 4：162 passed。
- Preview publisher + Stage 6：22 passed。

## Stage 4 全量统计

| 指标 | 数量 |
|---|---:|
| 文献 | 20 |
| conditions | 87 |
| scalar properties | 61 |
| unresolved properties | 35 |
| property series | 105 |
| series points | 691 |
| degraded 空壳 | 0 |
| 有 salvage 删除的文献 | 0 |
| raw response | 20 |

## 最终 candidate 对账

| 对账项 | Stage 输入 | candidate |
|---|---:|---:|
| Measurement conditions | 87 | 87 |
| Property observations（Stage 4 + Stage 5） | 61 + 127 | 188 |
| Unresolved properties | 35 | 35 |
| Property series | 105 | 105 |
| Series points | 691 | 691 |
| Characterizations | 77 | 77 |

- 完整 candidate：20/20。
- 精确对账通过：20/20。
- 对账错误：0。

## 与旧 demo20 Stage 4 对比

| 指标 | 旧结果 | 新结果 |
|---|---:|---:|
| conditions | 25 | 87 |
| scalar properties | 91 | 61 |
| unresolved | 7 | 35 |
| series | 5 | 105 |
| points | 30 | 691 |
| degraded 空壳 | 14 | 0 |

新结果中 scalar 数量低于旧结果不代表丢失：大量表格多值数据改为结构化 `property_series/points`，无法安全归一的内容进入 `unresolved`；新结果新增 105 个 series、691 个 points。

## 逐篇 Stage 4 与 candidate 状态

| 文献 | condition | scalar | unresolved | series | points | candidate |
|---|---:|---:|---:|---:|---:|---|
| reference_no_0101911 | 6 | 11 | 0 | 9 | 42 | complete / exact=true |
| reference_no_0025452 | 4 | 1 | 3 | 5 | 50 | complete / exact=true |
| reference_no_0043955 | 5 | 1 | 0 | 3 | 8 | complete / exact=true |
| reference_no_0043590 | 3 | 4 | 1 | 1 | 5 | complete / exact=true |
| reference_no_0042367 | 4 | 7 | 1 | 2 | 6 | complete / exact=true |
| reference_no_0043541 | 10 | 6 | 0 | 8 | 24 | complete / exact=true |
| reference_no_0042480 | 4 | 0 | 11 | 5 | 20 | complete / exact=true |
| reference_no_0042246 | 5 | 3 | 1 | 2 | 16 | complete / exact=true |
| reference_no_0039705 | 9 | 3 | 2 | 16 | 96 | complete / exact=true |
| reference_no_0038813 | 0 | 0 | 6 | 15 | 156 | complete / exact=true |
| reference_no_0038527 | 6 | 2 | 0 | 8 | 109 | complete / exact=true |
| reference_no_0037921 | 1 | 1 | 0 | 0 | 0 | complete / exact=true |
| reference_no_0037886 | 2 | 4 | 2 | 1 | 1 | complete / exact=true |
| reference_no_0037645 | 5 | 3 | 2 | 3 | 20 | complete / exact=true |
| reference_no_0037607 | 2 | 0 | 2 | 4 | 28 | complete / exact=true |
| reference_no_0037268 | 3 | 2 | 0 | 1 | 3 | complete / exact=true |
| reference_no_0033617 | 6 | 11 | 0 | 3 | 12 | complete / exact=true |
| reference_no_0020284 | 6 | 0 | 0 | 7 | 42 | complete / exact=true |
| reference_no_0021296 | 6 | 2 | 0 | 11 | 49 | complete / exact=true |
| reference_no_0073324 | 0 | 0 | 4 | 1 | 4 | complete / exact=true |

## 已知限制与安全处理

- 4 篇共有 8 个 series 标记 `property_series_incomplete`，合计 10 个未覆盖数值点。
- 这些点没有被删除，均保留 point、主体关系和 table locator。
- 原表单元格为真正空白、`-a)` 或非数值脚注字母（如 `k/l`），本轮不猜测、不伪造数值。
- 没有出现 `table_property_column_missing` 或 `table_property_column_represented_as_coordinate` warning。
- Preview 允许语义校验 warning 继续；需要发表级准确性时，应另跑 Strict 并人工复核。

### incomplete series 清单

| 文献 | series | 性质 | points | covered/expected |
|---|---|---|---:|---:|
| reference_no_0043955 | series002 | M_{\text{w}} | 3 | 2/3 |
| reference_no_0043955 | series003 | M_{\text{w}}/M_{\text{n}} | 2 | 1/2 |
| reference_no_0038527 | series001 | $M_{n}^{a,b}$ | 15 | 14/15 |
| reference_no_0038527 | series002 | $M_{w}^{a,c}$ | 15 | 14/15 |
| reference_no_0038527 | series006 | $T_{m}^{g}$ | 15 | 13/15 |
| reference_no_0038527 | series007 | $T_{c}^{h}$ | 15 | 13/15 |
| reference_no_0037607 | series003 | T_g | 7 | 6/7 |
| reference_no_0073324 | series001 | PMT, °C. | 4 | 3/4 |

## 机器可读报告

- `stage4_preview_validation_report.json` / `.csv`
- `candidate_validation_report.json` / `.csv`
- `stage4_warning_audit.json`
- `old_vs_new_stage4_report.json`
- `incomplete_series_audit.json`
- `incomplete_series_missing_points.json`
