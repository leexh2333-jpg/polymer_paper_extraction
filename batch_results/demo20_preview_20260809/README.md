# 固定 20 篇 Preview 修复验证结果（2026-08-09）

本目录保存 2026-08-09 的 Stage 4 Preview 修复验证产物。为保留历史和便于对照，本批次**没有覆盖** `../demo20_20260807/`。

## 验证范围

本次使用已有的 Stage 0/1/2/3/5 中间结果，真实重跑 Stage 4，然后重新生成 `candidate.json` 和 `report_candidate.html`：

```text
已有 Stage 0/1/2/3/5
        ↓
重跑 Stage 4（Preview）
        ↓
发布 candidate.json
        ↓
生成 report_candidate.html
```

本批次不是“删除所有中间结果后从原始 PDF 冷启动”的验收，因此不把它表述为 PDF→Stage 0–5 的全链路冷启动证明。

## 验收结果

- 文献数：20/20；
- Stage 4 非空：20/20；
- `preview_degraded_empty_shell`：0/20；
- Stage 4 原始模型响应已保存：20/20；
- `publication.status == complete`：20/20；
- Candidate 对账检查通过：20/20；
- Candidate `stage_failures` 总数：0。

汇总数量：

- measurement conditions：87；
- Stage 4 scalar properties：61；
- Stage 4 unresolved properties：35；
- property series：105；
- series points：691；
- Candidate property observations：188（其中 Stage 5 properties 为 127）；
- Stage 5 characterizations：77。

## 每篇文件

每个 `reference_no_*` 目录包含：

- `stage0_blocks.json`
- `stage1_mentions.json`
- `stage2_entities.json`
- `stage3_process.json`
- `stage4_llm_response.json`：本轮 Stage 4 完整响应，用于审计和离线回放；
- `stage4_properties.json`
- `stage5_characterizations.json`
- `candidate.json`
- `report_candidate.html`

HTML 共用本目录 `_assets/`。原始 PDF 位于仓库根目录 `source_pdfs/`。

## 重要限制

- 18 篇带有 `preview_semantic_validation_bypassed`：这是按当前目标允许 Preview 带 warning 继续，不代表 Strict 合规；
- 4 篇共 8 个 series 标记 `property_series_incomplete`，对应原表空白、脚注或无法安全转换成数值的单元格；系统保留定位信息但不猜值；
- 本轮目标是提高 Preview 跑通率和数据保留率，不承诺发表级准确率，也不宣称 20 篇全部通过 Strict。

## 验证报告

- `demo20_preview_fix_validation_report.md`：人工可读总报告；
- `stage4_preview_validation_report.json/.csv`：Stage 4 状态与数量；
- `candidate_validation_report.json/.csv`：Stage 4/5 到 Candidate 的精确对账；
- `stage4_warning_audit.json`：warning 审计；
- `old_vs_new_stage4_report.json`：旧批次与新批次对比；
- `incomplete_series_audit.json`、`incomplete_series_missing_points.json`：不完整 series 审计；
- `RESULT_INDEX.json`：逐篇状态、数量和文件 SHA-256。

直接查看任一文章：

```text
batch_results/demo20_preview_20260809/<reference_no>/report_candidate.html
```
