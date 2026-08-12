# 20 篇 Preview 最终发布结果（2026-08-12）

本目录是新的不可变批次，未覆盖历史批次。它发布 20 篇文献的 Preview 候选结果、Stage 4R 恢复审计、Stage 6 对象级隔离结果以及 HTML 报告。

## 流程与血缘

```text
复用 demo20_stage4r_preview_20260810 的 Stage 0-5 / Stage 4R
→ 使用代码 270dacba9f11f006a0a480c279b7489a8a7d7af7 重跑 Stage 6
→ 生成 final.json / report.html
→ 规范化为本发布批次
```

- 模式：Preview；
- 文献：20/20；
- 本轮模型调用：无；
- PDF 冷启动验证：否；
- Strict 合规声明：否；
- 配置 SHA-256：`3ac6f5ad517e8bef50564130e129eda0df01d29d14e80269b95a4ccfc8535c05`；
- Stage 0-5 的历史 Git SHA 无可靠记录，因此索引中保持 `null`，不编造；
- 本批复用的 Stage 3 早于 `polymer_type/material_type` 新字段，不能视为该字段的 20 篇重跑结果。

## 验证结果

- `candidate.publication.status == complete`：20/20；
- Stage 6 error 文献：0/20；
- `final.json`：20/20；
- `report.html`：20/20；
- Preview 发布对象：3485；
- 隔离对象：15；
- 清扫悬空引用：78；
- 对象守恒：20/20；
- Stage 4R 恢复值：108；
- Stage 4R 因歧义跳过：78。

Candidate 汇总：聚合物实体 279、样品 232、工艺步骤 80、性质对象 188、未解析性质 143、性质系列 105、表征 77、证据 2499。

`partially_validated` 仅表示自动结构、引用、证据和对象守恒检查已经执行；本次打包没有新增专家科学语义复核，不能直接宣称可入正式数据库。

## 每篇文件

每个 `reference_no_*` 目录包含：

- `candidate.json`、`report_candidate.html`；
- Stage 0-5 中间 JSON；
- `stage4_properties.pre_recovery.json`；
- `stage4r_recovery.json`；
- `stage4_properties.recovery_preview.json`；
- `stage4_properties.json`；
- `stage6_validation.json`；
- `final.json`、`report.html`。

`stage4_properties.recovery_preview.json` 是 Stage 4R 使用 `--apply` 时写入 `stage4_properties.json` 的同一份合并结果，为满足发布审计契约补存，不改变数据语义。

## 排除内容

本批次不包含原始模型响应、`pre_name_fix` 备份、日志、SQLite、缓存、临时文件或新增 PDF。原始 PDF 是否可公开分发由仓库现有 `source_pdfs/` 权限状态单独控制。

详细索引见 `RESULT_INDEX.json`，验证统计见 `validation_summary.json`。
