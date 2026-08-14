# 20 篇类型补全 Preview 结果（2026-08-14）

本目录是新的不可变批次，未覆盖历史结果。它基于同一组 20 篇文献，确定性重算 Stage 2/3 的类型字段，随后重新执行 Preview Stage 6、候选发布和两份 HTML 报告生成。

## 流程与血缘

```text
复用 demo20_types_samplefix_20260813_161437 的 Stage 0/1/4/4R/5
→ 用代码 7118a4ab877c9ce7bf96550418d326542fe53fcc 重算 Stage 2/3
→ 重跑 Preview Stage 6 与 candidate 发布
→ 生成本规范化批次
```

- 模式：Preview；
- 文献：20/20；
- 本轮模型调用：0；
- PDF 冷启动验证：否；
- Strict 合规声明：否；
- 生成代码 SHA：`7118a4ab877c9ce7bf96550418d326542fe53fcc`；
- 配置 Git blob（LF）SHA-256：`af6938a4aed3f56348c213729b0a9556ee6c5ffc369fbdb21fb00f876e0de0d6`。

## 自动验证

- `candidate.publication.status == complete`：20/20；
- Stage 6 error 文献：0/20；
- `final.json`、`report.html`、`candidate.json`、`report_candidate.html`：20/20；
- Preview 对象守恒：20/20；
- 输入对象 3107，发布对象 3090，隔离对象 17；
- 清扫悬空引用：6；
- Stage 4R 恢复值：147；因歧义跳过：108；
- 代码测试：564 passed；
- 发布批次结构、哈希、关系引用、敏感路径和文件大小检查：通过。

Candidate 汇总：聚合物实体 291、样品 201、工艺步骤 80、性质对象 272、未解析性质 79、性质系列 48、表征 76、证据 2084。

## 类型评测摘要

评测 GT 固定为 222 行、137 个唯一 `(document, polymer_id)`；`polymer_type` 行级分布为 Homopolymer 139、Copolymer 28、Blend 55，`material_type` 标签分布为 Neat resin 181、Compound 37、Composite 8。私有 GT 文件和本机路径未随批次发布。

- 实体类型弃权：31/291（10.65%）；
- Sample `material_type` 弃权：28/201（13.93%）；
- 预测实体类型：Homopolymer 202、Copolymer 27、Blend 31；
- 预测材料类型：Neat resin 136、Compound 27、Composite 10；
- Blend 文档检测（仅实体）：precision 1.00、recall 0.80；
- Blend 文档检测（实体或 Sample）：precision 1.00、recall 1.00；
- GT majority baseline：`polymer_type` 78.10%，`material_type` 81.53%。

预测实体/GT 唯一聚合物的粒度比为 2.12，预测 Sample/GT 行的粒度比为 0.91，因此上述数字用于覆盖率、弃权率和文档级检测审计，不等同于名称匹配后的类别准确率。

## 已知边界

- 本轮不修改名称策略。原文只有 `PC-5`、`PC-6` 等代号时仍保留代号，不从外部 GT 反推名称；
- `reference_no_0042246` 的 LiClO4 配方已恢复 8 个 Compound Sample，但与 GT 的 23 行相比仍欠覆盖；
- `reference_no_0043590` 的加工链 Sample 已恢复 Blend/Composite，实体层仍未产生 Blend，因此实体口径 Blend recall 为 0.80；
- 4 篇定向工程核查用于验证规则和链路，不构成专家科学语义验收；本批次状态仍为 `partially_validated`。

## 每篇文件

每个 `reference_no_*` 目录包含：

- `candidate.json`、`report_candidate.html`；
- Stage 0–5 中间 JSON及 Stage 4R 审计文件；
- `stage6_validation.json`；
- `final.json`、`report.html`。

本批次不包含私有 GT、原始模型响应、回填 manifest、日志、SQLite、缓存、临时文件或新增 PDF。详细文件哈希见 `RESULT_INDEX.json`，汇总统计见 `validation_summary.json`。
