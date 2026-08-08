# 固定 20 篇跑批结果（2026-08-07）

本目录是固定 20 篇的已完成跑批产物，来源于随机 3 篇、随机 7 篇和剩余 10 篇三批结果，合计 20/20。

## 每篇包含

- `stage0_blocks.json`
- `stage1_mentions.json`
- `stage2_entities.json`
- `stage3_process.json`
- `stage4_properties.json`
- `stage5_characterizations.json`
- `candidate.json`
- `report_candidate.html`

直接用浏览器打开对应文献目录下的 `report_candidate.html` 可查看结果；HTML 共用本目录的 `_assets/`。

## 验收模式

这些结果是 Preview Candidate 产物。三批均通过 Preview 完整性验收，并通过 Strict 数据校验报告。详细批次报告位于 `batch_metadata/`，文献到批次的映射及每个文件的 SHA-256 位于 `RESULT_INDEX.json`。

## 未包含内容

为避免上传运行噪声和无用状态，本目录不包含：

- worker 日志；
- SQLite 批处理状态库；
- 历史 `*_failure.json`；
- retry 状态文件；
- 三份重复的 MathJax 资源。

这些排除项不影响查看最终 Stage、Candidate 和 HTML 报告。
