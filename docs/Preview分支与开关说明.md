# Preview / Strict 分支与开关说明

本文说明 Stage 0–6 全流程里所有影响「宽严」的分支和开关，以及跑 Preview 时该怎么设。

**核心边界：Strict 分支代码未改动，严格性不受任何影响。**

本文中的命令都假定当前目录是交付包根目录。

---

## 一、两条分支的差别

| | Strict（默认） | Preview（`--preview`） |
|---|---|---|
| Stage 4R 表格恢复 | 不跑 | 跑，插在 Stage 4 和 Stage 5 之间 |
| Stage 1–5 校验失败 | 直接判失败 | `--preview-relaxed`，可离线重放恢复 |
| Stage 6 证据表示层差异 | 判 error | 经 `evidence_matcher` 确定性定位确认后降级为 warning |
| Stage 6 科学语义校验 | 严格 | **一样严格，不放宽** |
| 产物 | `final.json` / `report.html` | 同左，外加 `candidate.json` / `report_candidate.html` |
| 某篇跑不过 | 整篇失败 | 仍发布 candidate（状态 `candidate_partial`），不阻断整批 |

---

## 二、命令行开关

### `extraction/batch_runner.py`（入口）

```powershell
python extraction/batch_runner.py --preview `
  --input-dir  ./sample_data/processed_documents `
  --output-dir ./output_preview `
  --workers 8 --llm-workers 4
```

| 开关 | 作用 |
|---|---|
| `--preview` | 打开预览分支。自动给 Stage 1–5 和 Stage 6 加 `--preview-relaxed`，把 Stage 4R 插进流水线（自动带 `--apply`），最后发布 candidate |
| `--validate-existing` | 只用现有 Stage 0–5 产物重跑 Stage 6，不调模型（不花钱）。**与 `--preview` 互斥**——它走的是固定的 Strict Stage 6 |
| `--force` | 覆盖已有产物。Stage 6 不接受此参数，代码里已排除 |
| `--retry-failed` / `--retry-interrupted` / `--recheck-completed` | 选择要重跑哪批文献 |
| `--dry-run` / `--status` | 只看计划 / 只看状态，不执行 |

### `extraction/stages/stage6_validate_merge.py`

```powershell
python extraction/stages/stage6_validate_merge.py --batch --preview-relaxed `
  --input-root ./output_preview --output-root ./output_preview
```

`--preview-relaxed`：证据的**表示层**差异（Stage 0 存 HTML 表格和 LaTeX、Stage 4R 写管道
渲染行、Stage 4/5 写 LaTeX 剥离后的可读文本——三者指同一处原文但字面互不包含）经
`evidence_matcher` 确定性定位确认后降级为 warning，仍产出 `final.json` 和 `report.html`。

以下情况**照样判 error**，不受此开关影响：

- 证据内容根本不在所引 block 里（按词多重集覆盖率判定，不做模糊数值替换——
  `44` 不会被原文的 `446` 顶掉）；
- `cell_id` 指向的单元格与 `cell_value` 声明的值不符；
- 同一个值在表里出现多次又没有 `cell_id`，无法唯一定位（判 `ambiguous`，不猜第一个）；
- `table_locator` 指向整张表而不是某个单元格；
- `derived_property_ids` 引用了任何 Stage 都不存在的 property。

不带这个开关时，代码路径与本次改动前完全一致。

### `extraction/stages/stage4r_table_recovery.py`

| 开关 | 作用 |
|---|---|
| `--apply` | 把恢复结果写回 `stage4_properties.json`（`batch_runner` 在 Preview 下自动加）。不加则只产出 `.recovery_preview.json` 供人查看 |
| `--threshold` | 恢复的置信度门槛 |
| `--force` | 覆盖已有产物 |

### Stage 1–5 各自的开关

| 开关 | 作用 |
|---|---|
| `--preview-relaxed` | 放宽该 Stage 自身的校验 |
| `--replay-failure` | 从上次的 `stageN_failure.json` 离线重放，**不调模型、不花钱** |
| `--force` | 覆盖已有产物 |

---

## 三、配置文件里的两个 retry（容易混淆）

`extraction/config/pipeline.yaml` 里有两个名字相近但作用完全不同的重试设置。

### `stages.<stage>.max_validation_retries`

反馈驱动的**修复重试**：校验失败时把错误信息回灌给模型，让它改。
代码默认 1，但交付配置里五个 Stage 全设成了 0。

> 设为 0 意味着校验一失败就没有第二次机会。想让重跑更容易通过，可以改成 1 或 2；
> 代价是失败时多一次模型调用的费用。

### `llm.*.max_retries`

只包住**传输层异常**（`requests.Timeout` / `ConnectionError` / `HTTPError` / `ValueError`）。
校验失败它一次都不会重试。当前 `default` 为 0，`stage4` 为 2，`stage5` 为 0。

> 网络抖动、代理错误、HTTP 5xx 属于这一类，调高它有用；但它治不了 schema 校验失败。

---

## 四、怎么跑一遍完整 Preview

```powershell
# 全流程（要调模型，会花钱）
python extraction/batch_runner.py --preview `
  --input-dir ./sample_data/processed_documents `
  --output-dir ./output_preview `
  --workers 8 --llm-workers 4

# 只补 Stage 6 和最终产物（不调模型，不花钱）
python extraction/stages/stage6_validate_merge.py --batch --preview-relaxed `
  --input-root ./output_preview --output-root ./output_preview

# 只补 candidate（不调模型，不花钱）
python preview/publish_candidate.py --batch `
  --input-root ./output_preview --output-root ./output_preview
```

`candidate.json` 由 `publish_candidate.py` 直接读取 Stage 0–5 的 JSON 合并而成，
**不经过 Stage 6**，因此 Stage 6 是否通过不影响它能否生成。

---

## 五、怎么看产物是不是降级通过的

Preview 产出的 `final.json` 顶层带标记，Strict 产出的**没有**这个字段：

```json
{
  "validation_mode": "preview",
  "validation_summary": {
    "validation_status": "degraded",
    "degraded_codes": [
      "evidence_matched_after_normalization",
      "table_locator_matched"
    ]
  }
}
```

降级码含义：

| code | 含义 |
|---|---|
| `evidence_matched` | 按 `cell_id` 精确命中，只是 `source_sentence` 渲染形态不同 |
| `evidence_matched_after_normalization` | 归一化（LaTeX 命令、控制字符、被 PDF 拆散的数字）后一致 |
| `evidence_matched_after_block_recovery` | 词覆盖率 90–98%，或一个 `cell_value` 拼了多个单元格且各分量均可定位 |
| `table_locator_matched*` | locator 标签的渲染形态不同，但单元格可确定性定位 |
| `table_locator_label_missing` | `row_label` 为 null（表格首列为空），但 `cell_id` 能定位到该格 |
| `table_locator_blank_cell_recovered` | 空单元格 locator 未走稳定路径，但按坐标确认该格确实为空 |

带 `validation_mode: preview` 的数据属于**候选**，未经科学语义人工确认，
不得直接宣称可入库。

---

## 六、`evidence_matcher` 的定位优先级

`extraction/stages/evidence_matcher.py` 只在 Preview 下被 Stage 6 调用，按以下顺序
逐层尝试，**能唯一恢复才恢复，不能唯一恢复就报 ambiguous，绝不随意选第一个**：

1. **`cell_id`**——最可靠的表格锚点，直接取那一格；
2. **行列下标**（`row_index` / `column_index`）——`cell_id` 缺失时按坐标取格；
3. **单元格集合匹配**——按值在表内查找，命中唯一才接受，命中多个判 `ambiguous`；
4. **归一化后重试**——展开 LaTeX 命令、去控制字符、合并被 PDF 拆散的数字
   （只在整段都是数字和小数点时才合并，所以 `394 446` 这样两个独立的数不会被粘成
   `394446`）；
5. **词多重集覆盖率**——覆盖率 ≥ 0.98 视为一致，0.90–0.98 记为 block recovery，
   低于 0.90 判不通过。粘连回退只对**纯字母词**生效，数字一律要求精确命中，
   因此 `44` 不可能被 `446` 满足。
