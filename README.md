# 聚合物文献抽取流程交付包

首次交付日期：2026-08-07

最近更新日期：2026-08-12

固定数据集：20 篇文献

运行环境：Windows PowerShell + Python 3.12（建议）
交付目标：既可从随包的标准化文档直接运行 Stage 0–6，也可从随包 PDF 启动 MinerU OCR/解析和抽取全流程。

> 交付用户优先使用包根目录的 `run_demo20_delivery.ps1` 和 `run_pdf_pipeline_delivery.ps1`。`preview/run_demo20.ps1` 是组件级高级入口，详见第 6 节。

> **2026-08-09 Preview 更新：** Stage 4 已改为尽量确定性修复并逐对象保留合法数据，避免单个格式错误导致整篇清空。新版 20 篇结果位于 `batch_results/demo20_preview_20260809/`；旧批次保留用于对照。
>
> **2026-08-10 Stage 4R 更新：** Preview 在 Stage 4 与 Stage 5 之间增加确定性表格补抽。Stage 4R 按稳定 `cell_id` 恢复明确缺失的表格性质；无法唯一归属的值保留为 unresolved，不随意绑定实体。当前版本进一步支持 `0-2-0-6` 等数字连字符样品编码、多层表头、LaTeX 数值和扩展性质别名。Strict 流程不执行 Stage 4R。
>
> **2026-08-10 Stage 2 名称更新：** 同一实体同时包含具体聚合物名称和样品代号/缩写时，优先采用有原文 mention 支持的具体名称；原代号继续保留用于追溯。无法安全确定时保持原名称，不生成或猜测 canonical name。
>
> **2026-08-11 Stage 6 Preview 更新：** Preview 现在会执行 Stage 6（带 `--preview-relaxed`），产出 `final.json` 和 `report.html`，此前 Preview 直接跳过 Stage 6。新增 `extraction/stages/evidence_matcher.py`：Stage 0 存 HTML 表格和 LaTeX，Stage 4R 写管道渲染行，Stage 4/5 写可读文本，三者指同一处原文但字面互不包含，旧版按字面子串判定会误报。Preview 下这类**表示层**差异经确定性定位确认后降级为 warning；**科学语义校验没有放宽**，定位不上的证据、指错单元格的 locator、引用不存在 property 的 `derived_property_ids` 一律仍判 error。**Strict 分支代码未改动，判定结果逐条不变。**
>
> **2026-08-12 Preview 完整发布更新：** Stage 6 新增逐对象隔离和引用清扫。坏对象进入 `rejected_objects`，合法对象继续进入 `final.json`；悬空 `derived_property_ids` 只剪枝、不猜 ID。Characterization 允许表级 locator，Property 和 Series Point 仍要求单元格级定位。Stage 4R 的 evidence 直接保存稳定 `cell_id` 对应的 Stage 0 单元格文本，不再重新拼接管道行。Stage 3 同时新增 `polymer_type` 和 `material_type`。Strict 的校验规则和失败语义保持不变。最新规范化 20 篇结果位于 `batch_results/demo20_preview_final_20260812/`。

## 1. 交付包目录总览

```text
polymer_extraction_delivery_20260807/
├─ pipeline_runner.py                  PDF 全流程总入口
├─ run_demo20_delivery.ps1             标准化 20 篇一键入口（推荐）
├─ run_pdf_pipeline_delivery.ps1       PDF 20 篇一键入口
├─ verify_delivery.py                  无费用交付结构检查
├─ extraction/                         Stage 0–6 抽取引擎
├─ ocr/                                MinerU、结果整理和标准化
├─ preview/                            演示编排、候选发布和验收工具
├─ sample_data/processed_documents/    固定 20 篇标准化输入 JSON
├─ source_pdfs/                        固定 20 篇原文 PDF
├─ acceptance/                         历史验收摘要与报告
├─ batch_results/demo20_20260807/      2026-08-07 历史跑批结果
├─ batch_results/demo20_preview_20260809/  2026-08-09 Preview 修复验证结果
├─ batch_results/demo20_preview_final_20260812/  2026-08-12 Preview 最终发布结果
├─ docs/                               项目和风险说明
├─ .env.example                        密钥占位模板，不含真实密钥
├─ requirements.txt                    运行依赖
├─ requirements-dev.txt                测试依赖
├─ MANIFEST.json                       交付文件清单和哈希
└─ SHA256SUMS.txt                      逐文件 SHA-256
```

本包明确不包含：

- 真实 API 密钥和 `.env`；
- 开发机缓存、字节码和 `.pytest_cache`；
- 历史运行日志和 SQLite 状态库；
- 与交付无关的历史模型原始响应；`demo20_preview_20260809` 为审计和离线回放保留本轮 Stage 4 完整响应；
- 开发目录中的 `output_test` 等历史输出。

## 2. 三个容易混淆的目录

### 2.1 `extraction/`：真正执行抽取的核心代码

`extraction/` 负责模型调用、JSON 解析、Stage 运行、缓存、重试、失败回放和结果生成。

主要内容：

| 路径 | 用途 |
|---|---|
| `extraction/batch_runner.py` | 多文档批处理、断点续跑、状态库和失败恢复；Preview 编排 Stage 4R |
| `extraction/llm_client.py` | 模型请求、响应记录、JSON 解析和有限修复 |
| `extraction/config/pipeline.yaml` | 模型、Stage 参数、并发和相对路径配置 |
| `extraction/config/polymer_schema.yaml` | Stage 4/5 使用的聚合物词表和 Schema 配置 |
| `extraction/stages/stage2_polymer_entity.py` | 聚合物实体归并、名称选择、重复 mention 修复和 unresolved 处理 |
| `extraction/stages/stage4r_table_recovery.py` | Preview-only 表格缺口恢复，按 `cell_id` 合并；支持数字连字符样品编码的严格别名匹配 |
| `extraction/stages/table_recall_audit.py` | 单元格级表格召回审计，识别性质值、坐标、LaTeX 数值和多层表头 |
| `extraction/stages/evidence_matcher.py` | Preview-only 证据定位器：按 `cell_id` → 行列下标 → 单元格集合逐层确定性定位，判定表示层差异；仅被 Stage 6 的 Preview 分支调用 |
| `extraction/prompts/` | 各 Stage Prompt |
| `extraction/schema/` | Pydantic/JSON 数据结构 |
| `extraction/stages/` | Stage 0–6 实现 |
| `extraction/reports/` | HTML 报告资源和渲染代码 |
| `extraction/tools/` | failure 离线回放等维护工具 |
| `extraction/tests/` | 核心抽取代码的自动化回归测试 |

### 2.2 `extraction/tests/`：自动化测试，不是运行数据

该目录包含 19 个 `test_*.py` 和一个公共辅助文件 `helpers.py`。文件较多是因为 Stage 0–6、模型客户端、批处理器、HTML、缓存和失败回放都分别有行为测试。

典型文件：

- `test_stage1_material_mention.py`：mention 提取、原文定位和 Preview 降级；
- `test_stage2_polymer_entity.py`：实体解析、重复 mention 和 unresolved；
- `test_stage3_sample_process.py`：样品标签、Process 图和 evidence；
- `test_stage4_property.py`：性能、条件、单位和 evidence；
- `test_stage5_characterization.py`：表征数据；
- `test_stage6_validate_merge.py`：严格合并和一致性校验，以及 Preview 降级分支（表示层差异降级、指错单元格仍判 error、降级标记写入 `final.json`）；
- `test_evidence_matcher.py`：证据定位器的分层匹配与**拒绝**行为（`44` 不被 `446` 顶掉、两个独立整数不被粘连、重复值无 `cell_id` 时判 ambiguous）；
- `test_llm_client.py`：JSON 解析、非法转义修复和传输错误；
- `test_batch_runner.py`：批处理、缓存、续跑、partial、Stage 4R Preview 编排、Stage 6 Preview 编排和退出码；
- `test_stage4r_table_recovery.py`：`cell_id` 合并、确定性实体归属、备份和强制重跑；
- `test_table_recall_audit.py`：表格数值单元格角色和召回缺口审计。

这些测试：

- 正常运行 pipeline 时不会自动执行；
- 不会被当作论文输入；
- 默认使用测试替身或保存响应，不产生模型费用；
- 用于证明 Preview 放松没有破坏 Strict，并方便接收方验证环境。

如果只想运行程序，可以忽略 `tests/`；如果需要修改代码或验收交付，建议保留。

### 2.3 `preview/`：演示和验收工具，不是另一套抽取模型

`preview/` 不负责重新实现 Stage 0–6。它负责固定 20 篇、调用批处理、聚合候选结果、生成 HTML，以及区分 Preview 完成和 Strict 通过。

| 文件 | 用途 |
|---|---|
| `preview/demo_latest_20_refs.txt` | 唯一的固定 20 篇 reference_no 清单 |
| `preview/demo_latest_20_selection.json` | 20 篇选择过程的审计信息 |
| `preview/publish_candidate.py` | 将 Stage 0–5 汇总为 `candidate.json` 和 `report_candidate.html` |
| `preview/verify_demo20.py` | 验收 Stage 文件、candidate、HTML 和发布状态 |
| `preview/run_demo20.ps1` | 组件级 Preflight/Verify/Cached/Fresh 高级入口 |
| `preview/tests/` | 候选发布和验收器测试 |

数据流：

```text
标准化 document.json
  ↓
Preview：Stage 0 → 1 → 2 → 3 → 4 → 4R → 5 → 6（--preview-relaxed）
  ↓
preview/publish_candidate.py
  ├─ candidate.json
  └─ report_candidate.html
  ↓
preview/verify_demo20.py
  ├─ Preview 完成检查
  └─ Strict 严格验收
```

`candidate.json` 是聚合运行视图；`report_candidate.html` 是供人查看的候选报告。它们不替代各 Stage 原始 JSON。

Preview 下 Stage 6 会隔离可定位到单个对象的错误，清扫悬空引用后产出 `final.json` 和 `report.html`（带降级标记，见 8.5）。只有文档级错误或无法安全隔离的错误仍会阻止 final；此时仍发布 `candidate.json`，状态记为 `candidate_partial`，不阻断整批推进。

Strict 仍按 `Stage 0 → 1 → 2 → 3 → 4 → 5 → 6` 执行，不经过 Stage 4R。Preview 中 Stage 4R 会生成 `stage4r_recovery.json` 和 `stage4_properties.recovery_preview.json`，应用前的 Stage 4 保存在 `stage4_properties.pre_recovery.json`；补抽后的 `stage4_properties.json` 再交给 Stage 5 和候选发布器。

## 3. 环境准备

建议使用 Python 3.12，以及 Windows PowerShell 5.1 或 PowerShell 7。

```powershell
cd <解压后的交付目录>
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt
```

需要运行自动化测试时再安装开发依赖：

```powershell
python -m pip install -r ./requirements-dev.txt
```

## 4. 配置密钥

复制占位文件并在本机填写。不要把真实密钥写回交付 ZIP，也不要转发 `.env`。

```powershell
Copy-Item ./.env.example ./.env
notepad ./.env
```

当前配置：

- 从标准化 JSON 开始抽取：需要 `DMX_API_KEY`，兼容 `LLM_API_KEY`；
- 从 PDF 开始完整流程：还需要 `MINERU_API_KEY`。

也可以只在当前 PowerShell 会话中设置：

```powershell
$env:DMX_API_KEY = "<你的密钥>"
$env:MINERU_API_KEY = "<你的密钥>"
```

程序不会主动把 API key 写入候选产物。运行日志、错误消息和交付文件中也不应出现真实密钥。

## 5. 先做无费用检查

以下命令不调用模型或 MinerU：

```powershell
python ./verify_delivery.py
./run_demo20_delivery.ps1 -DryRun
./run_pdf_pipeline_delivery.ps1 -DryRun
```

`verify_delivery.py` 会检查：

- 固定清单是否恰好为 20 篇且无重复；
- 20 个标准化 JSON 是否存在并可解析；
- 20 个 PDF 是否存在且具有 PDF 文件头；
- 包内是否误含 `.env`、缓存或字节码；
- `pipeline.yaml` 是否残留密钥形式或 Windows 绝对路径。

## 6. 从已标准化的 20 篇 JSON 开始（推荐）

### 6.1 Preview

```powershell
./run_demo20_delivery.ps1
```

默认读取：

```text
sample_data/processed_documents/
```

默认输出：

```text
output_preview/
```

每篇文献会得到 Stage 0–5 的原始 JSON、Stage 4R 的恢复报告、`candidate.json` 和
`report_candidate.html`；Stage 6 校验通过的文献另外得到 `final.json` 和 `report.html`。

#### 直接调用 batch_runner（需要自定义参数时）

```powershell
python extraction/batch_runner.py --preview `
  --input-dir  ./sample_data/processed_documents `
  --output-dir ./output_preview `
  --workers 8 --llm-workers 4
```

`--preview` 一个开关就同时做了四件事，不需要再逐个 Stage 指定：

1. 在 Stage 4 和 Stage 5 之间插入 Stage 4R 表格恢复（自动带 `--apply`）；
2. 给 Stage 1–5 加 `--preview-relaxed`，失败时先尝试离线重放恢复；
3. 给 Stage 6 加 `--preview-relaxed`（本次新增）；
4. 最后发布 `candidate.json` 和 `report_candidate.html`，某个 Stage 失败时也照常发布。

#### 只补 Stage 6 和最终产物（不调模型，不花钱）

Stage 0–5 的产物已经在手上、只想重跑校验和合并时：

```powershell
python extraction/stages/stage6_validate_merge.py --batch --preview-relaxed `
  --input-root ./output_preview --output-root ./output_preview
```

去掉 `--preview-relaxed` 就是 Strict 校验，代码路径与本次改动前完全一致。

#### 只补 candidate（不调模型，不花钱）

```powershell
Get-ChildItem ./output_preview -Directory -Filter reference_no_* | ForEach-Object {
  python preview/publish_candidate.py --ref-no $_.Name `
    --input-root ./output_preview --output-root ./output_preview
}
```

`publish_candidate.py` 一次只处理一篇，没有 `--batch`，所以这里用循环。
`--ref-no` 需要传完整目录名（含 `reference_no_` 前缀）。传错或目录不存在时脚本会
报错退出（exit code 非 0），不会创建输出目录，也不会写出空的 `candidate.json`。

`candidate.json` 由这个脚本直接读取 Stage 0–5 的 JSON 合并而成，**不经过 Stage 6**，
因此 Stage 6 是否通过不影响它能否生成。

> Preview / Strict 的全部分支、命令行开关、两个容易混淆的 retry 设置，以及
> `evidence_matcher` 的定位优先级，见 `docs/Preview分支与开关说明.md`。

### 6.2 Strict

```powershell
./run_demo20_delivery.ps1 -Strict
```

默认输出：

```text
output_strict/
```

### 6.3 强制重跑

```powershell
./run_demo20_delivery.ps1 -Force
```

`-Force` 会忽略可复用缓存并重新执行，可能增加模型费用。日常恢复失败文档时优先使用默认续跑，不要无必要地全量 `-Force`。

### 6.4 `preview/run_demo20.ps1` 何时使用

交付用户通常不需要直接使用这个组件级脚本。包根目录的 `run_demo20_delivery.ps1` 已经把配置、输入、清单和输出设置好。

`preview/run_demo20.ps1` 支持：

- `Preflight`：检查输入和配置；
- `Verify`：验收已有结果；
- `Cached`：复用缓存续跑；
- `Fresh`：空目录全新重跑。

它历史上默认使用同级 `extraction/output_test`，而交付包不包含该历史目录。因此在交付包中直接使用时，必须显式提供输入和输出：

```powershell
./preview/run_demo20.ps1 `
  -Mode Preflight `
  -ConfigPath ./extraction/config/pipeline.yaml `
  -InputDir ./sample_data/processed_documents `
  -RefList ./preview/demo_latest_20_refs.txt
```

需要续跑且允许模型调用时：

```powershell
./preview/run_demo20.ps1 `
  -Mode Cached `
  -ConfigPath ./extraction/config/pipeline.yaml `
  -InputDir ./sample_data/processed_documents `
  -OutputDir ./output_preview_advanced `
  -RefList ./preview/demo_latest_20_refs.txt `
  -AllowModelCalls
```

交付使用以本文件为准；`preview/README.md` 是组件级说明，里面若出现开发目录 `testcode`，不要直接照搬开发机绝对路径。

## 7. 从 20 篇原文 PDF 开始完整流程

Preview：

```powershell
./run_pdf_pipeline_delivery.ps1
```

Strict：

```powershell
./run_pdf_pipeline_delivery.ps1 -Strict
```

扫描型 PDF 或需要 OCR 时：

```powershell
./run_pdf_pipeline_delivery.ps1 -Ocr
```

目录：

```text
source_pdfs/                    输入 PDF
work_pdf_pipeline/mineru/       MinerU 原始结果
work_pdf_pipeline/organized/    整理后结果
work_pdf_pipeline/processed/    标准化文档
output_pdf_preview/             Preview 抽取结果
output_pdf_strict/              Strict 抽取结果
```

该方式会调用 MinerU 和抽取模型，费用和耗时通常高于直接使用标准化 JSON。

## 8. Preview 与 Strict 的区别

### 8.1 Strict：正式质量验收

Strict 保持严格数据约束，例如：

- mention 和 evidence 必须能够追溯到原文；
- 同一 mention 不能同时归属多个实体；
- block、entity、sample 等引用必须有效；
- `raw` 字段应能在指定 evidence 中定位；
- resolved 和 unresolved 不能重叠；
- Stage 6 必须通过一致性校验。

Strict 发现不满足条件的数据时会报错，不以“强行跑完”为目标。

### 8.2 Preview：局部降级后尽量完成流程

Preview 的原则是：不伪造事实、不修改数值和证据；局部对象或字段不合规时，尽量降级并记录 warning，而不是让整篇立即失败。

当前主要行为：

- Stage 1：非原文 mention 尝试确定性恢复；无法恢复时只丢弃该 mention；
- Stage 2：重复 mention 能唯一归属时自动修复，否则标记 unresolved；实体同时包含具体名称和样品代号时，优先采用有原文支持的具体名称；
- Stage 3：结构和 Process 图合法时，局部 `sample_label_raw` evidence 定位问题可保留并 warning；
- Stage 4/5：单个可选字段 evidence 无法定位时删除字段；对象整体不可信时删除对象；
- Stage 6：证据的**表示层**差异经 `evidence_matcher` 确定性定位确认后降级为 warning；无法通过的单个对象进入 `rejected_objects`，不再连坐整篇；
- Stage 6：删除对象后确定性清扫悬空引用，并输出 `preview_publication_summary` 做对象守恒检查；
- Stage 6：Characterization 方法允许表级 locator；Property 和 Series Point 仍要求单元格级定位；
- Stage 4R：恢复值的 evidence 直接使用稳定 `cell_id` 对应的 Stage 0 单元格文本；
- Stage 3：Sample 新增 `polymer_type` 和 `material_type`，无证据时保持 `null`。
- 非法 JSON：先做有限、确定性的语法修复；仍无法解析时保存原始响应，并生成 degraded 空运行视图；
- 所有恢复、删除、unresolved 和空壳结果都必须写入 warning，不允许静默放行。

Stage 6 不降低对象事实标准。以下情况不会作为有效对象发布：

- 证据内容根本不在所引 block 里（按词多重集覆盖率判定，不做模糊数值替换——`44` 不会被原文的 `446` 顶掉）；
- `cell_id` 指向的单元格与 `cell_value` 声明的值不符；
- 同一个值在表里出现多次又没有 `cell_id`，无法唯一定位（判 `ambiguous`，不猜第一个）；
- `table_locator` 指向整张表而不是某个单元格；
- Property / Series Point 只有整表级 locator、无法定位到具体格子。

能够确定性清扫的反向悬空引用（例如不存在的 `derived_property_ids`）会被删除并记录 warning；不会猜测 `prop003 → prop_s5_003`。无法安全隔离的文档级错误仍会导致整篇失败。

### 8.3 “流程跑完”不等于“数据完整”

Preview 可能产生三种质量状态：

| 状态 | 含义 |
|---|---|
| `complete` | 所有 Stage 正常生成，未发生关键降级 |
| `degraded` | 流程执行到最后，但有字段、对象或某个 Stage 使用降级结果 |
| `partial` | 某些 Stage 无法生成可供下游使用的结果 |

例如模型完整响应仍无法解析时，Preview 可能生成：

```json
{
  "document_id": "reference_no_xxx",
  "properties": [],
  "property_series": [],
  "unresolved_properties": [],
  "warnings": [
    {
      "code": "preview_raw_response_unparseable",
      "message": "模型响应无法解析，原始结果已保存，使用空运行视图继续"
    }
  ]
}
```

这表示格式可供下游读取，但该 Stage 的数据不完整，不能算 Strict 通过。

建议汇报时同时给出：

```text
流程完成数量
完整结果数量
降级结果数量
Partial 数量
Strict 通过数量
```

不能只用“20/20 跑完”代替数据质量结论。

### 8.4 聚合物名称与样品代号

Stage 2 的 `polymer_name` 用于展示实体名称，`source_names`、`resolved_from_mentions` 和 Stage 3 的 `sample_label_raw` 用于保留原文代号和追溯关系。当前确定性名称优先规则支持：

- `PC-1`、`P3` 等字母数字样品代号；
- `PTh`、`NBR` 等较短缩写；
- `8b`、`9a` 等简单数字样品号；
- `PVC/ABS/SMIA` 等共混物简称在原文明确写出 `blend`/`composite` 时补充材料类别。

只有同一实体的 resolved mention 中存在可靠 `polymer_name` 时才会替换。两字符类别代号、复杂配方编码和无法唯一判断的名称继续保留，例如 `HS`、`0-2-0-I`、`1AQA-PPDI`。流程不会根据常识翻译、扩写或猜测名称。

名称更新不改变 `entity_id`、`sample_id`、Sample→Entity 关联或 Property/Series/Point ID。重新发布 Candidate 后，具体名称会同步展示在 `candidate.json` 和 `report_candidate.html`。

### 8.5 怎么判断一份 `final.json` 是不是降级通过的

Preview 产出的 `final.json` 顶层带标记，Strict 产出的**没有**这个字段，可以直接区分：

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

带 `validation_mode: preview` 的数据属于**候选**，未经科学语义人工确认，不得直接宣称可入库。

## 9. 非法 JSON、网络错误与 failure 回放

### 9.1 非法 JSON

处理顺序：

```text
保存完整原始响应
  ↓
标准 JSON 解析
  ↓ 失败
去除代码围栏、提取唯一完整 JSON、修复非法反斜杠等确定性问题
  ↓ 仍失败
Preview：生成 degraded 空运行视图 + warning
Strict：报错
```

非法内容应保存为 `stageN_raw_response.txt`；调用元数据可单独保存为 JSON。不能把残缺响应猜测补成正式数据。

### 9.2 网络/代理错误

`ProxyError`、连接超时、连接重置、`Response ended prematurely`、HTTP 429/502/503/504 属于传输层问题，不应通过放松语义校验解决。

合理行为：

- 仅对明确可重试的传输错误有限重试；
- 指数退避；
- 流式响应中断时丢弃残片并重新请求；
- 校验错误不进行无意义的网络式盲重试；
- 重试仍失败时保存 failure，后续从失败 Stage 续跑。

### 9.3 离线回放

如果 failure 文件保存了完整模型响应，修复解析或校验逻辑后应优先离线回放，避免再次产生模型费用。

```powershell
python ./extraction/tools/replay_failures.py `
  --roots <输出目录> `
  --scratch <独立临时目录> `
  --report <报告路径>
```

默认跳过已经存在成功 Stage 产物的历史 failure；审计历史 failure 时再增加：

```text
--include-resolved
```

## 10. 输出目录和主要产物

以 `output_preview/` 为例：

```text
output_preview/
├─ reference_no_xxxxxxx/
│  ├─ stage0_blocks.json
│  ├─ stage1_mentions.json
│  ├─ stage2_entities.json
│  ├─ stage3_process.json
│  ├─ stage4_properties.json
│  ├─ stage4r_recovery.json                  Preview-only 表格召回审计/恢复记录
│  ├─ stage4_properties.recovery_preview.json Preview-only Stage 4R 合并结果
│  ├─ stage5_characterizations.json
│  ├─ final.json                             Stage 6 通过时才有；Preview 下带降级标记
│  ├─ report.html                            同上
│  ├─ candidate.json
│  ├─ report_candidate.html
│  ├─ stageN_failure.json           仅失败或历史失败时存在
│  └─ stageN_raw_response.txt       需要保留不可解析原始响应时存在
└─ _batch/
   ├─ batch_state.sqlite3
   └─ run_summary.json
```

注意：成功产物存在时，同目录旧 `stageN_failure.json` 可能只是历史记录。验收器应根据当前成功 Stage 产物判断，而不是仅看到 failure 文件就认定当前失败。

## 11. 自动化测试

测试分布：

- `extraction/tests/`：19 个 `test_*.py`，覆盖抽取核心；
- `ocr/tests/`：2 个 `test_*.py`，覆盖 OCR/标准化；
- `preview/tests/`：2 个 `test_*.py`，覆盖候选发布（含输入目录校验）和验收。

运行全部交付测试：

```powershell
python -m pytest ./extraction/tests ./ocr/tests ./preview/tests -q
```

也可以只运行与本次修改相关的测试，例如：

```powershell
python -m pytest `
  ./extraction/tests/test_stage2_polymer_entity.py `
  ./extraction/tests/test_stage4r_table_recovery.py `
  ./extraction/tests/test_table_recall_audit.py `
  -q
```

测试失败表示当前环境、依赖或代码行为与交付预期不一致；测试文件本身不是运行输出，不建议为了“精简目录”直接删除。

2026-08-10 在交付仓库根目录执行完整测试，结果为：

```text
484 passed
```

2026-08-11 增加 Stage 6 Preview 与 `evidence_matcher` 后重跑：

```text
extraction 490 passed / ocr 13 passed / preview 18 passed
```

（`preview` 由 13 增至 18，是因为新增了 `publish_candidate.py` 输入目录校验的
5 个用例，见下方 2026-08-11 条目。）

## 12. 已有验收结论

截至 2026-08-07，固定 20 篇历史结果为：

- 随机 3 篇：`3/3`；
- 随机 7 篇：`7/7`；
- 剩余 10 篇：`10/10`；
- 合计：`20/20`。

完整 Stage 0–5、Candidate 和 HTML 结果已收录于：

`batch_results/demo20_20260807/`

每篇包含 8 个最终产物，可直接用浏览器打开：

`batch_results/demo20_20260807/<reference_no>/report_candidate.html`

文献与批次映射、文件 SHA-256 和验收状态见：

- `batch_results/demo20_20260807/README.md`；
- `batch_results/demo20_20260807/RESULT_INDEX.json`。

为避免上传运行噪声，该目录不包含日志、SQLite 状态库、历史 failure JSON、retry 状态和重复资源。

另对曾未一次完成的候选随机抽取 3 篇，在全新输出目录、无缓存条件下复测，21 个 Stage attempt 全部成功，Preview `3/3`、Strict `3/3`。

打包前仓库测试为 `416 passed`；便携路径修复和交付新增测试后，包内完整测试为 `441 passed`。

以上是 2026-08-07 的历史验收结论，不代表未来每次外部 API 调用都必然成功。网络、供应商、模型版本、配置和输出随机性仍可能影响新运行结果。

### 2026-08-09 Preview 修复验证

在不放松 Strict Schema 的前提下，本次仅增强 Preview：

- Stage 4 输入补齐全部表格，并在 Methods/Results 缺失时回落非 References 正文；
- 对唯一可确定的 evidence、condition、series/point 状态和字段形态做确定性修复；
- 单个对象或字段不合法时逐对象保留，避免整篇降级为空壳；
- 降级和成功路径都保留 Stage 4 完整响应，便于离线回放；
- Stage 4 瞬时网络错误重试设为 2 次，输入字符预算提高到 110000；交付配置仍使用相对路径。

新版验证结果位于：

`batch_results/demo20_preview_20260809/`

本轮结果为：

- Stage 4 非空 `20/20`；
- `preview_degraded_empty_shell` 为 `0/20`；
- Candidate `publication.status == complete` 为 `20/20`；
- Candidate `stage_failures == []` 为 `20/20`；
- Stage 4/5 到 `candidate.json` 的 ID 和 series/points 数量对账为 `20/20`；
- Stage 4 汇总：87 conditions、61 scalar properties、35 unresolved、105 series、691 points；
- Candidate 汇总：188 property observations、77 characterizations。

相关回归测试：Stage 4、Candidate Publisher、Stage 6 共 `184 passed`。

需要注意：18 篇按 Preview 策略带有语义校验 bypass warning；本轮没有声称 Strict 全部通过，也尚未将这 20 篇全部做“从 PDF 删除中间结果后冷启动”的新一轮验收。详细限制见新批次 README 和验证报告。

### 2026-08-10 Stage 2 / Stage 4R 代码验证

本次代码更新包括：

- Stage 2 implementation `1.3.5`、Prompt `1.2.1`：具体聚合物名称优先于样品代号，同时保留原代号和全部引用关系；
- Stage 4R：对 `0-2-0-6` 等数字连字符样品编码使用未过滤行标签做严格等值归属，不进行高风险包含匹配；
- 表格召回审计：扩展热分解温度、黏度、冲击、拉伸、弯曲和硬度别名，支持 LaTeX 误差/度数/乘号数值及三级行表头；
- 性质别名归一名必须存在于 `property_vocabulary`，避免词表外名称被下游静默丢弃。

交付仓库完整测试为 `484 passed`。本次提交更新代码与文档，仓库自带的历史结果目录仍为 `demo20_20260807` 和 `demo20_preview_20260809`；使用当前代码运行 Preview 时会生成新的 Stage 4R 产物。

### 2026-08-11 Stage 6 Preview 代码验证

本次改动只新增 Preview 分支和 `evidence_matcher` 模块，Strict 分支代码未改动。

在开发机 20 篇上做的对照验证（不调模型，仅重跑 Stage 6）：

- **Strict：改动前后判定逐条一致**，通过数 `0/20` 不变，错误码分布也不变
  （`evidence_not_in_source` 294、`table_locator_not_in_source` 81、
  `invalid_table_locator` 16、`unknown_property_reference` 12）；
- **Preview：`0/20` → `12/20` 通过**，产出 12 份 `final.json` 和 `report.html`，
  全部带 `validation_mode: preview` / `validation_status: degraded`；
- 对全部 2809 条 evidence 单独跑 matcher：旧检查已通过 2277 条不受影响（回归 0），
  旧检查失败的 532 条中救回 520 条（97.7%），其余 9 条 `unresolved`、3 条 `ambiguous`；
- **12 条经核对确认是真的错**（引用 block 里根本没有该数值），保持判 error，未降级；
- 数量守恒对账：Stage 4 恢复前 96 + Stage 4R 123 = 恢复后 219；4R 的 123 个 `cell_id`
  全部进入 candidate（缺失 0）；`property_series` 105 条前后一致且无重编号；异常 0 条。

仍判 error 的残留问题**没有**被降级掩盖，属于待修的真实缺陷：

- `unknown_property_reference` 12 条 / 3 篇：Stage 5 的 `derived_property_ids` 引用了
  任何 Stage 都不存在的 `prop011`…`prop045`，属于 Stage 5 输出问题，Stage 4R 前后一致；
- `table_locator_not_in_source` 10 条、`invalid_table_locator` 4 条：locator 指向整张表
  或坐标字段全为 null；
- `evidence_not_in_source` 9 条：证据内容确实不在所引 block 内。

本条结论只覆盖 Stage 6 校验行为，不代表重新调用模型抽取时结果不变。

### 2026-08-11 `publish_candidate.py` 输入校验修复

`publish_candidate.py` 此前对不存在的 `--ref-no` **不报错**：它会按传入的名字
新建输出目录，写出一份 0 条 `property_observations` 的 `candidate.json`，
并以 exit code 0 退出。最常见的触发方式是漏掉 `reference_no_` 前缀
（传 `0020284` 而不是 `reference_no_0020284`）。

静默产出错误结果比直接失败更危险——看起来"跑成功了"，实际数据是空的。

现在的行为：

- 输入目录不存在时抛 `CandidatePublishError`，CLI 以非 0 exit code 退出；
- **不创建输出目录，不写任何 candidate 文件**；
- 若只是漏了前缀、加上前缀后目录确实存在，错误信息会直接给出正确写法。

不受影响的行为：目录存在、只是缺少某几个 Stage 文件时，仍按原样发布
candidate（`candidate_partial`），这是 Preview 的既定设计。

### 2026-08-12 A 期与 Stage 3 类型字段验证

本次先完成 Preview A 期和 B 期第一项，不包含正文 fallback、Stage 4 Prompt 增强或 Caption 主体恢复：

- Stage 6 Preview：逐对象隔离、引用清扫、对象守恒统计；
- locator 分级：Characterization 可表级，Property / Series Point 保持单元格级；
- Stage 4R：按稳定 `cell_id` 写入精确单元格 evidence；
- Stage 3：新增 `polymer_type` / `material_type`，并升级 Stage 3 Schema、Prompt 和实现版本；旧 Stage 3 缓存不再静默复用。

在开发机已有 20 篇 Stage 0–5 / Stage 4R 产物上，仅离线重跑 Stage 6（不调用模型）：

- `final.json`：`20/20`；
- `report.html`：`20/20`；
- Stage 6 errors：每篇均为 `0`；
- 发布对象：`3485`；
- 隔离对象：`15`；
- 清扫悬空引用：`78`；
- 对象守恒：`20/20` 通过。

完整测试：`513 passed`。规范化后的 20 篇结果已作为独立数据提交发布到
`batch_results/demo20_preview_final_20260812/`，没有与功能代码提交混合。

新增 5 个用例（`preview/tests/test_publish_candidate.py`），并通过移除守卫的
反向对照确认这些用例确实能捕获该缺陷。

## 13. 便携配置和路径规则

包内 `extraction/config/pipeline.yaml` 使用交付相对路径：

```text
paths.input_dir: sample_data/processed_documents
paths.output_dir: output
paths.source_root: source_pdfs
Stage 4/5 词表: extraction/config/polymer_schema.yaml
```

根目录两个 PowerShell 入口会自动切换到交付包根目录，因此推荐直接运行它们。

如果直接调用 Python：

1. 先 `cd` 到交付包根目录；或
2. 通过命令行显式传入配置、输入和输出路径。

不要依赖开发机上的：

```text
D:/1work/1_2026/polymer/testcode/...
```

交付包移动到其他磁盘或目录后，根目录入口仍应使用包内相对路径运行。

## 14. 完整性校验

```powershell
python ./verify_delivery.py
```

检查交付 ZIP 本身：

```powershell
Get-FileHash -Algorithm SHA256 <交付ZIP路径>
```

`MANIFEST.json` 记录包内文件大小和 SHA-256；`SHA256SUMS.txt` 便于逐文件核验。Manifest 不记录自身和 `SHA256SUMS.txt`，用于避免自引用哈希。

修改交付包内任意文件后，原 Manifest 和 SHA256 清单将不再完全匹配；正式重新打包前应重新生成完整性清单。

## 15. 常见问题

### 为什么有很多 test 文件？

因为这是多 Stage 流程，每个 Stage 和公共组件都需要独立回归测试。测试不会自动运行，也不会进入论文结果。

### `preview/` 是前端页面吗？

不是。它是演示编排、候选发布、HTML 生成和验收工具。真正的抽取逻辑在 `extraction/`。

### Preview 跑完是否表示数据完整？

不一定。必须检查 `warnings`、`publication.status`、`stage_failures` 和 degraded 状态。Strict 通过才代表满足严格约束。

### 可以直接删除 tests 吗？

运行时通常不依赖 tests，但交付和后续维护建议保留。删除后将失去离线验证能力，并导致 Manifest 哈希变化。

### 可以直接运行 `preview/run_demo20.ps1` 吗？

可以，但交付环境必须显式提供 `-InputDir` 和 `-OutputDir`。普通用户优先使用根目录 `run_demo20_delivery.ps1`。

### 为什么看到历史 failure 文件？

续跑和离线回放会保留审计记录。应同时检查对应成功 Stage 文件和验收报告，不能只根据 failure 文件名判断当前状态。

## 16. 进一步说明

更详细的架构、阶段输入输出、现状和风险见：

- `docs/Preview分支与开关说明.md`（Preview / Strict 全部分支和开关的逐项对照）
- `docs/项目说明与文件清理建议.md`
- `docs/候选演示流程_20篇现状与重跑风险.md`
- `preview/README.md`（组件级说明；交付运行路径以本文件为准）
