# 聚合物文献抽取流程交付包

首次交付日期：2026-08-07

最近更新日期：2026-08-10

固定数据集：20 篇文献

运行环境：Windows PowerShell + Python 3.12（建议）
交付目标：既可从随包的标准化文档直接运行 Stage 0–6，也可从随包 PDF 启动 MinerU OCR/解析和抽取全流程。

> 交付用户优先使用包根目录的 `run_demo20_delivery.ps1` 和 `run_pdf_pipeline_delivery.ps1`。`preview/run_demo20.ps1` 是组件级高级入口，详见第 6 节。

> **2026-08-09 Preview 更新：** Stage 4 已改为尽量确定性修复并逐对象保留合法数据，避免单个格式错误导致整篇清空。新版 20 篇结果位于 `batch_results/demo20_preview_20260809/`；旧批次保留用于对照。
>
> **2026-08-10 Stage 4R 更新：** Preview 在 Stage 4 与 Stage 5 之间增加确定性表格补抽。Stage 4R 按稳定 `cell_id` 恢复明确缺失的表格性质；无法唯一归属的值保留为 unresolved，不随意绑定实体。Strict 流程不执行 Stage 4R。

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
| `extraction/stages/stage4r_table_recovery.py` | Preview-only 表格缺口恢复，按 `cell_id` 合并并保留歧义项 |
| `extraction/stages/table_recall_audit.py` | 单元格级表格召回审计，不依赖性质名白名单 |
| `extraction/prompts/` | 各 Stage Prompt |
| `extraction/schema/` | Pydantic/JSON 数据结构 |
| `extraction/stages/` | Stage 0–6 实现 |
| `extraction/reports/` | HTML 报告资源和渲染代码 |
| `extraction/tools/` | failure 离线回放等维护工具 |
| `extraction/tests/` | 核心抽取代码的自动化回归测试 |

### 2.2 `extraction/tests/`：自动化测试，不是运行数据

该目录包含 18 个 `test_*.py` 和一个公共辅助文件 `helpers.py`。文件较多是因为 Stage 0–6、模型客户端、批处理器、HTML、缓存和失败回放都分别有行为测试。

典型文件：

- `test_stage1_material_mention.py`：mention 提取、原文定位和 Preview 降级；
- `test_stage2_polymer_entity.py`：实体解析、重复 mention 和 unresolved；
- `test_stage3_sample_process.py`：样品标签、Process 图和 evidence；
- `test_stage4_property.py`：性能、条件、单位和 evidence；
- `test_stage5_characterization.py`：表征数据；
- `test_stage6_validate_merge.py`：严格合并和一致性校验；
- `test_llm_client.py`：JSON 解析、非法转义修复和传输错误；
- `test_batch_runner.py`：批处理、缓存、续跑、partial、Stage 4R Preview 编排和退出码；
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
Preview：Stage 0 → 1 → 2 → 3 → 4 → 4R → 5
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
- Stage 2：重复 mention 能唯一归属时自动修复，否则标记 unresolved；
- Stage 3：结构和 Process 图合法时，局部 `sample_label_raw` evidence 定位问题可保留并 warning；
- Stage 4/5：单个可选字段 evidence 无法定位时删除字段；对象整体不可信时删除对象；
- 非法 JSON：先做有限、确定性的语法修复；仍无法解析时保存原始响应，并生成 degraded 空运行视图；
- 所有恢复、删除、unresolved 和空壳结果都必须写入 warning，不允许静默放行。

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
│  ├─ stage5_characterizations.json
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

- `extraction/tests/`：16 个 `test_*.py`，覆盖抽取核心；
- `ocr/tests/`：2 个 `test_*.py`，覆盖 OCR/标准化；
- `preview/tests/`：2 个 `test_*.py`，覆盖候选发布和验收。

运行全部交付测试：

```powershell
python -m pytest ./extraction/tests ./ocr/tests ./preview/tests -q
```

也可以只运行与本次修改相关的测试，例如：

```powershell
python -m pytest `
  ./extraction/tests/test_llm_client.py `
  ./extraction/tests/test_stage3_sample_process.py `
  -q
```

测试失败表示当前环境、依赖或代码行为与交付预期不一致；测试文件本身不是运行输出，不建议为了“精简目录”直接删除。

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

- `docs/项目说明与文件清理建议.md`
- `docs/候选演示流程_20篇现状与重跑风险.md`
- `preview/README.md`（组件级说明；交付运行路径以本文件为准）