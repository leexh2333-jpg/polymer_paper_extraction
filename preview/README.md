# Preview 候选输出：固定 20 篇可运行流程

Preview 模式复用标准化输入和 Stage 0–5 抽取，最后发布：

```text
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5
                                             ↓
                              candidate.json + report_candidate.html
```

## 交付内容

- `run_demo20.ps1`：固定 20 篇的一键入口；
- `demo_latest_20_refs.txt`：唯一的 20 篇清单；
- `verify_demo20.py`：严格验收 Stage 0–5、candidate 和 HTML；
- `extraction/batch_runner.py`：断点续跑、failure 优先离线回放、最终退出码；
- `extraction/tools/replay_failures.py`：离线回放未解决 failure。

默认配置文件：

```text
extraction/config/pipeline.yaml
```

脚本不会修改配置，也不会输出 API key。Stage 1 的历史响应仅在当前输入为单 chunk 时允许离线回放；多 chunk 会拒绝回放并回到正常模型调用，防止把最后一次响应错误复用于所有 chunk。

## 1. 环境与输入检查（不调用模型）

```powershell
powershell -ExecutionPolicy Bypass -File `
  ./preview/run_demo20.ps1 `
  -Mode Preflight
```

通过条件：

- `pipeline.yaml` 可解析；
- ref-list 恰好 20 个且无重复；
- 20 个 `reference_no_*_document.json` 全部存在；
- Python 可执行。

## 2. 验收当前基线结果（不调用模型）

未指定 `-OutputDir` 时，`Verify` 验收：

```text
output
```

命令：

```powershell
powershell -ExecutionPolicy Bypass -File `
  ./preview/run_demo20.ps1 `
  -Mode Verify
```

退出码 0 仅表示以下条件全部满足：

- 固定清单正好 20 篇；
- 每篇 Stage 0–5 JSON 都存在、可解析且 `document_id` 一致；
- 每篇 `candidate.json` 为 `publication.status=complete`；
- 每篇 `stage_failures=[]`，且至少有一类结构化结果；
- 每篇 `report_candidate.html` 存在且非空。

已有成功 Stage 产物时，同目录的旧 `stage*_failure.json` 只记为历史文件，不判失败。

## 3. 断点续跑/缓存运行（可能调用模型）

`Cached` 会复用仍然有效的 Stage 缓存，但模型配置、Prompt、输入或实现版本变化时，相应 Stage 可能重新调用模型。因此必须显式提供输出目录和 `-AllowModelCalls`。

```powershell
powershell -ExecutionPolicy Bypass -File `
  ./preview/run_demo20.ps1 `
  -Mode Cached `
  -OutputDir ./output_demo20 `
  -AllowModelCalls
```

脚本固定使用 1 个文档 worker 和 1 个 LLM worker；如确认服务限流允许，可显式调整：

```powershell
-Workers 2 -LlmWorkers 1
```

## 4. 全新重跑（会调用模型）

`Fresh` 会传递 `--force`，要求输出目录不存在或为空，并禁止使用基线 `output_test`：

```powershell
powershell -ExecutionPolicy Bypass -File `
  ./preview/run_demo20.ps1 `
  -Mode Fresh `
  -OutputDir ./output_demo20_fresh `
  -AllowModelCalls
```

> Fresh 会产生实际模型费用。在未实际执行 Fresh 前，只能声明“代码和离线测试通过”，不能声明“新配置下 20/20 Fresh 已通过”。

## 5. 输出与退出码

每次 `Cached` / `Fresh` 完成后都会自动运行严格验收器，并写入：

```text
<OutputDir>\_batch\demo20_state.sqlite3
<OutputDir>\_batch\demo20_run_summary.json
<OutputDir>\_batch\demo20_verify_report.json
```

退出码：

- `0`：批处理接受且严格验收 20/20 通过；
- `1`：任一篇 partial、failed、缺 Stage、缺 candidate/HTML 或文档 ID 不一致；
- PowerShell 异常：路径、配置、输入或安全前置条件不满足。

## 6. 单独使用验收器

```powershell
python ./preview/verify_demo20.py `
  --ref-list ./preview/demo_latest_20_refs.txt `
  --output-dir ./output `
  --expected-count 20 `
  --report-out ./output/_batch/demo20_verify_report.json
```

## 7. 离线检查未解决 failure

默认跳过已经存在成功 Stage 产物的历史 failure：

```powershell
python ./extraction/tools/replay_failures.py `
  --roots <输出目录> `
  --scratch <独立临时目录> `
  --report <报告路径>
```

需要审计历史 failure 时额外传入：

```powershell
--include-resolved
```

该工具只使用 failure JSON 中保存的响应，不调用模型。
