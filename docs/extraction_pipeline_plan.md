# 高分子文献结构化抽取 Pipeline 设计方案

> 版本：v1.7 | 日期：2026-07-29
> OCR 实现：MinerU 批处理脚本（已有，可直接复用）
> 参考实现：`D:\1work\1_2026\prompt\article skills_V5`、`D:\1work\1_2026\2d\code`
> 建模规范：`D:\1work\1_2026\polymer\docs\高分子文献数据库建模与实施补充方案.md`

---

## 1. 总体架构

### 端到端三段式流程

```
PDF 文件
  ↓
【阶段 1: OCR】（已有实现；当前样本已完成）
  工具：code/ocr/mineru_batch_parse.py
  配置：code/ocr/.env（MINERU_API_KEY）+ requirements.txt
  输出：mineru_output/{ref_no}/
          ├── {uuid}_content_list.json
          ├── {uuid}_content_list_v2.json
          ├── {ref_no}.md
          ├── {uuid}_origin.pdf
          └── {ref_no}_images/*.jpg
        mineru_output/
          ├── batch_{batch_id}_manifest.json
          └── batch_{batch_id}_status.json

  ↓
【阶段 2: 预处理标准化】（新增）
  工具：stage_minus1_reorganize_mineru.py + transform_mineru_to_standard.py
  输出：wenxian/{ref_no}/
          ├── content.json
          ├── origin.pdf
          └── images/*.jpg
        processed_data/documents/
          └── {ref_no}_document.json

  ↓
【阶段 3: 结构化抽取】
  工具：polymer/code/extraction/
  Stage 0: 加载标准化 document JSON
  Stage 1: MaterialMention 识别
  Stage 2: PolymerEntity 构建
  Stage 3: Sample + ProcessStep 抽取
  Stage 4: PropertyObservation + MeasurementCondition 抽取
  Stage 5: Characterization 抽取
  Stage 6: 合并、引用完整性校验与 warning 汇总
  输出：extraction/output/{ref_no}/final.json
```

### 架构选型说明

| 维度 | **本方案** |
|------|----------|
| OCR / PDF 解析 | **复用 `mineru_batch_parse.py` 与现有 `.env` 配置** |
| 输入格式 | `.md` 正文 + `_content_list_v2.json` 结构 + `_content_list.json` 兼容 |
| Prompt 管理 | **独立 `.md` 文件（可见可改）** |
| 阶段结构 | **stage0→1→2→3→4→5→6** |
| LLM 客户端 | **借鉴 V5 的重试与响应解析，按本项目配置适配** |
| 质量控制 | **Stage 6 硬校验 + warning + 人工审核接口** |
| 输出格式 | **阶段性 JSON + final.json** |

**实体、工艺、性质和表征均纳入正式 Pipeline；Stage 4/5 先实现可审核的基础版本，不作为远期预留。**

---

## 2. 数据模型

参考建模文档 §3，正式 Pipeline 使用以下核心对象：

| 对象 | 说明 | 对应抽取阶段 |
|------|------|-------------|
| `Paper` | 文献书目元信息：DOI、题目、期刊、年份、作者、PDF 文件名（LLM 从首页提取，预处理阶段写入 document.json） | 预处理 |
| `MaterialMention` | 原文中的聚合物名称/缩写/样品编号 | Stage 1 |
| `PolymerEntity` | 规范化化学定义（化学形态、结构特征） | Stage 2 |
| `Sample` | 真实批次/加工状态（7 种 sample_kind） | Stage 3 |
| `ProcessStep` | 合成/加工步骤，支持多输入多输出（DAG） | Stage 3 |
| `PropertyObservation` | 性质原始值、规范值、单位及样品关联 | Stage 4 |
| `MeasurementCondition` | 温度、湿度、频率等测量条件 | Stage 4 |
| `Characterization` | FTIR、NMR、GPC、DSC、TGA、XRD、SEM、TEM 等表征方法记录；Stage 5（Option B）同步输出光谱/结构数值 PropertyObservation（`property_category: composition_structure / morphology`） | Stage 5 |
| `Evidence` | 原文位置（block_id + page + bbox） | 各阶段输出时附带 |
| `Provenance` | OCR 批次/模型/选项，以及各 LLM 阶段实际模型、prompt 文件哈希和运行时间（Stage 6 汇总到 final.json） | OCR + Stage 6 |

所有对象允许保留 `unresolved` 状态。不得根据常识补齐文中未报告的样品归属、测量条件、数值、单位或结构信息。

### Sample kind 受控词表

```
synthesis_batch     合成批次
commercial_batch    商购物料
intermediate        中间体
processed_material  加工材料（成膜/纺丝/热压等）
conditioned_state   预处理状态（干燥/水化/预平衡）
test_specimen       测试试样
post_test_state     测试后状态
```

### 判断逻辑（来自建模文档 §4）

```
1. 共价结构/重复单元/连接方式改变？→ 新建 PolymerEntity + Sample
2. 化学形态/反离子改变？→ 新建 PolymerEntity variant + Sample
3. 新批次/配方/加工/预处理状态？→ 新建 Sample（通过 ProcessStep 连接）
4. 只改变测量条件？→ 只新建 PropertyObservation + MeasurementCondition
5. 信息不足？→ 保留 MaterialMention，标记 unresolved
```

---

## 3. 目录结构

### 预处理目录

```
polymer/code/
├── ocr/
│   ├── .env                           # 已有：本地配置，只读取 MINERU_API_KEY
│   ├── requirements.txt               # 已有：requests 依赖
│   ├── mineru_batch_parse.py          # 已有：上传、轮询、下载和安全解压
│   ├── tests/
│   │   └── test_mineru_batch_parse.py # 已有：环境变量、上传重试、安全解压等测试
│   └── transform_mineru_to_standard.py # 新增：语义标准化与首页元数据提取
└── extraction/
    └── stage_minus1_reorganize_mineru.py # 已有：PDF、图片和 MinerU 内容整理
```

职责边界：

- `mineru_batch_parse.py` 负责 PDF 收集、MinerU 批量上传、状态轮询、结果下载与安全解压；不承担文献实体或性质抽取。
- `stage_minus1_reorganize_mineru.py` 只负责复制、重命名和整理 PDF、图片及 MinerU 原始产物。
- `transform_mineru_to_standard.py` 负责融合 Markdown 正文与 MinerU 结构，生成抽取使用的标准化 document JSON。
- 三个脚本之间通过目录和 JSON 产物交接，不重复实现 OCR、素材复制或结构化抽取。
- `.env` 仅供本地运行读取，API 密钥不写入 pipeline.yaml、日志、manifest 或任何输出 JSON。

### 抽取项目目录

```
polymer/code/extraction/
├── config/
│   ├── pipeline.yaml              # 参考 V5 结构，使用本项目路径和阶段配置
│   ├── llm_models.yaml            # 参考 V5 配置结构，按本项目模型适配
│   └── polymer_schema.yaml        # 受控词表（性质、单位、工艺、表征方法）
│
├── prompts/                       # 每阶段独立 prompt（.md 格式，可见可改）
│   ├── meta_extract.md            # 元数据提取（DOI、题目、期刊、年份、作者）
│   ├── stage1_material_mention.md
│   ├── stage2_polymer_entity.md
│   ├── stage3_sample_process.md
│   ├── stage4_property.md
│   ├── stage5_characterization.md
│   └── common_guardrails.md       # 公共约束（证据、语言、不翻译）
│
├── stages/
│   ├── stage0_load_document.py    # 读取 document JSON → element list
│   ├── stage1_material_mention.py
│   ├── stage2_polymer_entity.py
│   ├── stage3_sample_process.py
│   ├── stage4_property.py
│   ├── stage5_characterization.py
│   └── stage6_validate_merge.py
│
├── schema/
│   └── polymer_schema.py          # 核心对象的 Pydantic 模型
│
├── prompt_loader.py               # 按 prompt_id 加载、拼装、渲染和计算哈希
├── llm_client.py                  # 借鉴 V5，保留本项目独立配置和错误类型
├── orchestrator.py                # 串行调度 stage0→1→2→3→4→5→6
├── main.py                        # CLI 入口
└── requirements.txt
```

### 素材整理目录（现有 Stage -1 输出）

```
polymer/wenxian/
└── reference_no_0001016/
    ├── content.json
    ├── origin.pdf
    └── images/*.jpg
```

### 标准化文档目录（新建）

```
polymer/processed_data/
└── documents/
    ├── reference_no_0001016_document.json
    └── ... (30 个 JSON)
```

### 抽取输出目录

```
polymer/code/extraction/output/
└── reference_no_0001016/
    ├── stage0_blocks.json         # 标准化 block 列表
    ├── stage1_mentions.json       # MaterialMention
    ├── stage2_entities.json       # PolymerEntity
    ├── stage3_process.json        # Sample + ProcessStep
    ├── stage4_properties.json     # PropertyObservation + MeasurementCondition
    ├── stage5_characterizations.json # Characterization
    ├── stage6_validation.json     # 引用完整性、DAG、证据等校验结果
    └── final.json                 # 合并结果
```

---

## 4. 阶段 1-2：OCR 与预处理标准化

### 阶段 1：PDF OCR / 版面解析（已有）

**脚本**：`code/ocr/mineru_batch_parse.py`

该脚本作为整个 Pipeline 的正式入口，直接复用现有实现和配置，不在抽取模块中重复封装 MinerU API。

**输入契约**：

- `--input-dir` 指向包含 PDF 的目录；脚本只扫描该目录第一层的 `*.pdf`。
- 单批最多 200 篇，超过上限时应由调用层分批，不修改脚本内的 MinerU 限制。
- 推荐输入 PDF 使用 `reference_no_*.pdf` 命名，使解压目录可直接作为下游 `ref_no`。其他命名必须保留 manifest 映射，不能在下游凭目录顺序推断文献编号。

**现有配置**：

- 依赖安装：`code/ocr/requirements.txt`
- 默认环境文件：`code/ocr/.env`
- 密钥字段：`MINERU_API_KEY`
- 进程中已存在的环境变量优先于 `.env`，`.env` 不覆盖已有值。
- 默认 MinerU 模型为 `vlm`；公式和表格解析默认开启。

`.env` 内容不得复制进本方案、pipeline.yaml、命令行示例、日志或输出 JSON。

**运行方式**：

```powershell
# 安装 OCR 脚本现有依赖
python -m pip install -r "D:\1work\1_2026\polymer\code\ocr\requirements.txt"

# 数字 PDF：使用脚本默认解析模式
python "D:\1work\1_2026\polymer\code\ocr\mineru_batch_parse.py" `
  --input-dir "D:\1work\1_2026\polymer\polyinfo数据\sample_exprot_34" `
  --output-dir "D:\1work\1_2026\polymer\polyinfo数据\sample_exprot_34\mineru_output"

# 扫描 PDF：显式启用 OCR
python "D:\1work\1_2026\polymer\code\ocr\mineru_batch_parse.py" `
  --input-dir "<扫描PDF目录>" `
  --output-dir "<MinerU输出目录>" `
  --ocr

# 超时或中断后，使用已有 batch_id 继续查询和下载，不重复上传
python "D:\1work\1_2026\polymer\code\ocr\mineru_batch_parse.py" `
  --output-dir "<原MinerU输出目录>" `
  --batch-id "<batch_id>"
```

数字 PDF 与扫描 PDF 混合时应拆成两个批次，分别使用默认模式和 `--ocr`，避免对已有文本层的 PDF 无差别重复 OCR。

**输出契约**：

```text
mineru_output/
├── batch_{batch_id}_manifest.json
├── batch_{batch_id}_status.json
└── {ref_no}/
    ├── {uuid}_content_list.json
    ├── {uuid}_content_list_v2.json
    ├── {ref_no}.md
    ├── {uuid}_origin.pdf
    └── {ref_no}_images/*.jpg
```

现有 manifest 已记录 `batch_id`、输入目录、文件列表和 `model_version`。为支持完整 OCR Provenance，实施时对 manifest 做最小字段补充：记录 `ocr_enabled`、`language`、`page_ranges`、`enable_formula`、`enable_table` 和 `extra_formats`；不改变上传、轮询或下载行为。

**完成门禁与恢复规则**：

1. `batch_*_status.json` 中单篇状态为 `done`，且解压目录存在，才进入阶段 2。
2. 每篇至少要求 `{uuid}_content_list.json` 和 `{uuid}_origin.pdf`；当前标准化方案还需要 `.md`，缺失时该篇停止并写入错误。
3. `content_list_v2.json` 或图片缺失时按实际引用决定：未使用可 warning，正文或表格明确引用时转为错误。
4. 批次包含 failed 文献时，已完成文献可以继续下游；失败文献保留错误原因，不阻塞其他文献。
5. 已有完整 MinerU 产物时，端到端调度直接跳过 OCR。现有脚本本身没有“按输出目录自动跳过上传”的参数，不应在无检查情况下重复调用。
6. 轮询超时或进程中断时，从 manifest/status 取得 `batch_id`，使用 `--batch-id` 恢复。

当前 `sample_exprot_34/mineru_output` 已有 OCR 产物，执行本方案时从完成门禁开始检查，不重复调用 MinerU。

### 阶段 2A：素材整理：`stage_minus1_reorganize_mineru.py`

沿用现有脚本整理 PDF、图片和 MinerU 内容。后续补充保留同篇文献的 Markdown 与 `content_list_v2.json`，但不在该脚本内实现章节重建或抽取逻辑。

### 阶段 2B：语义标准化：`transform_mineru_to_standard.py`

**输入**：

- `{ref_no}.md`：正文、标题和自然段的主要来源
- `{uuid}_content_list_v2.json`：页面、段落、表格和图片结构的主要来源
- `{uuid}_content_list.json`：兼容输入，并补充表格、图片等原始字段
- `{uuid}_origin.pdf` 与图片目录：保留原始资料

### 首页元数据提取：`extract_paper_meta()`

在生成 document.json 时调用 LLM，从首页文本提取文献书目元数据并写入顶层 `paper`。

**处理流程**：

```text
content_list.json v1
  → 使用原数组下标生成 block_index
  → 过滤 page_idx <= 1 的元数据候选 block
  → 按 page_idx + block_index 排序并拼接为纯文本
  → LLM 调用（stage_id: meta_extract，prompt: prompts/meta_extract.md）
  → 校验固定 5 字段 JSON
  → 写入 document.json["paper"]
```

元数据候选 block 包括：

- `type=text`，其中标题是 `type=text + text_level`，不是独立的 `title` type。
- `type=header`；若 MinerU 版本输出 footer 类 block，也一并保留，因为 DOI 可能位于页眉或页脚。
- 排除 `page_number`、表格、图片、公式和参考文献。该过滤只用于元数据提取，不改变正文标准化规则。

**Prompt 文件**：`prompts/meta_extract.md`

Prompt 要求只输出 JSON，且 LLM 只负责以下 5 个字段：

```json
{
  "doi": "10.xxxx/xxx or null",
  "title": "original title or null",
  "authors": ["original author names"],
  "journal": "original journal name or null",
  "year": 1969
}
```

Prompt 规则：

- 只提取输入原文明确出现的信息，不使用外部知识，不猜测。
- 缺失的 `doi`、`title`、`journal`、`year` 填 `null`，作者缺失时填 `null`，不得填空数组伪装为已识别。
- DOI 仅提供格式提示 `10.xxxx/xxx`，不能仅因字符串形似 DOI 就修改原文。
- 扫描首页及第二页中的页眉、页脚和书目信息区域。
- 返回内容必须通过 JSON 解析和字段类型校验；代码围栏由客户端统一剥离。

`ref_no`、PDF 文件名、状态和提取记录由程序生成，不交给 LLM：

```json
"paper": {
  "ref_no": "reference_no_0001016",
  "pdf_filename": "afad6fc5-94c6-4896-81c6-cbb3107cca3c_origin.pdf",
  "source_pdf_path": "mineru_output/reference_no_0001016/afad6fc5-94c6-4896-81c6-cbb3107cca3c_origin.pdf",
  "organized_pdf_path": "wenxian/reference_no_0001016/origin.pdf",
  "doi": null,
  "title": "Cohesive Energy Density of cis-Polybutadiene",
  "authors": ["Sant K. BHATNAGAR"],
  "journal": "Die Makromolekulare Chemie",
  "year": 1969,
  "metadata_status": "partial",
  "metadata_extraction": {
    "method": "llm",
    "model": "claude-sonnet-5",
    "prompt_file": "prompts/meta_extract.md",
    "source_pages": [0, 1]
  }
}
```

`pdf_filename` 始终记录 MinerU 的 `{uuid}_origin.pdf` 原始文件名，不改成 `origin.pdf`。`source_pdf_path` 和 `organized_pdf_path` 分别记录原始位置与 Stage -1 复制位置，便于溯源。

`metadata_status` 使用三态：

- `complete`：5 个 LLM 字段均非 null，且通过类型校验。
- `partial`：返回有效 JSON，但至少一个字段为 null。
- `failed`：LLM 调用失败、响应不是有效 JSON，或固定字段校验失败。此时保留全 null 的 5 字段和错误 warning，document.json 仍可生成。

**缓存机制**：

- document.json 已存在且 `paper.metadata_status == "complete"` 时，复用已有 `paper`，不重复调用元数据 LLM；正文和表格仍可重新标准化。
- `partial` 和 `failed` 默认在下次运行时重试。
- `--force-meta` 仅强制重新提取 `paper`，用于 DOI 或其他书目信息更正，不强制重跑后续抽取阶段。
- 批量模式扫描输入目录下的 `reference_no_*` 子目录，不硬编码篇数；当前样本目录实测为 30 篇。除命中 complete 缓存外，每篇约调用 1 次元数据 LLM。

**输出**：
- `processed_data/documents/{ref_no}_document.json`

PDF、图片及整理后的 MinerU 内容继续使用 `wenxian/{ref_no}/` 下的 Stage -1 产物，转换脚本不重复复制。

**融合原则**（基于实际文件验证，决策 1 Option A）：

经检查实际文件，`content_list.json` v1 的 text block **已经是完整段落**（非碎片），每块包含完整正文 + bbox + page_idx，不存在碎片化问题。三个来源按内容类型分工：

| 内容类型 | 主要来源 | 原因 |
|---------|---------|------|
| 正文段落 / 标题 | `content_list.json` v1 | 完整段落 + bbox + page_idx，直接用于 LLM 输入与 Evidence 定位 |
| 表格内容 | `{ref_no}.md` | HTML `<table>` 格式保留完整行列结构；v1 表格块仅为纯文本 |
| 图片路径 | `{ref_no}.md` | 含 `![](images/xxx.jpg)` 引用，直接关联 `{ref_no}_images/` 目录 |
| 内联公式 | `content_list_v2.json`，Markdown 校验/兜底 | v2 提供 `equation_inline` 及顺序；Markdown 可验证公式是否已嵌入完整句子 |
| 独立公式 | `content_list_v2.json` + v1 | v2 的 `equation_interline` 辅助判定，v1 保留公式文本、页码和 bbox |

具体融合规则：
1. **段落文本**：读取 v1 中 `type=text`（无 `text_level`）的 block，保留 bbox 用于 Evidence。
2. **标题**：读取 v1 中 `type=text` 且含 `text_level` 的 block，推断 section 标签。
3. **表格**：从 `.md` 中解析 HTML `<table>` 块，匹配 v1 对应 `type=table` block 的 bbox 作为坐标。
4. **图片**：从 `.md` 的 `![]()` 引用提取路径，关联 `{ref_no}_images/` 目录。
5. **行内公式**：按 v2 的页面、阅读顺序和 `equation_inline` 信息，将 LaTeX 合并回对应 v1 正文 block；合并后的段落仍是一个 text element。
6. **独立公式**：真正单独成行的公式保留为 equation element，不拼入前后段落。
7. 无法对齐的内容保留原文并标记 `alignment_status: unresolved`，禁止静默丢弃。

### 公式处理规则

MinerU 的 LaTeX block 不等于语义上的独立公式。预处理时区分两类：

| 类型 | 典型特征 | 标准化处理 |
|------|---------|-----------|
| 行内公式 | 位于句子内部；常见为化学式、单位、变量或短参数表达式，如 `$C_{33}H_{22}F_6N_2O_2$`、`$(cal/ml)^{1/2}$` | 合并回相邻 text element，保留 LaTeX 原文，不再单独输出 equation element |
| 独立公式 | 单独占行；包含完整等式、推导、反应式或公式编号，如 `Eq. (1)` | 输出独立 equation element，供后续阶段按需读取 |

**判定与合并顺序**：

1. v2 明确标记为 `equation_inline` 时，优先按 v2 中的阅读顺序插回对应文本。
2. v2 标记为 `equation_interline` 时，优先视为独立公式候选，再结合 v1 和上下文确认。
3. 使用 Markdown 中的完整句子验证公式前后文本和 LaTeX 顺序；v2 无法稳定匹配时，允许使用 Markdown 作为重建兜底。
4. 对 v1 的 `type=equation` block，结合是否独立成行、前后同页文本是否构成同一句、是否有公式编号等判断 display/inline。
5. 很短的 equation block 不能仅凭长度直接丢弃。只有确认它已合并进 text element、属于重复残留时才跳过独立输出。
6. 无法可靠判断时保留为 equation element，设置 `equation_kind: unresolved` 和 `alignment_status: unresolved`，同时生成 warning。

**合并后的记录要求**：

- text element 的 `text` 保留行内 LaTeX，例如：`The monomer HFBAPP ($C_{33}H_{22}F_6N_2O_2$) was dissolved in DMAc...`。
- 使用 `merged_source_block_ids` 记录参与合并的 v1/v2 block，避免后续重复抽取。
- 独立公式使用 `equation_kind: display`；已合并行内公式不再进入 Stage 1 的独立 equation 输入。

**content_list block 类型映射**：

| MinerU type | 含 text_level | → element_subtype |
|-------------|---------------|-------------------|
| `text` | 是（1/2） | `title` |
| `text` | 否 | `text` |
| `table` | - | `table` |
| `chart` | - | `image` |
| `equation` | - | inline → 合并到 `text`；display/unresolved → `equation` |
| `ref_text` | - | `references` |
| `header` | - | **跳过** |
| `page_number` | - | **跳过** |

**输出 document.json 格式**：

```json
{
  "document_id": "reference_no_0001016",
  "paper": {
    "ref_no": "reference_no_0001016",
    "pdf_filename": "afad6fc5-94c6-4896-81c6-cbb3107cca3c_origin.pdf",
    "doi": null,
    "title": "Cohesive Energy Density of cis-Polybutadiene",
    "authors": ["Sant K. BHATNAGAR"],
    "journal": "Die Makromolekulare Chemie",
    "year": 1969,
    "metadata_status": "partial"
  },
  "source_files": {
    "markdown": "reference_no_0001016.md",
    "content_v1": "content.json",
    "content_v2": "content_v2.json",
    "pdf": "origin.pdf"
  },
  "ocr": {
    "engine": "mineru",
    "batch_id": "<batch_id>",
    "model_version": "vlm",
    "ocr_enabled": false,
    "manifest_file": "batch_<batch_id>_manifest.json",
    "status_file": "batch_<batch_id>_status.json"
  },
  "elements": [
    {
      "block_id": "P_0_0",
      "page_id": 0,
      "block_index": 0,
      "element_type": "title",
      "section": "DocumentTitle",
      "text": "Cohesive Energy Density of cis-Polybutadiene",
      "bbox": [103, 105, 913, 137],
      "alignment_status": "matched"
    },
    {
      "block_id": "T_2_5",
      "page_id": 2,
      "block_index": 5,
      "element_type": "table",
      "caption": "Table 1. Swelling, viscosity and turbidiometric titration data",
      "table_body": "| No. | Solvents | δs | Vs | ...",
      "bbox": [94, 231, 896, 716],
      "image_path": "images/reference_no_0001016/43b864c1da6bd94be51a7c5098d73a7d27ff8fe2398fd1e706cf6deab5dd22a7.jpg"
    }
  ]
}
```

**运行方式**：

```bash
# 处理当前目录全部文献
python transform_mineru_to_standard.py

# 单篇调试
python transform_mineru_to_standard.py --ref-no reference_no_0001016

# 强制重新提取单篇元数据
python transform_mineru_to_standard.py --ref-no reference_no_0001016 --force-meta

# 自定义路径
python transform_mineru_to_standard.py \
  --mineru-output "D:\1work\1_2026\polymer\polyinfo数据\sample_exprot_34\mineru_output" \
  --processed-output "D:\1work\1_2026\polymer\processed_data"
```

---

## 5. 阶段 3：结构化抽取

### Stage 0：加载标准化文档（无 LLM）

**输入**：`processed_data/documents/{ref_no}_document.json`

**任务**：
1. 读取 document.json，保留顶层 `paper` 并加载 `elements`
2. 过滤 `references`（参考文献）
3. 利用 `title` element 重建 section 标签
4. 保留 table、image，以及 `equation_kind=display/unresolved` 的独立 equation element；inline equation 已合并进 text
5. 输出标准化 element 列表：

```json
{
  "document_id": "reference_no_0001016",
  "elements": [
    {
      "block_id": "P_2_3",
      "type": "text",
      "section": "Methods",
      "text": "SPAEK-NA-60 was synthesized by...",
      "page": 2,
      "bbox": [103, 265, 913, 380]
    }
  ]
}
```

**Section 映射**（用于从标题 block 推断 section 名）：

```yaml
Methods:      [experimental, methods, materials and methods, synthesis, preparation, procedure]
Results:      [results, results and discussion, discussion]
Introduction: [introduction, background]
Abstract:     [abstract, summary]
Conclusion:   [conclusion, conclusions, summary]
```

---

### Stage 1：MaterialMention 识别（LLM）

**Prompt 文件**：`prompts/stage1_material_mention.md`

**LLM 输入**：Abstract + Methods + Results 的 text/title block；行内公式已包含在 text 中，跳过独立 equation/image

**输出格式（JSONL）**：

```json
{"mention_id": "m001", "text": "SPAEK-NA-60", "mention_role": "sample_label", "evidence": {"block_id": "P_2_3", "page": 2, "source_sentence": "SPAEK-NA-60 was synthesized by..."}}
{"mention_id": "m002", "text": "SPAEK-NA", "mention_role": "polymer_name", "evidence": {"block_id": "P_0_1", "page": 0, "source_sentence": "..."}}
```

**mention_role 受控词表**：`polymer_name | abbreviation | sample_label | commercial_name`

**关键规则**：
- 只提取聚合物相关名称，排除单体/溶剂/催化剂（只作为原料出现时）
- 保留原文语言，不翻译
- 同一文本在不同位置出现 → 多个 mention（后续 entity 阶段合并）

---

### Stage 2：PolymerEntity 构建（LLM）

**Prompt 文件**：`prompts/stage2_polymer_entity.md`

**LLM 输入**：stage1_mentions.json + Methods/Results 正文 blocks

**输出格式（JSONL）**：

```json
{"entity_id": "pe001", "polymer_name": "sulfonated poly(aryl ether ketone) sodium form", "polymer_type": "random_copolymer", "variant_of": null, "representation_status": "name_only", "structural_features": ["sulfonic_acid_group", "aryl_ether_ketone_backbone"], "resolved_from_mentions": ["m001", "m002"], "evidence": {"block_id": "P_0_1", "source_sentence": "..."}}
{"entity_id": "pe002", "polymer_name": "sulfonated poly(aryl ether ketone) acid form", "polymer_type": "random_copolymer", "variant_of": "pe001", "representation_status": "name_only", "structural_features": ["sulfonic_acid_group", "aryl_ether_ketone_backbone"], "resolved_from_mentions": ["m005"], "evidence": {...}}
```

**关键字段**：

| 字段 | 说明 |
|------|------|
| `polymer_type` | `homopolymer / random_copolymer / block_copolymer / graft_copolymer / crosslinked_network / blend` |
| `variant_of` | 系列变体的父 entity_id（如 SPAEK-60 variant_of SPAEK-family） |
| `representation_status` | `structure_verified / structure_partial / monomer_defined / name_only / expert_review_required` |
| `structural_features` | 受控标签列表，不是 SMARTS |

**规则（来自建模文档 §5.3）**：
- 不自动生成 SMILES/BigSMILES，一律标记 `expert_review_required`
- 保留原始图片引用（如有）
- 同一化学定义 → 合并 mentions；不同化学形态（酸式/盐式）→ 分别建 entity + `variant_of`

---

### Stage 3：Sample + ProcessStep 抽取（LLM）

**Prompt 文件**：`prompts/stage3_sample_process.md`

**LLM 输入**：stage2_entities.json + Methods section blocks（含参数描述段落）

**输出格式（JSONL）**：

**Sample**：
```json
{"sample_id": "s001", "sample_kind": "synthesis_batch", "refers_to_entity": "pe001", "polymer_name": "SPAEK-NA-60 (sodium form)", "evidence": {"block_id": "P_3_2", "source_sentence": "..."}}
{"sample_id": "s002", "sample_kind": "processed_material", "refers_to_entity": "pe001", "polymer_name": "SPAEK-NA-60 cast film", "evidence": {...}}
{"sample_id": "s003", "sample_kind": "processed_material", "refers_to_entity": "pe002", "polymer_name": "SPAEK-NA-60 acid form film", "evidence": {...}}
{"sample_id": "s004", "sample_kind": "conditioned_state", "refers_to_entity": "pe002", "polymer_name": "hydrated acid form film", "evidence": {...}}
```

**ProcessStep**：
```json
{"step_id": "ps001", "process_type": "polymerization", "input_sample_ids": [], "output_sample_ids": ["s001"], "parameters": {"temperature": "160°C", "time": "6h", "solvent": "DMAc"}, "evidence": {"block_id": "P_3_5", "source_sentence": "..."}}
{"step_id": "ps002", "process_type": "casting", "input_sample_ids": ["s001"], "output_sample_ids": ["s002"], "parameters": {"solvent": "DMAc", "thickness": "50 μm"}, "evidence": {...}}
{"step_id": "ps003", "process_type": "ion_exchange", "input_sample_ids": ["s002"], "output_sample_ids": ["s003"], "parameters": {"reagent": "1M H2SO4", "time": "24h"}, "evidence": {...}}
{"step_id": "ps004", "process_type": "hydration", "input_sample_ids": ["s003"], "output_sample_ids": ["s004"], "parameters": {"RH": "100%", "temperature": "25°C", "time": "24h"}, "evidence": {...}}
```

**process_type 受控词表**：
`polymerization | copolymerization | blending | casting | film_formation | ion_exchange | annealing | hydration | drying | fractionation | sulfonation | crosslinking | hot_pressing | electrospinning | other`

**规则**：
- ProcessStep 支持 DAG：多输入（共混）、多输出（批次分样）均合法
- 同一个物理样品在多个性质表中出现 → 合并为一个 Sample，不重复建
- 只改变测量温度/频率 → 不新建 Sample（由 Stage 4 建立 MeasurementCondition）

---

### Stage 4：PropertyObservation + MeasurementCondition 抽取（LLM）

**Prompt 文件**：`prompts/stage4_property.md`

**LLM 输入**：

- Stage 2 的 PolymerEntity
- Stage 3 的 Sample 与 ProcessStep
- Results、Methods 中的正文和表格 element
- 表格 caption、原始 `table_body` 和必要的 footnote

**输出格式（JSONL）**：

**MeasurementCondition**：

```json
{"condition_id": "mc001", "temperature": {"raw": "35 ± 0.01 °C", "value": 35, "unit": "°C"}, "frequency": null, "humidity": null, "condition_status": "reported", "evidence": {"block_id": "T_2_5", "page": 2, "source_sentence": "Table 1 ... at 35 ± 0.01 °C"}}
```

**PropertyObservation**：

```json
{"property_id": "prop001", "sample_id": "s001", "property_name_raw": "solubility parameter", "property_name_normalized": "solubility_parameter", "value_raw": "8.5 to 8.6", "value_min": 8.5, "value_max": 8.6, "unit_raw": "(cal/ml)^1/2", "unit_normalized": "(cal/mL)^0.5", "measurement_condition_id": "mc001", "source_type": "text", "evidence": {"block_id": "P_0_5", "page": 0, "source_sentence": "The average value of the solubility parameter ... was observed to lie between 8.5 to 8.6 (cal/ml)^1/2."}}
```

**关键规则**：

- 性质必须关联到 `Sample`。原文只支持 PolymerEntity、无法确定具体 Sample 时，保留未解析引用并生成 warning，不得猜测样品。
- 同一性质在不同温度、频率、湿度或测试模式下分别建立 PropertyObservation，共享或分别关联 MeasurementCondition。
- 测量条件未报告时显式使用 `condition_status: not_reported`，不得根据测试标准或常识补齐。
- 同时保留 `value_raw`、`unit_raw` 和规范化字段；规范化不能改变原文精度、范围、上下限或约数语义。
- 表格抽取必须保留 `table_id + row_label + column_label + cell_value`，多级表头先重建层级再逐行抽取。
- 正文、表格和图注中重复报告的同一结果允许合并，但必须保留全部 evidence。

---

### Stage 5：Characterization 抽取（LLM，决策 3 Option B）

**Option B 策略**：除建立表征方法记录（Characterization）外，同步将光谱/结构确认数值抽取为 PropertyObservation，使光谱数据可数值检索，与 Stage 4 宏观性质共用同一查询接口。

**与 Stage 4 的边界**：

| Stage | 抽取目标 | 典型来源 |
|-------|---------|---------|
| Stage 4 | 宏观/体相性质：Tg、Tm、Mn、Mw、模量、导电率、溶解度参数等（17 类性质 P1110–P9160） | 结果表格、正文直接数值陈述 |
| Stage 5 | 结构确认数据：NMR 化学位移、FTIR 吸收峰、XRD 衍射峰/结晶度、SEM/TEM 形貌尺寸 | 表征描述段落、谱图说明、图注 |

> 注意：Tg（DSC 测得）→ Stage 4；Mn/Mw（GPC 测得）→ Stage 4。DSC/GPC 建 Characterization，通过 `derived_property_ids` 引用 Stage 4 已建的 PropertyObservation，不重复抽取数值。

**Prompt 文件**：`prompts/stage5_characterization.md`

**LLM 输入**：

- Stage 2 的 PolymerEntity
- Stage 3 的 Sample
- Stage 4 的 PropertyObservation（用于去重，避免重复抽取 Tg/Mn 等体相性质）
- Methods、Results 中的正文、表格、图片 caption 和 equation element

**输出格式（JSONL）**：

**Characterization（方法记录）**：
```json
{“characterization_id”: “char001”, “method_raw”: “FTIR spectroscopy”, “method_normalized”: “FTIR”, “sample_id”: “s002”, “entity_id”: “pe001”, “instrument”: null, “parameters”: {“wavenumber_range”: “4000-400 cm-1”}, “result_summary”: “Absorption bands at 1650 and 1580 cm-1 confirm sulfonation.”, “derived_property_ids”: [“prop_s5_001”, “prop_s5_002”], “evidence”: {“block_id”: “P_5_2”, “page”: 5, “source_sentence”: “...”}}
```

**PropertyObservation（光谱/结构数值，Option B 新增）**：
```json
{“property_id”: “prop_s5_001”, “sample_id”: “s002”, “property_name_raw”: “FTIR absorption band”, “property_name_normalized”: “ftir_peak_wavenumber”, “property_category”: “composition_structure”, “value_raw”: “1650”, “unit_raw”: “cm-1”, “unit_normalized”: “cm⁻¹”, “spectral_assignment”: “C=C stretching”, “source_stage”: “stage5”, “evidence”: {“block_id”: “P_5_2”, “page”: 5, “source_sentence”: “...”}}
{“property_id”: “prop_s5_002”, “sample_id”: “s002”, “property_name_raw”: “1H NMR chemical shift”, “property_name_normalized”: “nmr_chemical_shift”, “property_category”: “composition_structure”, “value_raw”: “3.82”, “unit_raw”: “ppm”, “spectral_assignment”: “ArH (sulfonated ring)”, “solvent”: “DMSO-d6”, “source_stage”: “stage5”, “evidence”: {“block_id”: “P_4_1”, “page”: 4, “source_sentence”: “...”}}
{“property_id”: “prop_s5_003”, “sample_id”: “s002”, “property_name_raw”: “XRD peak 2θ”, “property_name_normalized”: “xrd_diffraction_peak_2theta”, “property_category”: “morphology”, “value_raw”: “18.5”, “unit_raw”: “°”, “source_stage”: “stage5”, “evidence”: {“block_id”: “P_6_3”, “page”: 6, “source_sentence”: “...”}}
{“property_id”: “prop_s5_004”, “sample_id”: “s002”, “property_name_raw”: “domain size (SEM)”, “property_name_normalized”: “morphology_domain_size”, “property_category”: “morphology”, “value_raw”: “50-80”, “value_min”: 50, “value_max”: 80, “unit_raw”: “nm”, “source_stage”: “stage5”, “evidence”: {“block_id”: “P_7_1”, “page”: 7, “source_sentence”: “...”}}
```

**property_category 受控词表（Stage 5 专用）**：`composition_structure`（NMR、FTIR、Raman）、`morphology`（XRD、SEM、TEM、AFM、SAXS）

**关键规则**：

- Characterization 记录方法元数据；PropertyObservation 记录具体数值；二者通过 `derived_property_ids` 双向关联。
- `source_stage: “stage5”` 标记光谱/结构来源，与 Stage 4 体相性质区分。
- 只抽取原文明确报告的数值，不将定性描述（”broad peak around...”）转化为精确数值。
- DSC/TGA/GPC 产生的 Tg/Td/Mn 已由 Stage 4 处理，Stage 5 只建 Characterization 并通过 `derived_property_ids` 引用，不重复抽取数值。
- 优先关联到具体 Sample；只明确 PolymerEntity 时标记 `sample_resolution_status: unresolved`。
- 图片只作辅助证据；不从图片内容推断峰值，只抽取 caption 和正文中明确报告的数值。

---

### Stage 6：合并与基础校验（无 LLM）

**脚本**：`stages/stage6_validate_merge.py`

**任务**：

1. 合并 Paper 与 Stage 0-5 产物，生成统一 `final.json`。
2. 校验所有 ID 引用存在，禁止悬空的 entity、sample、condition、property 和 evidence 引用。
3. 校验 Sample/ProcessStep 构成的谱系为 DAG；循环属于硬错误，无来源输出和孤立节点进入 warning。
4. 校验 PropertyObservation 的样品、测量条件和 evidence；无法解析的情况必须有明确状态和 warning。
5. 校验 Characterization 至少关联 Sample 或 PolymerEntity，并带 evidence。
6. 校验 `source_sentence`、表格单元格来源及原始数值字段非空。
7. 校验 Paper Schema；`partial` 或 `failed` 生成 warning，但不阻断正文抽取结果。
8. 从 OCR manifest/status 汇总 batch_id、model_version、解析选项和单篇状态，并汇总各 LLM 阶段实际 provider、model、prompt 文件及运行时间到 Provenance。
9. 汇总错误与 warning 到 `stage6_validation.json`；硬错误阻止发布 final，warning 保留结果并进入人工审核。

---

### Orchestrator 调度

**`orchestrator.py`** 串行调度，每步有断点续跑：

```python
def run_pipeline(ref_no, config):
    out = output_dir / ref_no
    source_document = read_json(
        documents_dir / f"{ref_no}_document.json"
    )

    blocks = load_or_run(
        out / "stage0_blocks.json",
        lambda: load_document_elements(source_document),
    )
    mentions = load_or_run(
        out / "stage1_mentions.json",
        lambda: extract_mentions(blocks, config),
    )
    entities = load_or_run(
        out / "stage2_entities.json",
        lambda: extract_entities(mentions, blocks, config),
    )
    process_result = load_or_run(
        out / "stage3_process.json",
        lambda: extract_samples_processes(entities, blocks, config),
    )
    property_result = load_or_run(
        out / "stage4_properties.json",
        lambda: extract_properties(process_result, entities, blocks, config),
    )
    characterizations = load_or_run(
        out / "stage5_characterizations.json",
        lambda: extract_characterizations(
            process_result, entities, property_result, blocks, config
        ),
    )

    final, validation = validate_and_merge(
        source_document["paper"],
        blocks,
        mentions,
        entities,
        process_result,
        property_result,
        characterizations,
    )
    write_json(out / "stage6_validation.json", validation)
    if validation["error_count"] > 0:
        return
    write_json(out / "final.json", final)
```

**CLI 调用**：

```bash
# 单篇
python main.py --ref-no reference_no_0001016

# 批量（并发 2 篇）
python main.py --batch --workers 2

# 强制重跑某阶段及其下游阶段
python main.py --ref-no reference_no_0001016 --force-stage 4
```

### 端到端运行顺序

一期不新增第二套 OCR 客户端，按以下顺序调用现有脚本和新增抽取组件：

```powershell
# 1. OCR：新 PDF 才执行；已有完整 MinerU 产物时跳过
python "D:\1work\1_2026\polymer\code\ocr\mineru_batch_parse.py" `
  --input-dir "<PDF输入目录>" `
  --output-dir "<MinerU输出目录>"

# 2. 整理 MinerU 产物
python "D:\1work\1_2026\polymer\code\extraction\stage_minus1_reorganize_mineru.py" `
  --input "<MinerU输出目录>" `
  --output "D:\1work\1_2026\polymer\wenxian"

# 3. 生成 document.json，并提取 Paper 元数据
python "D:\1work\1_2026\polymer\code\ocr\transform_mineru_to_standard.py" `
  --mineru-output "<MinerU输出目录>" `
  --processed-output "D:\1work\1_2026\polymer\processed_data"

# 4. 执行 Stage 0-6
python "D:\1work\1_2026\polymer\code\extraction\main.py" --batch --workers 2
```

其中步骤 1 和 2 复用现有脚本；`transform_mineru_to_standard.py`、`main.py` 及 Stage 0-6 为本方案待实现部分。当前样本已有 MinerU 输出，首次实施从步骤 2 开始。

---

## 6. LLM 配置

**借鉴 V5 的配置结构和重试策略**，保留本项目独立配置；API 密钥只从环境变量读取，不写入 YAML：

```yaml
# config/pipeline.yaml
llm:
  default:
    provider: dmx
    model: claude-sonnet-5
    base_url: https://www.dmxapi.cn/v1
    api_format: anthropic-messages
    thinking_effort: low
    timeout_seconds: 600
    max_retries: 2
    retry_backoff_seconds: 2

  stage_overrides:          # 未配置的字段继承 default
    stage1_material_mention:
      api_format: openai-chat-completions
      thinking_effort: null

pricing:
  currency: CNY
  models:
    claude-sonnet-5:
      input_per_million: "2"
      output_per_million: "10"

concurrency:
  max_doc_workers: 1     # 先单文档调试
  max_llm_workers: 1

paths:
  input_dir: D:\1work\1_2026\polymer\processed_data\documents
  output_dir: D:\1work\1_2026\polymer\code\extraction\output
  source_root: D:\1work\1_2026\polymer\wenxian
```

每个调用方传入稳定的 `stage_id`：

```python
client = LLMClient(config, stage="stage4_property")
response = client.call(messages)
```

`LLMClient` 的配置解析顺序：

1. 读取 `llm.default`。
2. 按 `stage` 查找 `llm.stage_overrides`。
3. 将阶段配置覆盖到 default；未配置字段继续继承。
4. 暴露本次请求的 resolved provider、model 和基础参数，供 Provenance 记录。

约定的 stage_id 为：

```text
meta_extract
stage1_material_mention
stage2_polymer_entity
stage3_sample_process
stage4_property
stage5_characterization
```

Paper 的 `metadata_extraction.model` 和 Stage 6 汇总的 Provenance 都从客户端 resolved 配置自动写入，禁止在业务脚本或输出模板中手动填写模型名。重试、降级或路由导致实际模型变化时，记录最终实际调用的模型。

### Prompt 管理规范

Prompt 继续使用独立 Markdown 文件管理。Markdown 是开发和交付阶段的源文件，便于领域人员阅读、评审和通过 Git 查看修改，不引入数据库或 `prompt_store.json`。

#### 文件格式

每个 Prompt 使用统一 YAML front matter，并包含稳定 `prompt_id`：

```markdown
---
prompt_id: polymer.stage3.sample_process
version: 1.0.0
stage: stage3_sample_process
output_schema: process_schema.v1
---

# Role

你是高分子文献工艺抽取助手。

# Task

抽取 Sample 和 ProcessStep，并建立步骤顺序及样品流转关系。

# Rules

1. 只提取原文明确出现的信息。
2. 不根据常识补齐缺失参数。
3. 每条结果必须关联 Evidence。

# Input

{{document_blocks}}
{{polymer_entities}}
{{output_schema}}
```

`common_guardrails.md` 同样具有独立 `prompt_id` 和版本号，由各阶段共同引用。

#### PromptLoader

`prompt_loader.py` 负责：

1. 扫描 `prompts/*.md`，解析 front matter 和正文。
2. 校验 `prompt_id` 唯一、版本号存在、stage 与配置一致。
3. 根据 `prompt_id` 加载 Prompt，业务脚本中不得硬编码 Prompt 正文或依赖具体文件名。
4. 按固定顺序拼装公共规则、阶段 Prompt 和输出 Schema。
5. 渲染模板变量，调用 LLM 前检查不存在未替换的 `{{...}}`。
6. 对最终发送给模型的完整 Prompt 计算 SHA256。

固定拼装顺序：

```text
common_guardrails
  → stage-specific prompt
  → 当前 Pydantic 输出 Schema
  → 本次文献及上游阶段输入
```

其中 guardrails、阶段任务和输出约束进入 system message；文献正文及上游 JSON 进入 user message，并使用明确边界标记为不可信输入。

#### 配置引用

pipeline.yaml 只引用 `prompt_id`：

```yaml
stages:
  meta_extract:
    prompt_id: polymer.meta.extract
  stage3_sample_process:
    prompt_id: polymer.stage3.sample_process
  stage4_property:
    prompt_id: polymer.stage4.property
```

模型选择仍由 `llm.default` 和 `llm.stage_overrides` 管理，Prompt 与模型配置相互独立。

#### Schema 单一来源

Pydantic 模型是输出结构的唯一事实来源。Prompt 可以解释字段语义，但不手工维护另一份完整 JSON Schema；运行时由 PromptLoader 将当前阶段的 Pydantic JSON Schema 注入 `{{output_schema}}`，避免 Prompt 与代码结构漂移。

`polymer_schema.yaml` 继续管理性质名称、单位、工艺类型和表征方法等受控词表，不替代 Pydantic 输出模型。

#### 缓存与 Provenance

阶段缓存键至少包含：

```text
input_hash
+ rendered_prompt_hash
+ model_config_hash
+ output_schema_version
```

任一 Prompt、公共 guardrails、模型配置或输出 Schema 变化时，旧缓存自动失效。Stage 6 在内部 Provenance 中记录：

```json
{
  "prompt_id": "polymer.stage3.sample_process",
  "prompt_version": "1.0.0",
  "prompt_sha256": "<rendered_prompt_sha256>",
  "output_schema_version": "process_schema.v1"
}
```

#### 基础测试

- 所有 Prompt 的 front matter 可解析，`prompt_id` 不重复。
- pipeline.yaml 引用的每个 `prompt_id` 都存在。
- 必填模板变量齐全，渲染后不存在 `{{...}}`。
- Prompt 修改后哈希变化，并触发阶段缓存失效。
- 每个阶段至少保留一组小型固定输入，验证输出能够通过对应 Pydantic Schema。

---

## 7. 输出格式

### `final.json` 顶层结构

```json
{
  "schema_version": "1.2",
  "document_id": "reference_no_0001016",
  "paper": {
    "ref_no": "reference_no_0001016",
    "pdf_filename": "afad6fc5-94c6-4896-81c6-cbb3107cca3c_origin.pdf",
    "title": "...",
    "doi": null,
    "authors": ["..."],
    "journal": "...",
    "year": 1969,
    "metadata_status": "partial",
    "metadata_extraction": {...}
  },
  "material_mentions": [...],
  "polymer_entities": [...],
  "samples": [...],
  "process_steps": [...],
  "property_observations": [...],
  "measurement_conditions": [...],
  "characterizations": [...],
  "evidence": [...],
  "provenance": [...],
  "warnings": [...],
  "validation_summary": {
    "status": "passed_with_warnings",
    "error_count": 0,
    "warning_count": 2
  }
}
```

### Evidence 对象

Stage 1-5 的中间产物可以内嵌 Evidence，便于单阶段调试。Stage 6 合并时对 Evidence 去重，写入 `final.json` 顶层 `evidence`，业务对象通过 `evidence_ids` 引用。

```json
{
  "evidence_id": "ev001",
  "block_id": "P_3_2",
  "page": 3,
  "bbox": [103, 265, 913, 380],
  "source_type": "text",
  "source_sentence": "SPAEK-NA-60 was dissolved in DMAc...",
  "table_locator": null
}
```

表格来源使用 `table_locator` 保存 `table_id`、`row_label`、`column_label` 和 `cell_value`。每条实体、工艺、性质和表征结果必须至少有一条 Evidence；无法定位时保留结果并标记 `evidence_status: unresolved`，进入 warning 和人工审核。

### Provenance 对象

OCR 与 LLM 阶段统一写入 `final.json.provenance`，但字段按阶段类型区分：

```json
[
  {
    "stage": "ocr",
    "tool": "mineru_batch_parse.py",
    "batch_id": "<batch_id>",
    "model_version": "vlm",
    "ocr_enabled": false,
    "status": "done",
    "manifest_file": "batch_<batch_id>_manifest.json"
  },
  {
    "stage": "meta_extract",
    "provider": "dmx",
    "model": "claude-sonnet-5",
    "prompt_file": "prompts/meta_extract.md",
    "prompt_sha256": "<sha256>",
    "status": "success"
  }
]
```

OCR Provenance 不得包含 `MINERU_API_KEY`、授权头、上传 URL 或下载 URL。

---

## 8. 实施节奏

| 周次 | 任务 | 验收标准 |
|------|------|---------|
| Week 0 | 接入现有 `mineru_batch_parse.py`，实现 OCR 完成门禁 | 现有 OCR 单测通过；可识别 done/failed/缺失产物；已有完整结果不会重复上传 |
| Week 0.5 | 明确素材整理职责，写 `transform_mineru_to_standard.py` 与 `meta_extract.md` | `processed_data/` 结构完整；Paper 元数据、正文和表格均进入 document.json；缓存与 `--force-meta` 生效 |
| Week 1 | 搭环境，写 `stage0_load_document.py` | 输出 `stage0_blocks.json`，section 与 element 类型正确 |
| Week 2 | Stage 1 MaterialMention | 1-2 篇人工核对，polymer mention 召回率 ≥ 80% |
| Week 3 | Stage 2 PolymerEntity | variant_of 关系准确，SPAEK-30/40/60 正确识别为同一 family |
| Week 4 | Stage 3 Sample + ProcessStep | DAG 完整，工艺链 ≥ 3 步，参数有 evidence |
| Week 5 | Stage 4 PropertyObservation + MeasurementCondition | 正文和表格性质均可抽取，样品、条件、原始值和证据关联完整 |
| Week 6 | Stage 5 Characterization | 至少覆盖 FTIR/NMR/GPC/DSC/TGA 中样本文献实际出现的方法，关联 Sample 或 PolymerEntity |
| Week 7 | Stage 6 校验与端到端测试 | 3 篇代表文献通过 Schema、引用完整性、DAG 和 evidence 校验 |

---

## 9. 验收标准

选择至少 3 篇代表文献覆盖正文性质、复杂表格和表征结果。单篇文献全流程跑通后，`final.json` 满足：

1. **OCR**：每篇有明确 done/failed 状态；done 文献满足 MinerU 必需产物门禁；可通过 `batch_id` 恢复中断任务，且输出不包含密钥或授权信息。
2. **Schema**：Stage 0-6 输出均通过 Pydantic 校验，`schema_version` 明确。
3. **Paper**：固定 5 个 LLM 字段、三态 `metadata_status` 和提取记录齐全；缺失信息保持 null，不猜测。
4. **MaterialMention / PolymerEntity**：人工标注集上的 mention 召回率达到约定目标；无法归一的 mention 保留为 `unresolved`。
5. **Sample / ProcessStep**：所有输入输出引用有效，样品谱系无环；没有证据时不创建虚构工艺步骤。
6. **PropertyObservation**：每条性质保留原始值和单位，关联 Sample、MeasurementCondition 与 Evidence；缺失项显式标记。
7. **Characterization**：每条表征关联 Sample 或 PolymerEntity，方法、结果与 Evidence 可审核。
8. **Evidence**：每条数据有 `block_id + page + bbox` 可追溯；表格性质同时保留行、列和值定位信息。
9. **Provenance**：OCR 记录 batch/model/options，LLM 调用记录阶段实际 provider/model 与 prompt，不依赖手工填写。
10. **质量控制**：Stage 6 无硬错误才生成可发布结果，warning 汇总到 `final.json`。

---

## 10. 关键约束与风险

| 风险 | 应对 |
|------|------|
| MinerU API 超时、网络中断或批次未完成 | 保留 manifest/status，通过 `--batch-id` 继续轮询和下载，不重复上传 |
| 同一批次部分文献解析失败 | done 文献继续下游，failed 文献单独记录错误并重试，不让单篇失败阻塞整批 |
| 数字 PDF 与扫描 PDF 混用同一 OCR 参数 | 按 PDF 类型拆分批次；扫描件使用 `--ocr`，已有文本层的 PDF 使用默认模式 |
| MinerU 产物不完整 | 阶段 2 前执行必需文件门禁，缺失 `.md`、content v1 或 origin PDF 时停止该篇 |
| 复杂合并表头（多行表头/斜线表头）| MinerU 表格 block 保留原始 `table_body`；Stage 4 专用 prompt 重建表头，必要时保存原始图片 |
| 聚合物命名多样（缩写/商品名/化学名混用）| Stage 1 few-shot 覆盖多种命名模式；先建受控词表 |
| 同一论文多个不同类型的 PolymerEntity | Stage 2 明确 variant 判断逻辑，宁可多建不合并 |
| OCR 识别错误（化学式/数字）| Evidence 绑定到原始 page+bbox，审核时可回溯 `wenxian/{ref_no}/origin.pdf` 和 `images/` |
| 性质无法确定对应样品 | 保留原始样品称谓和 unresolved 引用，生成 warning，不绑定到推测 Sample |
| 测量条件分散在方法、表头和脚注 | Stage 4 同时读取 Methods、caption、table body 与 footnote，统一生成 MeasurementCondition |
| DSC/TGA/GPC 同时属于表征方法并产生性质值 | Stage 5 建 Characterization，Stage 4 建 PropertyObservation，通过 `derived_property_ids` 关联 |
| unresolved 情况 | 保留 MaterialMention，标记 `resolution_status: unresolved`，禁止强行合并 |

---

## 11. 后续增强（不影响一期完整性）

一期已经包含实体、工艺、性质和表征的完整基础流程。后续增强项包括：

- 扩展 17 类性质受控词表、单位换算和同义词映射。
- 增加领域规则与 LLM 复核闭环，对低置信度结果定向重跑。
- 完善人工审核状态和修改记录，并利用已有 Provenance 支持结果复现与版本比较。
- 在人工标注集上持续评估 mention 召回率、关系准确率和数值抽取准确率。
