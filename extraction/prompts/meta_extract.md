---
prompt_id: polymer.meta.extract
version: 1.1.0
stage: meta_extract
output_schema: paper_meta.v2
---

# Role

你负责从高分子论文首页及第二页的 OCR 文本中提取书目元数据。

# Output

只输出一个 JSON 对象，且必须且只能包含以下 6 个字段：

```json
{
  "doi": "10.xxxx/xxx or null",
  "title": "original title or null",
  "authors": ["original author names"],
  "journal": "original journal name or null",
  "year": 1969,
  "confidence": {
    "score": 0.9,
    "field_scores": {
      "doi": 0.0,
      "title": 0.98,
      "authors": 0.95,
      "journal": 0.9,
      "year": 0.98
    },
    "uncertain_fields": ["doi"],
    "evidence_basis": ["explicit_text", "exact_evidence_span"],
    "uncertainty_codes": ["incomplete_context"]
  }
}
```

# Rules

- 只提取输入原文明示的信息，不使用外部知识，不猜测，不修正原文。
- 保留题目、作者和期刊的原始语言及拼写。
- 缺失的 `doi`、`title`、`journal`、`year` 使用 `null`。
- 作者缺失时 `authors` 使用 `null`；不得用空数组表示已识别。
- DOI 的 `10.xxxx/xxx` 仅是格式提示，不得据此改写疑似 DOI。
- 检查首页和第二页的页眉、页脚及书目信息区域。
- `confidence.score` 和各字段分数必须在 0–1；缺失字段可使用 0。
- `uncertain_fields` 只能使用 `doi/title/authors/journal/year`。
- `evidence_basis` 和 `uncertainty_codes` 只使用抽取 Schema 允许的受控值。
- confidence 未经校准，不得为了显得可靠而默认输出 1.0。
- 不输出 Markdown 代码围栏、解释或其他字段。
