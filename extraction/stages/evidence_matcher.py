"""Evidence 匹配器：判定 Evidence 是否真的落在 Stage 0 原文里。

## 为什么需要单独一层

Stage 0 存的是 MinerU 的原始表示（表格是 HTML，公式是 LaTeX），而下游各 Stage
写回的 source_sentence 是各自渲染过的形态：

    Stage 0   <tr><td>PC-1</td><td>394</td><td>446</td></tr>
    Stage 4R  PC-1 | 394 | 446
    Stage 4/5 25 ± 0.02 °C          （Stage 0 里是 $25 \\pm 0.02\\ ^{\\circ}\\mathrm{C}$）

三者指的是同一处原文，字面却互不包含。旧版 Stage 6 用「字面子串 + 一张替换表」
判定，于是把这种表示差异记成 evidence_not_in_source 错误。全批 2809 条 evidence
里有 532 条这样被判错，经逐条核查**没有一条是编造或归属错误**，全部是表示层差异。

## 分层策略

表格类（有 table_locator）——**不做整表模糊匹配**，只走确定性定位：

  1. cell_id 精确解析：从 Stage 0 的 table_cells 里按 cell_id 取格，比对 cell_value
  2. 行列下标 + 标签：cell_id 缺失时按 (row_index, column_index) 取格
  3. 单元格集合：把 locator 的三个字段拿去和该表所有格子的文本比对

正文类——逐级放宽，但每一级都必须保持「不会把 446 和 464 判成同一个」：

  1. 严格子串
  2. 安全归一化（控制字符 / NFKC / LaTeX 命令 / PDF 拆散的数字）后子串
  3. 词多重集覆盖率：≥98% 记 matched_after_normalization，≥90% 记 matched 但带
     warning。用多重集而不是集合，"446" 和 "464" 分词后不同，不会互相顶替。
  4. 唯一块恢复：归一化后在**恰好一个**其他块里命中才认，多命中一律判 ambiguous。
     这一级在当前 20 篇数据上一次都没触发，属防御性实现。

返回的状态供调用方自行决定升级为 error 还是降级为 warning：Strict 只接受
matched；Preview 接受到 matched_after_block_recovery 为止。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


# 匹配状态，按可信度从高到低
MATCHED = "matched"
MATCHED_AFTER_NORMALIZATION = "matched_after_normalization"
MATCHED_AFTER_BLOCK_RECOVERY = "matched_after_block_recovery"
AMBIGUOUS = "ambiguous"
UNRESOLVED = "unresolved"

# Preview 可以接受的状态；Strict 只接受 MATCHED
RELAXED_ACCEPTABLE = frozenset({
    MATCHED,
    MATCHED_AFTER_NORMALIZATION,
    MATCHED_AFTER_BLOCK_RECOVERY,
})

# 词覆盖率阈值：达到 HIGH 视为纯字符层打散；达到 LOW 仍算命中但要提醒人看
COVERAGE_HIGH = 0.98
COVERAGE_LOW = 0.90

# PDF 抽取残留的控制字符（保留 \t \n \r）
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WORD_RE = re.compile(r"[a-z0-9]+")

# LaTeX 命令 → 目标字符。只列可逆、不会造成数值歧义的
_LATEX_LITERALS = (
    (r"\pm", "\u00b1"),
    (r"\mp", "\u2213"),
    (r"\times", "\u00d7"),
    (r"\cdot", "\u00b7"),
    (r"\%", "%"),
    (r"\mu", "\u03bc"),
    (r"\eta", "\u03b7"),
    (r"\delta", "\u03b4"),
    (r"\Delta", "\u0394"),
    (r"\chi", "\u03c7"),
    (r"\alpha", "\u03b1"),
    (r"\beta", "\u03b2"),
    (r"\gamma", "\u03b3"),
    (r"\lambda", "\u03bb"),
    (r"\sigma", "\u03c3"),
    (r"\rho", "\u03c1"),
    (r"\theta", "\u03b8"),
    (r"\omega", "\u03c9"),
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class EvidenceMatch:
    """一次匹配的结论。

    status  见模块顶部的常量
    detail  给人看的一句话，进 warning message
    """

    status: str
    detail: str = ""
    recovered_block_id: str | None = None
    coverage: float | None = None
    candidates: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok_strict(self) -> bool:
        return self.status == MATCHED

    @property
    def ok_relaxed(self) -> bool:
        return self.status in RELAXED_ACCEPTABLE


def normalize_evidence_text(value: str) -> str:
    """把两侧表示折叠到同一形态，但保证数值不被改写。

    只做「删除装饰」和「合并被拆开的同一个记号」，不做任何数值替换或近似。
    """
    text = _CONTROL_RE.sub(" ", value or "")
    text = unicodedata.normalize("NFKC", text)
    # \mathrm{C} / \text{max} 这类只是排版包装，取花括号里的内容
    text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\text(?:rm|bf|it)?\s*\{([^{}]*)\}", r"\1", text)
    # ^\circ / ^{\circ} 都是度符号
    text = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "\u00b0", text)
    for command, target in _LATEX_LITERALS:
        text = text.replace(command, target)
    # 表格 HTML 的标签本身不是内容
    text = _HTML_TAG_RE.sub(" ", text)
    # 剩余的 LaTeX 结构符号
    text = re.sub(r"[\\${}]", " ", text)
    # 各种连字符/减号统一
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    # PDF 常把一个数字整个拆开写成 "0 . 0 6 3"。只在**整段都是数字和小数点**
    # 时才合并，这样 "394 446"（两个独立的数）不会被粘成 "394446"：
    # 下面的模式要求序列里至少含一个小数点或千分位逗号。
    text = re.sub(
        r"(?<![\w.])(\d(?:\s*\d)*\s*[.,]\s*\d(?:\s*[\d.,])*)(?![\w])",
        lambda m: re.sub(r"\s+", "", m.group(1)),
        text,
    )
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+(?=[.,;:])", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _words(value: str) -> list[str]:
    return _WORD_RE.findall(normalize_evidence_text(value))


def word_coverage(needle: str, haystack: str) -> float:
    """needle 的词有多少比例出现在 haystack 里，按多重集计。

    用多重集的理由：句子里出现两次 "446" 时，haystack 只有一次就不该算全覆盖。

    对**纯字母**的词额外允许一种命中：它是原文某个词的一部分。这是为了
    PDF 把相邻两词粘掉空格的情形（原文 "ofOT(2)"，evidence 写的是 "of OT(2)"）。
    这条回退**不适用于数字**——否则 "44" 会被 "446" 顶掉，把不同的数值判成同一个。
    """
    needle_words = _words(needle)
    if not needle_words:
        return 0.0
    pool = Counter(_words(haystack))
    # 只有纯字母词才参与粘连回退
    glued = [word for word in pool if word.isalpha()]
    hit = 0
    for word in needle_words:
        if pool[word] > 0:
            pool[word] -= 1
            hit += 1
        elif word.isalpha() and any(word in candidate for candidate in glued):
            hit += 1
    return hit / len(needle_words)


def _cells(block: Any) -> list[Any]:
    return list(getattr(block, "table_cells", None) or [])


def _cell_text(cell: Any) -> str:
    return (getattr(cell, "text", "") or "").strip()


def match_table_evidence(
    block: Any,
    locator: dict[str, Any] | None,
    source_text: str,
) -> EvidenceMatch:
    """表格 Evidence：只用确定性定位，绝不对整张表做模糊匹配。"""
    if not locator:
        return EvidenceMatch(UNRESOLVED, "缺少 table_locator")
    cells = _cells(block)
    if not cells:
        # Stage 0 没有拆出格子，退回文本层判断
        return match_text_evidence(
            locator.get("cell_value") or "",
            source_text,
        )

    expected = locator.get("cell_value")
    cell_id = locator.get("cell_id")

    # 第一优先级：cell_id 直接取格
    if cell_id:
        hit = next((c for c in cells if getattr(c, "cell_id", None) == cell_id), None)
        if hit is not None:
            return _compare_cell(hit, expected, "cell_id")
        return EvidenceMatch(
            UNRESOLVED,
            f"table_locator.cell_id 在 Stage 0 表格里不存在：{cell_id}",
        )

    # 第二优先级：行列下标
    row = locator.get("row_index")
    column = locator.get("column_index")
    if row is not None and column is not None:
        hit = next(
            (
                c
                for c in cells
                if getattr(c, "row_index", None) == row
                and getattr(c, "column_index", None) == column
            ),
            None,
        )
        if hit is not None:
            return _compare_cell(hit, expected, "row/column 下标")
        return EvidenceMatch(
            UNRESOLVED,
            f"table_locator 的行列下标越界：r{row} c{column}",
        )

    # 第三优先级：单元格集合。只在值唯一命中时认，多个同值格子判歧义 ——
    # 此时行列标签都没有，无法判断指的是哪一个。
    if not isinstance(expected, str) or not expected.strip():
        return EvidenceMatch(UNRESOLVED, "table_locator 既无稳定坐标也无 cell_value")
    target = normalize_evidence_text(expected)
    matches = [c for c in cells if normalize_evidence_text(_cell_text(c)) == target]
    if len(matches) == 1:
        return EvidenceMatch(
            MATCHED_AFTER_NORMALIZATION,
            f"按单元格值唯一定位到 {getattr(matches[0], 'cell_id', '?')}",
        )
    if len(matches) > 1:
        return EvidenceMatch(
            AMBIGUOUS,
            f"cell_value={expected!r} 在该表命中 {len(matches)} 个格子，无法唯一定位",
            candidates=tuple(
                str(getattr(c, "cell_id", "")) for c in matches[:8]
            ),
        )
    return _match_multi_value_cell(cells, expected)


def _match_multi_value_cell(cells: list[Any], expected: str) -> EvidenceMatch:
    """cell_value 写成 "0.75, 0.70, 0.66, 0.64" 这种一列多值的形式。

    模型偶尔会把同一列的若干格拼成一个 cell_value —— locator 的语义是错的
    （它应该指向单个格子），但数据本身未必错。只有**每个分量都能在表里
    找到对应格子**时才降级，任何一个找不到就照旧 unresolved。
    """
    parts = [part.strip() for part in expected.split(",") if part.strip()]
    if len(parts) < 2:
        return EvidenceMatch(UNRESOLVED, f"cell_value={expected!r} 不在该表任何格子里")
    cell_texts = {normalize_evidence_text(_cell_text(c)) for c in cells}
    missing = [p for p in parts if normalize_evidence_text(p) not in cell_texts]
    if missing:
        return EvidenceMatch(
            UNRESOLVED,
            f"cell_value={expected!r} 拆成 {len(parts)} 个值后仍有 {missing} 不在表里",
        )
    return EvidenceMatch(
        MATCHED_AFTER_BLOCK_RECOVERY,
        f"cell_value 把 {len(parts)} 个单元格拼成了一条，"
        "各分量均能在该表定位，但 locator 未指向单个格子",
    )


def _compare_cell(cell: Any, expected: Any, how: str) -> EvidenceMatch:
    actual = _cell_text(cell)
    if expected is None:
        # 空格子 locator：Stage 0 那一格也必须是空的
        if not actual:
            return EvidenceMatch(MATCHED, f"按 {how} 命中空单元格")
        return EvidenceMatch(
            UNRESOLVED,
            f"locator 声明为空单元格，Stage 0 实际是 {actual!r}",
        )
    if not isinstance(expected, str):
        return EvidenceMatch(UNRESOLVED, "cell_value 类型非法")
    if expected.strip() == actual:
        return EvidenceMatch(MATCHED, f"按 {how} 精确命中")
    if normalize_evidence_text(expected) == normalize_evidence_text(actual):
        return EvidenceMatch(
            MATCHED_AFTER_NORMALIZATION,
            f"按 {how} 命中，归一化后一致（Stage 0 原文 {actual!r}）",
        )
    return EvidenceMatch(
        UNRESOLVED,
        f"按 {how} 定位到的单元格是 {actual!r}，与 cell_value={expected!r} 不符",
    )


def match_text_evidence(sentence: str, source_text: str) -> EvidenceMatch:
    """正文 Evidence：严格 → 安全归一化 → 词覆盖率。"""
    if not sentence:
        return EvidenceMatch(UNRESOLVED, "source_sentence 为空")
    if sentence in source_text:
        return EvidenceMatch(MATCHED, "严格子串命中")
    normalized_sentence = normalize_evidence_text(sentence)
    normalized_source = normalize_evidence_text(source_text)
    if normalized_sentence and normalized_sentence in normalized_source:
        return EvidenceMatch(MATCHED_AFTER_NORMALIZATION, "归一化后子串命中")
    coverage = word_coverage(sentence, source_text)
    if coverage >= COVERAGE_HIGH:
        return EvidenceMatch(
            MATCHED_AFTER_NORMALIZATION,
            f"词覆盖率 {coverage:.1%}，内容在本块内但字符被打散",
            coverage=coverage,
        )
    if coverage >= COVERAGE_LOW:
        return EvidenceMatch(
            MATCHED_AFTER_BLOCK_RECOVERY,
            f"词覆盖率仅 {coverage:.1%}，疑似跨块或被截断，建议人工确认",
            coverage=coverage,
        )
    return EvidenceMatch(
        UNRESOLVED,
        f"词覆盖率 {coverage:.1%}，内容不在该 block 内",
        coverage=coverage,
    )


def resolve_evidence_block(
    sentence: str,
    blocks: Iterable[tuple[str, str]],
    *,
    exclude_block_id: str | None = None,
) -> EvidenceMatch:
    """block_id 写错时，看能不能在别的块里唯一恢复。

    blocks 传 (block_id, source_text) 序列。只有**恰好一个**块命中才恢复，
    多个命中一律判 ambiguous —— 随便选第一个会把证据挂到错误的块上。

    注：当前 20 篇数据上这一级一次都没触发过（0 条 block_id 归属错误），
    保留是为了 block_id 真的写错时有确定性的处理路径，而不是静默通过。
    """
    normalized = normalize_evidence_text(sentence)
    if not normalized:
        return EvidenceMatch(UNRESOLVED, "source_sentence 为空")
    hits = [
        block_id
        for block_id, text in blocks
        if block_id != exclude_block_id and normalized in normalize_evidence_text(text)
    ]
    if len(hits) == 1:
        return EvidenceMatch(
            MATCHED_AFTER_BLOCK_RECOVERY,
            f"原文实际在 {hits[0]}，block_id 归属有误",
            recovered_block_id=hits[0],
        )
    if len(hits) > 1:
        return EvidenceMatch(
            AMBIGUOUS,
            f"该句在 {len(hits)} 个 block 里都出现，无法唯一恢复",
            candidates=tuple(hits[:8]),
        )
    return EvidenceMatch(UNRESOLVED, "全文任何 block 里都找不到该句")
