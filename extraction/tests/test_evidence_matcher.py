"""evidence_matcher 的单元测试。

重点不是「能不能匹配上」，而是**该拒绝的时候有没有拒绝**：
数字不能互相顶替、多处命中不能随便选一个、值不在表里不能放行。
"""

import unittest

from stages import evidence_matcher as matcher
from stages.table_grid import parse_table_cells


class _Block:
    """只提供 matcher 用到的两个属性的最小替身。"""

    def __init__(self, table_cells=None, block_type="table"):
        self.table_cells = table_cells
        self.type = block_type


def _table(body: str, table_id: str = "T_1_1") -> _Block:
    return _Block(parse_table_cells(body, table_id))


TABLE_BODY = (
    "<table>"
    "<tr><td>Sample</td><td>Td5 (°C)</td><td>Td50 (°C)</td></tr>"
    "<tr><td>PC-1</td><td>394</td><td>446</td></tr>"
    "<tr><td>PC-2</td><td>446</td><td>512</td></tr>"
    "</table>"
)


class NormalizeTests(unittest.TestCase):
    def test_latex_and_html_fold_to_same_text(self) -> None:
        latex = r"$25 \pm 0.02\ ^{\circ}\mathrm{C}$"
        readable = "25 ± 0.02 °C"
        self.assertEqual(
            matcher.normalize_evidence_text(latex),
            matcher.normalize_evidence_text(readable),
        )

    def test_split_decimal_is_rejoined(self) -> None:
        self.assertIn("0.063", matcher.normalize_evidence_text("0 . 0 6 3"))

    def test_separate_integers_are_not_glued(self) -> None:
        """"394 446" 是两个数，绝不能被粘成 "394446"。"""
        self.assertEqual(
            matcher.normalize_evidence_text("PC-1 | 394 | 446"),
            "pc-1 | 394 | 446",
        )
        self.assertNotIn(
            "394446", matcher.normalize_evidence_text("temperatures 394 446 512")
        )

    def test_control_characters_removed(self) -> None:
        self.assertEqual(
            matcher.normalize_evidence_text("Tg\x00 = 85\x07 °C"),
            matcher.normalize_evidence_text("Tg = 85 °C"),
        )


class WordCoverageTests(unittest.TestCase):
    def test_digits_never_satisfied_by_a_longer_number(self) -> None:
        """"44" 不能被原文里的 "446" 顶掉——否则不同数值会被判成同一个。"""
        self.assertEqual(matcher.word_coverage("44", "the value is 446"), 0.0)

    def test_glued_word_still_counts(self) -> None:
        """PDF 把相邻两词粘掉空格：原文 ofOT(2)，evidence 写 of OT(2)。"""
        self.assertEqual(
            matcher.word_coverage("of OT(2)", "the content ofOT(2) was"), 1.0
        )

    def test_multiset_semantics(self) -> None:
        """句子里出现两次的词，原文只有一次时不算全覆盖。"""
        self.assertLess(matcher.word_coverage("446 and 446", "only 446 here"), 1.0)


class TableEvidenceTests(unittest.TestCase):
    def test_cell_id_exact_hit(self) -> None:
        block = _table(TABLE_BODY)
        match = matcher.match_table_evidence(
            block,
            {
                "table_id": "T_1_1",
                "cell_id": "T_1_1:r0001:c0002",
                "row_label": "PC-1",
                "column_label": "Td50 (°C)",
                "cell_value": "446",
            },
            "",
        )
        self.assertEqual(match.status, matcher.MATCHED)
        self.assertTrue(match.ok_strict)

    def test_cell_id_pointing_at_a_different_value_is_rejected(self) -> None:
        block = _table(TABLE_BODY)
        match = matcher.match_table_evidence(
            block,
            {
                "table_id": "T_1_1",
                "cell_id": "T_1_1:r0001:c0001",
                "row_label": "PC-1",
                "column_label": "Td5 (°C)",
                "cell_value": "446",
            },
            "",
        )
        self.assertEqual(match.status, matcher.UNRESOLVED)
        self.assertFalse(match.ok_relaxed)

    def test_row_column_index_fallback(self) -> None:
        block = _table(TABLE_BODY)
        match = matcher.match_table_evidence(
            block,
            {
                "table_id": "T_1_1",
                "row_index": 2,
                "column_index": 2,
                "row_label": "PC-2",
                "column_label": "Td50 (°C)",
                "cell_value": "512",
            },
            "",
        )
        self.assertEqual(match.status, matcher.MATCHED)

    def test_duplicate_value_without_coordinates_is_ambiguous(self) -> None:
        """446 在表里出现两次，又没有 cell_id —— 不许猜。"""
        block = _table(TABLE_BODY)
        match = matcher.match_table_evidence(
            block, {"table_id": "T_1_1", "cell_value": "446"}, ""
        )
        self.assertEqual(match.status, matcher.AMBIGUOUS)
        self.assertFalse(match.ok_relaxed)

    def test_multi_value_cell_accepted_only_when_every_part_locates(self) -> None:
        body = (
            "<table><tr><td>x</td></tr><tr><td>0.75</td></tr>"
            "<tr><td>0.70</td></tr><tr><td>0.66</td></tr></table>"
        )
        block = _table(body)
        ok = matcher.match_table_evidence(
            block, {"table_id": "T_1_1", "cell_value": "0.75, 0.70, 0.66"}, ""
        )
        self.assertEqual(ok.status, matcher.MATCHED_AFTER_BLOCK_RECOVERY)
        bad = matcher.match_table_evidence(
            block, {"table_id": "T_1_1", "cell_value": "0.75, 0.70, 9.99"}, ""
        )
        self.assertEqual(bad.status, matcher.UNRESOLVED)

    def test_value_absent_from_table_is_rejected(self) -> None:
        block = _table(TABLE_BODY)
        match = matcher.match_table_evidence(
            block, {"table_id": "T_1_1", "cell_value": "999"}, ""
        )
        self.assertFalse(match.ok_relaxed)


class TextEvidenceTests(unittest.TestCase):
    def test_pipe_rendered_row_matches_html_source(self) -> None:
        """Stage 4R 写管道行，Stage 0 存 HTML —— 同一处原文。"""
        match = matcher.match_text_evidence("PC-1 | 394 | 446", TABLE_BODY)
        self.assertTrue(match.ok_relaxed)

    def test_strict_substring(self) -> None:
        match = matcher.match_text_evidence("Tg was 85 °C", "We found Tg was 85 °C.")
        self.assertEqual(match.status, matcher.MATCHED)

    def test_unrelated_sentence_is_unresolved(self) -> None:
        match = matcher.match_text_evidence(
            "The tensile strength reached 133 MPa after annealing",
            "Thermal stability was assessed by TGA under nitrogen flow.",
        )
        self.assertEqual(match.status, matcher.UNRESOLVED)


class BlockRecoveryTests(unittest.TestCase):
    def test_unique_hit_recovers_block_id(self) -> None:
        blocks = [("P_1_0", "unrelated"), ("P_2_0", "the Tg was 85 °C indeed")]
        match = matcher.resolve_evidence_block(
            "Tg was 85 °C", blocks, exclude_block_id="P_1_0"
        )
        self.assertEqual(match.status, matcher.MATCHED_AFTER_BLOCK_RECOVERY)
        self.assertEqual(match.recovered_block_id, "P_2_0")

    def test_multiple_hits_stay_ambiguous(self) -> None:
        blocks = [("P_1_0", "Tg was 85 °C"), ("P_2_0", "Tg was 85 °C too")]
        match = matcher.resolve_evidence_block(
            "Tg was 85 °C", blocks, exclude_block_id="P_9_9"
        )
        self.assertEqual(match.status, matcher.AMBIGUOUS)
        self.assertFalse(match.ok_relaxed)


if __name__ == "__main__":
    unittest.main()
