import unittest
from dataclasses import dataclass
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


from llm_client import (
    DEFAULT_CONFIG_PATH,
    LLMJSONResponse,
    LLMOutputTruncatedError,
    LLMRequestError,
    ResolvedLLMConfig,
    load_pipeline_config,
)
from prompt_loader import PromptLoader
from schema.polymer_schema import (
    MentionCandidate,
    MentionChunkResponse,
    Stage0Document,
)
from tests.helpers import add_model_confidence
from stages.stage1_material_mention import (
    Stage1Error,
    _failure_replay_client,
    _materialize_mentions,
    _resolve_surface_text,
    _source_sentence,
    _source_sentence,
    chunk_blocks,
    extract_material_mentions,
    run_stage1,
    select_input_blocks,
)


def stage0_document(
    *,
    sections: tuple[str, ...] = ("Abstract", "Methods"),
) -> Stage0Document:
    elements = []
    for index, section in enumerate(sections):
        text = (
            "cis-Polybutadiene rubber (Buna CB) was tested."
            if index == 0
            else "The PB sample was dried before measurement."
        )
        elements.append({
            "block_id": f"P_0_{index}",
            "type": "text",
            "section": section,
            "text": text,
            "page": 0,
            "bbox": [1, 2, 3, 4],
            "source_block_index": index,
            "alignment_status": "matched",
        })
    return Stage0Document.model_validate({
        "schema_version": "1.0",
        "source_document_schema_version": "1.0",
        "document_id": "reference_no_0000001",
        "paper": {
            "ref_no": "reference_no_0000001",
            "pdf_filename": "uuid_origin.pdf",
            "source_pdf_path": "mineru_output/reference_no_0000001/uuid_origin.pdf",
            "organized_pdf_path": "wenxian/reference_no_0000001/origin.pdf",
            "doi": None,
            "title": "Demo",
            "authors": ["A. Author"],
            "journal": "Journal",
            "year": 2026,
            "metadata_status": "partial",
            "metadata_extraction": {"status": "success"},
        },
        "source_files": {},
        "ocr": {"status": "done"},
        "elements": elements,
        "warnings": [],
    })


class FakeClient:
    def __init__(self) -> None:
        self.resolved = ResolvedLLMConfig(
            provider="test",
            requested_model="fake",
            model="fake",
            base_url="https://example.test/v1",
            timeout_seconds=10,
            max_retries=0,
            retry_backoff_seconds=0,
        )
        self.calls = 0
        self.max_tokens_seen: list[int] = []

    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        if "P_0_0" in user_message:
            mentions = [
                {
                    "block_id": "P_0_0",
                    "text": "cis-Polybutadiene rubber",
                    "mention_role": "polymer_name",
                },
                {
                    "block_id": "P_0_0",
                    "text": "Buna CB",
                    "mention_role": "commercial_name",
                },
            ]
        else:
            mentions = [{
                "block_id": "P_0_1",
                "text": "PB",
                "mention_role": "abbreviation",
            }]
        return LLMJSONResponse(
            data=add_model_confidence({"mentions": mentions}),
            provider="test",
            model="fake-actual",
        )


class RetryClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMJSONResponse(
                data=add_model_confidence({
                    "mentions": [{
                        "block_id": "not-in-input",
                        "text": "PB",
                        "mention_role": "abbreviation",
                    }]
                }),
                provider="test",
                model="fake-actual",
            )
        return LLMJSONResponse(
            data=add_model_confidence({
                "mentions": [{
                    "block_id": "P_0_0",
                    "text": "Buna CB",
                    "mention_role": "commercial_name",
                }]
            }),
            provider="test",
            model="fake-actual",
        )


class NormalizingClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        return LLMJSONResponse(
            data=add_model_confidence({
                "mentions": [{
                    "block_id": "P_0_0",
                    "text": "CIS-POLYBUTADIENE  RUBBER",
                    "mention_role": "polymer_name",
                }]
            }),
            provider="test",
            model="fake-actual",
        )


class FailingClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        raise LLMRequestError("LLM 响应不是有效 JSON")


class TruncatedClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        raise LLMOutputTruncatedError("truncated")


class TableMentionClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        return LLMJSONResponse(
            data=add_model_confidence({
                "mentions": [{
                    "block_id": "T_0_2",
                    "text": "PE1",
                    "mention_role": "sample_label",
                }]
            }),
            provider="test",
            model="fake-actual",
        )


class Stage1Tests(unittest.TestCase):
    def test_mixed_token_recovers_unique_ocr_join(self) -> None:
        source = "which disappear in the spectrum ofP2. Instead"

        self.assertEqual(_resolve_surface_text(source, "P2"), "P2")

    def test_long_material_token_recovers_unique_ocr_join(self) -> None:
        source = "the chiral pendant ofpolyacetylenes with different ratios"

        self.assertEqual(
            _resolve_surface_text(source, "polyacetylenes"),
            "polyacetylenes",
        )
        self.assertIn(
            "ofpolyacetylenes",
            _source_sentence(source, "polyacetylenes"),
        )

    def test_latex_spaced_formula_recovers_original_group(self) -> None:
        source = r"related polymers such as $\mathrm { P C H M A }$ were used"

        self.assertEqual(
            _resolve_surface_text(source, r"$\mathrm{PCHMA}$"),
            r"\mathrm { P C H M A }",
        )

    def test_short_mixed_token_recovers_across_latex_wrappers(self) -> None:
        source = r"$\mathrm { P } \mathbf { 1 }$ was used"

        self.assertEqual(
            _resolve_surface_text(source, "P1"),
            r"P } \mathbf { 1",
        )

    def test_candidate_only_extra_closing_math_delimiter_is_removed(self) -> None:
        source = r"$\mathrm { P 1 . p E G } _ { 7 5 0 } ^ { - } > P2$"
        candidate = r"$\mathrm { P 1 . p E G } _ { 7 5 0 } ^ { - }$"

        self.assertEqual(
            _resolve_surface_text(source, candidate),
            r"\mathrm { P 1 . p E G } _ { 7 5 0 } ^ { - }",
        )

    def test_latex_formula_with_parentheses_recovers_original_surface(self) -> None:
        source = r"$\mathrm { P M } \left( \mathrm { E O / P O } \right)$"

        self.assertEqual(
            _resolve_surface_text(source, "PM（EO/PO）"),
            r"P M } \left( \mathrm { E O / P O",
        )

    def test_hyphenated_candidate_recovers_unhyphenated_ocr_surface(self) -> None:
        source = "can lead to PPAbased copolymers with controllable content"

        self.assertEqual(
            _resolve_surface_text(source, "PPA-based copolymers"),
            "PPAbased copolymers",
        )

    def test_extracts_material_label_found_only_in_table(self) -> None:
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"].append({
            "block_id": "T_0_2",
            "type": "table",
            "section": "Results",
            "caption": "General properties of polymers",
            "table_body": (
                "<table><tr><td>Polymer</td><td>Tg</td></tr>"
                "<tr><td>PE1 $^{\\text{b}}$</td><td>119</td></tr></table>"
            ),
            "page": 0,
            "bbox": [1, 2, 3, 4],
            "source_block_index": 2,
        })
        document = Stage0Document.model_validate(document_data)
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )

        result = extract_material_mentions(
            document,
            TableMentionClient(),
            prompt,
        )

        mention = result.material_mentions[0]
        self.assertEqual(mention.text, "PE1")
        self.assertEqual(mention.evidence.block_id, "T_0_2")
        self.assertEqual(mention.evidence.source_type, "table")
        self.assertEqual(
            mention.evidence.source_sentence,
            "PE1 $^{\\text{b}}$",
        )

    def test_extracts_mentions_and_builds_evidence(self) -> None:
        document = stage0_document()
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        client = FakeClient()

        result = extract_material_mentions(
            document,
            client,
            prompt,
            max_chunk_chars=2000,
        )

        self.assertEqual(
            [mention.mention_id for mention in result.material_mentions],
            ["m001", "m002"],
        )
        self.assertEqual(
            result.material_mentions[0].evidence.block_id,
            "P_0_0",
        )
        self.assertIn(
            "cis-Polybutadiene rubber",
            result.material_mentions[0].evidence.source_sentence,
        )
        self.assertEqual(result.provenance.model, "fake-actual")
        self.assertEqual(result.provenance.chunk_count, 1)
        self.assertEqual(client.max_tokens_seen, [8192])

    def test_stage1_pipeline_limits_are_configured(self) -> None:
        config = load_pipeline_config(DEFAULT_CONFIG_PATH)
        stage = config["stages"]["stage1_material_mention"]

        self.assertEqual(stage["max_chunk_chars"], 8000)
        self.assertEqual(stage["max_tokens"], 8192)

    def test_truncation_does_not_enter_schema_validation(self) -> None:
        with patch(
            "stages.stage1_material_mention._validate_chunk_candidates"
        ) as validate:
            with self.assertRaises(Stage1Error):
                extract_material_mentions(
                    stage0_document(),
                    TruncatedClient(),
                    PromptLoader().render_stage_prompt(
                        "polymer.stage1.material_mention",
                        MentionChunkResponse,
                        expected_stage="stage1_material_mention",
                        expected_output_schema="material_mention_schema.v2",
                    ),
                    max_validation_retries=1,
                )

        validate.assert_not_called()

    def test_confidence_score_must_be_between_zero_and_one(self) -> None:
        payload = add_model_confidence({
            "mentions": [{
                "block_id": "P_0_0",
                "text": "Buna CB",
                "mention_role": "commercial_name",
            }]
        })
        payload["mentions"][0]["confidence"]["score"] = 1.1

        with self.assertRaises(ValueError):
            MentionChunkResponse.model_validate(payload)

    def test_confidence_cannot_reference_unknown_field(self) -> None:
        payload = add_model_confidence({
            "mentions": [{
                "block_id": "P_0_0",
                "text": "Buna CB",
                "mention_role": "commercial_name",
            }]
        })
        payload["mentions"][0]["confidence"]["field_scores"] = {
            "invented_field": 0.3,
        }

        with self.assertRaises(ValueError):
            MentionChunkResponse.model_validate(payload)

    def test_fallback_is_explicit(self) -> None:
        document = stage0_document(sections=("Introduction", "Conclusion"))

        blocks, warnings = select_input_blocks(document)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(warnings[0]["code"], "section_fallback")

    def test_invalid_candidate_is_retried(self) -> None:
        document = stage0_document(sections=("Abstract",))
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        client = RetryClient()

        result = extract_material_mentions(
            document,
            client,
            prompt,
            max_validation_retries=1,
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual(result.material_mentions[0].text, "Buna CB")

    def test_invalid_surface_candidate_fails_in_strict_mode(self) -> None:
        document = stage0_document(sections=("Abstract",))
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        client = FakeClient()

        def call_with_invalid_surface(*args, **kwargs):
            client.calls += 1
            return LLMJSONResponse(
                data=add_model_confidence({
                    "mentions": [
                        {
                            "block_id": "P_0_0",
                            "text": "invented blend",
                            "mention_role": "polymer_name",
                        },
                        {
                            "block_id": "P_0_0",
                            "text": "Buna CB",
                            "mention_role": "commercial_name",
                        },
                    ]
                }),
                provider="test",
                model="fake-actual",
            )

        client.call_json = call_with_invalid_surface
        with self.assertRaisesRegex(Stage1Error, "不是 .*原文子串"):
            extract_material_mentions(
                document,
                client,
                prompt,
                max_validation_retries=0,
            )

    def test_preview_drops_only_invalid_surface_candidate(self) -> None:
        document = stage0_document(sections=("Abstract",))
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        client = FakeClient()

        def call_with_invalid_surface(*args, **kwargs):
            client.calls += 1
            return LLMJSONResponse(
                data=add_model_confidence({
                    "mentions": [
                        {
                            "block_id": "P_0_0",
                            "text": "invented blend",
                            "mention_role": "polymer_name",
                        },
                        {
                            "block_id": "P_0_0",
                            "text": "Buna CB",
                            "mention_role": "commercial_name",
                        },
                    ]
                }),
                provider="test",
                model="fake-actual",
            )

        client.call_json = call_with_invalid_surface
        result = extract_material_mentions(
            document,
            client,
            prompt,
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(
            [mention.text for mention in result.material_mentions],
            ["Buna CB"],
        )
        warning = next(
            item for item in result.warnings
            if item["code"] == "preview_invalid_mentions_removed"
        )
        self.assertEqual(warning["items"][0]["model_text"], "invented blend")

    def test_case_and_whitespace_are_mapped_back_to_source(self) -> None:
        document = stage0_document(sections=("Abstract",))
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )

        result = extract_material_mentions(
            document,
            NormalizingClient(),
            prompt,
        )

        self.assertEqual(
            result.material_mentions[0].text,
            "cis-Polybutadiene rubber",
        )

    def test_compact_alphanumeric_label_maps_to_spaced_source(self) -> None:
        self.assertEqual(
            _resolve_surface_text("The polyamide 5 a was tested.", "5a"),
            "5 a",
        )

    def test_html_entity_surface_maps_back_to_literal_character(self) -> None:
        self.assertEqual(
            _resolve_surface_text(
                "<td>BTDA/4,4&#x27;-BABBP</td>",
                "BTDA/4,4'-BABBP",
            ),
            "BTDA/4,4'-BABBP",
        )

    def test_table_evidence_keeps_original_html_entity_surface(self) -> None:
        document_data = stage0_document(sections=("Abstract",)).model_dump(
            mode="json"
        )
        document_data["elements"].append({
            "block_id": "T_1_0",
            "type": "table",
            "section": "Results",
            "table_body": "<table><tr><td>BTDA/4,4&#x27;-BABBP</td></tr></table>",
            "page": 1,
            "bbox": [1, 2, 3, 4],
            "source_block_index": 3,
        })
        document = Stage0Document.model_validate(document_data)
        candidate = MentionCandidate.model_validate({
            "text": "BTDA/4,4'-BABBP",
            "mention_role": "sample_label",
            "block_id": "T_1_0",
            "confidence": {"score": 0.9},
        })

        mention = _materialize_mentions(
            [candidate],
            {item.block_id: item for item in document.elements},
        )[0]

        self.assertEqual(mention.text, "BTDA/4,4'-BABBP")
        self.assertEqual(
            mention.evidence.source_sentence,
            "BTDA/4,4&#x27;-BABBP",
        )

    def test_superscript_markup_inside_mention_is_recovered(self) -> None:
        """原文被 <sup> 切断时应恢复为原文形式（契约 §8.1 上下标等价）。

        真实用例 reference_no_0030496：模型输出 `poly( - olefin)s`，
        原文为 `poly( - <sup>R</sup>olefin)s`。
        """
        source = (
            "It was proposed that coil dimensions for "
            "poly( - <sup>R</sup>olefin)s is determined by the backbone."
        )
        self.assertEqual(
            _resolve_surface_text(source, "poly( - olefin)s"),
            "poly( - <sup>R</sup>olefin)s",
        )

    def test_markup_recovery_requires_unique_match(self) -> None:
        """多个候选时不得选择"最像"的一个，必须交由上层硬失败。"""
        source = (
            "poly( - <sup>R</sup>olefin)s here and "
            "poly( - <sup>R</sup>olefin)s there"
        )
        self.assertIsNone(_resolve_surface_text(source, "poly( - olefin)s"))

    def test_markup_recovery_does_not_splice_across_words(self) -> None:
        """禁止删除有语义的词后冒充原文（契约 §8.2）。

        真实用例 reference_no_0070031：模型输出跳过了
        "$\\alpha$ -methylstyrene and"，属语义拼接，必须硬失败。
        """
        source = (
            "the polymers of   $\\alpha$ -methylstyrene and   "
            "$\\alpha,\\beta,\\beta'$ -trifluorostyrene."
        )
        self.assertIsNone(
            _resolve_surface_text(
                source,
                "polymers of   $\\alpha,\\beta,\\beta'$ -trifluorostyrene",
            )
        )

    def test_numeric_mention_uses_token_boundary(self) -> None:
        text = "The recovered mass was 3.48 g. Product 8 was isolated."

        self.assertIsNone(_resolve_surface_text("The mass was 3.48 g.", "8"))
        self.assertEqual(
            _source_sentence(text, "8"),
            "Product 8 was isolated.",
        )

    def test_compatible_output_cache_is_reused(self) -> None:
        document = stage0_document(sections=("Abstract",))
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            output_path = root / "stage1_mentions.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )

            _, first_cached = run_stage1(
                stage0_path,
                output_path,
                client,
                prompt,
            )
            calls_after_first = client.calls
            _, second_cached = run_stage1(
                stage0_path,
                output_path,
                client,
                prompt,
            )

            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertEqual(client.calls, calls_after_first)

    def test_failure_writes_stage1_audit_artifact(self) -> None:
        document = stage0_document(sections=("Abstract",))
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            output_path = root / "stage1_mentions.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )

            with self.assertRaises(Stage1Error):
                run_stage1(
                    stage0_path,
                    output_path,
                    FailingClient(),
                    prompt,
                    max_validation_retries=0,
                )

            failure = json.loads(
                (root / "stage1_failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["stage"], "stage1_material_mention")
            self.assertEqual(failure["document_id"], "reference_no_0000001")
            self.assertEqual(failure["error_type"], "Stage1Error")

    def test_truncation_failure_is_marked_output_truncated(self) -> None:
        document = stage0_document(sections=("Abstract",))
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            output_path = root / "stage1_mentions.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )

            with self.assertRaises(Stage1Error):
                run_stage1(
                    stage0_path,
                    output_path,
                    TruncatedClient(),
                    prompt,
                    max_validation_retries=1,
                )

            failure = json.loads(
                (root / "stage1_failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["error_code"], "output_truncated")

    def test_no_supported_sections_uses_all_nonempty_blocks(self) -> None:
        document = stage0_document(sections=("DocumentTitle",))

        blocks, warnings = select_input_blocks(document)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(warnings[0]["code"], "unsectioned_blocks_fallback")

    def test_no_nonempty_text_title_or_table_still_fails(self) -> None:
        data = stage0_document(sections=("DocumentTitle",)).model_dump(
            mode="json"
        )
        for element in data["elements"]:
            element["text"] = ""
        document = Stage0Document.model_validate(data)

        with self.assertRaises(Stage1Error):
            select_input_blocks(document)

    def test_chunking_preserves_order(self) -> None:
        document = stage0_document()
        blocks, _ = select_input_blocks(document)

        chunks = chunk_blocks(blocks, max_chunk_chars=2000)

        self.assertEqual(
            [block.block_id for chunk in chunks for block in chunk],
            ["P_0_0", "P_0_1"],
        )


    def test_single_chunk_failure_response_can_be_replayed_offline(self) -> None:
        document = stage0_document(sections=("Abstract",))
        response_data = add_model_confidence({
            "mentions": [{
                "block_id": "P_0_0",
                "text": "Buna CB",
                "mention_role": "commercial_name",
            }]
        })
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            failure_path = root / "stage1_failure.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            failure_path.write_text(
                json.dumps({
                    "raw_response": {
                        "provider": "test",
                        "model": "saved-model",
                        "content": json.dumps(response_data),
                        "usage": {},
                    }
                }),
                encoding="utf-8",
            )

            client = _failure_replay_client(
                failure_path,
                load_pipeline_config(DEFAULT_CONFIG_PATH),
                stage0_path=stage0_path,
                max_chunk_chars=8000,
                primary_sections=("Abstract", "Methods", "Results"),
                fallback_sections=("Introduction", "Conclusion"),
            )
            result = extract_material_mentions(
                document,
                client,
                prompt,
                max_validation_retries=0,
            )

        self.assertEqual(client.calls, 1)
        self.assertEqual(result.material_mentions[0].text, "Buna CB")
        with self.assertRaises(Stage1Error):
            client.call_json("system", "user")

    def test_multiple_chunks_are_not_replayed_from_one_saved_response(self) -> None:
        data = stage0_document().model_dump(mode="json")
        data["elements"][0]["text"] = "A" * 1500
        data["elements"][1]["text"] = "B" * 1500
        document = Stage0Document.model_validate(data)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            failure_path = root / "stage1_failure.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            failure_path.write_text(
                json.dumps({"raw_response": {"content": "{}"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(Stage1Error, "2 个 chunk"):
                _failure_replay_client(
                    failure_path,
                    load_pipeline_config(DEFAULT_CONFIG_PATH),
                    stage0_path=stage0_path,
                    max_chunk_chars=2000,
                    primary_sections=("Abstract", "Methods", "Results"),
                    fallback_sections=("Introduction", "Conclusion"),
                )

    def test_failure_without_raw_response_is_not_replayable(self) -> None:
        document = stage0_document(sections=("Abstract",))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            failure_path = root / "stage1_failure.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            failure_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(Stage1Error, "未保存可回放"):
                _failure_replay_client(
                    failure_path,
                    load_pipeline_config(DEFAULT_CONFIG_PATH),
                    stage0_path=stage0_path,
                    max_chunk_chars=8000,
                    primary_sections=("Abstract", "Methods", "Results"),
                    fallback_sections=("Introduction", "Conclusion"),
                )

    def test_replay_validation_failure_does_not_overwrite_original_failure(self) -> None:
        document = stage0_document(sections=("Abstract",))
        prompt = PromptLoader().render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        invalid_response = add_model_confidence({
            "mentions": [{
                "block_id": "not-in-input",
                "text": "PB",
                "mention_role": "abbreviation",
            }]
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            output_path = root / "stage1_mentions.json"
            failure_path = root / "stage1_failure.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            original_failure = json.dumps({
                "marker": "original",
                "raw_response": {
                    "provider": "test",
                    "model": "saved-model",
                    "content": json.dumps(invalid_response),
                    "usage": {},
                },
            })
            failure_path.write_text(original_failure, encoding="utf-8")
            client = _failure_replay_client(
                failure_path,
                load_pipeline_config(DEFAULT_CONFIG_PATH),
                stage0_path=stage0_path,
                max_chunk_chars=8000,
                primary_sections=("Abstract", "Methods", "Results"),
                fallback_sections=("Introduction", "Conclusion"),
            )

            with self.assertRaises(Stage1Error):
                run_stage1(
                    stage0_path,
                    output_path,
                    client,
                    prompt,
                    force=True,
                    max_validation_retries=0,
                    record_failure=False,
                )

            self.assertEqual(
                failure_path.read_text(encoding="utf-8"),
                original_failure,
            )
