import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


from llm_client import (
    DEFAULT_CONFIG_PATH,
    LLMJSONResponse,
    ResolvedLLMConfig,
    load_pipeline_config,
)
from prompt_loader import PromptLoader
from schema.polymer_schema import (
    PolymerEntityResponse,
    Stage0Document,
    Stage1Document,
)
from tests.helpers import add_model_confidence
from stages.stage2_polymer_entity import (
    Stage2Error,
    _element_source_text,
    _failure_replay_client,
    _materialize_entities,
    _preferred_polymer_name_mention,
    extract_polymer_entities,
    run_stage2,
    select_context_blocks,
)


def stage0_document(
    *,
    method_section: str = "Methods",
) -> Stage0Document:
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
        "elements": [
            {
                "block_id": "P_0_0",
                "type": "text",
                "section": "Abstract",
                "text": "Polybutadiene (PB) was studied.",
                "page": 0,
                "bbox": [1, 2, 3, 4],
                "source_block_index": 0,
            },
            {
                "block_id": "P_1_0",
                "type": "text",
                "section": method_section,
                "text": (
                    "Polybutadiene was oxidized to produce oxidized "
                    "polybutadiene (OPB)."
                ),
                "page": 1,
                "bbox": [5, 6, 7, 8],
                "source_block_index": 1,
            },
            {
                "block_id": "I_1_0",
                "type": "image",
                "section": method_section,
                "page": 1,
                "bbox": [10, 20, 30, 40],
                "source_block_index": 2,
                "caption": "Scheme 1. Oxidized polybutadiene.",
                "image_path": "images/scheme_1.jpg",
            },
        ],
        "warnings": [],
    })


def stage1_document() -> Stage1Document:
    digest = "a" * 64
    return Stage1Document.model_validate({
        "schema_version": "1.0",
        "document_id": "reference_no_0000001",
        "material_mentions": [
            {
                "mention_id": "m001",
                "text": "Polybutadiene",
                "mention_role": "polymer_name",
                "evidence": {
                    "block_id": "P_0_0",
                    "page": 0,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "text",
                    "source_sentence": "Polybutadiene (PB) was studied.",
                },
            },
            {
                "mention_id": "m002",
                "text": "PB",
                "mention_role": "abbreviation",
                "evidence": {
                    "block_id": "P_0_0",
                    "page": 0,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "text",
                    "source_sentence": "Polybutadiene (PB) was studied.",
                },
            },
            {
                "mention_id": "m003",
                "text": "oxidized polybutadiene",
                "mention_role": "polymer_name",
                "evidence": {
                    "block_id": "P_1_0",
                    "page": 1,
                    "bbox": [5, 6, 7, 8],
                    "source_type": "text",
                    "source_sentence": (
                        "Polybutadiene was oxidized to produce oxidized "
                        "polybutadiene (OPB)."
                    ),
                },
            },
        ],
        "provenance": {
            "provider": "test",
            "model": "fake",
            "models": ["fake"],
            "prompt_id": "polymer.stage1.material_mention",
            "prompt_version": "1.0.0",
            "prompt_sha256": digest,
            "input_hash": digest,
            "model_config_hash": digest,
            "cache_key": digest,
            "output_schema_version": "material_mention_schema.v1",
            "implementation_version": "1.0.0",
            "chunk_count": 1,
        },
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
                "entities": [
                    {
                        "entity_id": "pe010",
                        "polymer_name": "Polybutadiene",
                        "polymer_type": None,
                        "variant_of": None,
                        "structural_features": [],
                        "resolved_from_mentions": ["m001", "m002"],
                        "evidence": {
                            "block_id": "P_0_0",
                            "source_sentence": "Polybutadiene (PB) was studied.",
                        },
                        "source_image_block_ids": [],
                    },
                    {
                        "entity_id": "pe020",
                        "polymer_name": "oxidized polybutadiene",
                        "polymer_type": None,
                        "variant_of": "pe010",
                        "structural_features": [],
                        "resolved_from_mentions": ["m003"],
                        "evidence": {
                            "block_id": "P_1_0",
                            "source_sentence": (
                                "Polybutadiene was oxidized to produce "
                                "oxidized polybutadiene (OPB)."
                            ),
                        },
                        "source_image_block_ids": ["I_1_0"],
                    },
                ],
                "unresolved_mention_ids": [],
            }),
            provider="test",
            model="fake-actual",
        )


class LegacyConfidenceClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        response = super().call_json(
            system_prompt,
            user_message,
            max_tokens=max_tokens,
        )
        response.data["entities"][0]["confidence"].update({
            "field_scores": {"mention_resolution": 0.7},
            "uncertain_fields": ["mention_resolution"],
            "evidence_basis": ["explicit_text"],
            "uncertainty_codes": [],
        })
        return response


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
                    "entities": [],
                    "unresolved_mention_ids": ["m001"],
                }),
                provider="test",
                model="fake-actual",
            )
        self.calls -= 1
        return super().call_json(
            system_prompt,
            user_message,
            max_tokens=max_tokens,
        )


class NestedSplitClient(FakeClient):
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
                "entities": [
                    {
                        "entity_id": "pe001",
                        "polymer_name": "cis-Polybutadiene rubber",
                        "polymer_type": None,
                        "variant_of": None,
                        "structural_features": [],
                        "resolved_from_mentions": ["m001"],
                        "evidence": {
                            "block_id": "P_0_0",
                            "source_sentence": (
                                "The cis-Polybutadiene rubber was tested."
                            ),
                        },
                        "source_image_block_ids": [],
                    },
                    {
                        "entity_id": "pe002",
                        "polymer_name": "Polybutadiene",
                        "polymer_type": None,
                        "variant_of": None,
                        "structural_features": [],
                        "resolved_from_mentions": ["m002"],
                        "evidence": {
                            "block_id": "P_0_0",
                            "source_sentence": (
                                "The cis-Polybutadiene rubber was tested."
                            ),
                        },
                        "source_image_block_ids": [],
                    },
                ],
                "unresolved_mention_ids": [],
            }),
            provider="test",
            model="fake-actual",
        )


class BlendNestedSplitClient(FakeClient):
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
                "entities": [
                    {
                        "entity_id": "pe001",
                        "polymer_name": "GUR 415",
                        "polymer_type": None,
                        "variant_of": None,
                        "structural_features": [],
                        "resolved_from_mentions": ["m001"],
                        "evidence": {
                            "block_id": "P_0_0",
                            "source_sentence": "GUR 415–PIR was tested.",
                        },
                        "source_image_block_ids": [],
                    },
                    {
                        "entity_id": "pe002",
                        "polymer_name": "GUR 415–PIR",
                        "polymer_type": "blend",
                        "variant_of": None,
                        "structural_features": [],
                        "resolved_from_mentions": ["m002"],
                        "evidence": {
                            "block_id": "P_0_0",
                            "source_sentence": "GUR 415–PIR was tested.",
                        },
                        "source_image_block_ids": [],
                    },
                ],
                "unresolved_mention_ids": [],
            }),
            provider="test",
            model="fake-actual",
        )


class SurfaceEvidenceClient(FakeClient):
    def __init__(self, source_sentence: str) -> None:
        super().__init__()
        self.source_sentence = source_sentence

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
                "entities": [{
                    "entity_id": "pe001",
                    "polymer_name": "Polybutadiene",
                    "polymer_type": None,
                    "variant_of": None,
                    "structural_features": [],
                    "resolved_from_mentions": ["m001"],
                    "evidence": {
                        "block_id": "P_0_0",
                        "source_sentence": self.source_sentence,
                    },
                    "source_image_block_ids": [],
                }],
                "unresolved_mention_ids": ["m002", "m003"],
            }),
            provider="test",
            model="fake-actual",
        )


class InvalidNameClient(SurfaceEvidenceClient):
    def call_json(self, *args, **kwargs) -> LLMJSONResponse:
        response = super().call_json(*args, **kwargs)
        response.data["entities"][0]["polymer_name"] = "invented blend"
        return response


def rendered_prompt():
    return PromptLoader().render_stage_prompt(
        "polymer.stage2.polymer_entity",
        PolymerEntityResponse,
        expected_stage="stage2_polymer_entity",
        expected_output_schema="polymer_entity_schema.v2",
    )


class Stage2Tests(unittest.TestCase):
    def test_duplicate_mention_with_unique_evidence_owner_fails_strict(self) -> None:
        client = FakeClient()
        original_call = client.call_json

        def call_with_duplicate(*args, **kwargs):
            response = original_call(*args, **kwargs)
            response.data["entities"][0]["resolved_from_mentions"].append("m003")
            return response

        client.call_json = call_with_duplicate
        with self.assertRaisesRegex(Stage2Error, "同一 mention"):
            extract_polymer_entities(
                stage0_document(),
                stage1_document(),
                client,
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_duplicate_mention_without_unique_owner_fails_strict(self) -> None:
        client = FakeClient()
        original_call = client.call_json

        def call_with_ambiguous_duplicate(*args, **kwargs):
            response = original_call(*args, **kwargs)
            response.data["entities"][1]["resolved_from_mentions"].append("m002")
            return response

        client.call_json = call_with_ambiguous_duplicate
        with self.assertRaisesRegex(Stage2Error, "同一 mention"):
            extract_polymer_entities(
                stage0_document(),
                stage1_document(),
                client,
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_preview_keeps_unique_text_and_evidence_owner(self) -> None:
        client = FakeClient()
        original_call = client.call_json

        def call_with_duplicate(*args, **kwargs):
            response = original_call(*args, **kwargs)
            response.data["entities"][0]["resolved_from_mentions"].append("m003")
            return response

        client.call_json = call_with_duplicate
        result = extract_polymer_entities(
            stage0_document(),
            stage1_document(),
            client,
            rendered_prompt(),
            max_validation_retries=0,
            preview_relaxed=True,
        )

        entities = {
            item.polymer_name: item for item in result.polymer_entities
        }
        self.assertNotIn(
            "m003",
            entities["Polybutadiene"].resolved_from_mentions,
        )
        self.assertIn(
            "m003",
            entities["oxidized polybutadiene"].resolved_from_mentions,
        )
        warning = next(
            item for item in result.warnings
            if item["code"] == "preview_duplicate_mention_recovered"
        )
        self.assertEqual(
            warning["items"][0]["action"],
            "kept_unique_text_and_evidence_owner",
        )
        self.assertEqual(warning["items"][0]["kept_entity_id"], "pe020")

    def test_preview_marks_ambiguous_duplicate_unresolved(self) -> None:
        client = FakeClient()
        original_call = client.call_json

        def call_with_ambiguous_duplicate(*args, **kwargs):
            response = original_call(*args, **kwargs)
            response.data["entities"][1]["resolved_from_mentions"].append("m002")
            return response

        client.call_json = call_with_ambiguous_duplicate
        result = extract_polymer_entities(
            stage0_document(),
            stage1_document(),
            client,
            rendered_prompt(),
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertIn("m002", result.unresolved_mention_ids)
        self.assertTrue(all(
            "m002" not in entity.resolved_from_mentions
            for entity in result.polymer_entities
        ))
        warning = next(
            item for item in result.warnings
            if item["code"] == "preview_duplicate_mention_recovered"
        )
        self.assertEqual(
            warning["items"][0]["action"],
            "marked_unresolved",
        )
        self.assertIsNone(warning["items"][0]["kept_entity_id"])

    def test_failure_response_can_be_replayed_without_network(self) -> None:
        response = FakeClient().call_json("", "")
        with tempfile.TemporaryDirectory() as temp_dir:
            failure_path = Path(temp_dir) / "stage2_failure.json"
            failure_path.write_text(
                json.dumps({
                    "raw_response": {
                        "provider": response.provider,
                        "model": response.model,
                        "content": json.dumps(response.data),
                        "usage": {},
                    }
                }),
                encoding="utf-8",
            )
            client = _failure_replay_client(
                failure_path,
                load_pipeline_config(DEFAULT_CONFIG_PATH),
            )
            result = extract_polymer_entities(
                stage0_document(),
                stage1_document(),
                client,
                rendered_prompt(),
                max_validation_retries=0,
            )

        self.assertEqual(client.calls, 1)
        self.assertTrue(any(
            item["code"] == "failure_response_replayed"
            for item in result.warnings
        ))

    def test_table_source_text_includes_caption_and_body(self) -> None:
        data = stage0_document().model_dump(mode="json")
        data["elements"].append({
            "block_id": "T_2_0",
            "type": "table",
            "section": "Results",
            "caption": "Table 1. Polymer properties",
            "table_body": "<table><tr><td>PB</td></tr></table>",
            "page": 2,
            "bbox": [1, 2, 3, 4],
            "source_block_index": 3,
        })
        document = Stage0Document.model_validate(data)
        table = document.elements[-1]

        source_text = _element_source_text(table)

        self.assertIn("Table 1. Polymer properties", source_text)
        self.assertIn("<td>PB</td>", source_text)

    def test_legacy_confidence_details_are_compacted_with_warning(self) -> None:
        result = extract_polymer_entities(
            stage0_document(),
            stage1_document(),
            LegacyConfidenceClient(),
            PromptLoader().render_stage_prompt(
                "polymer.stage2.polymer_entity",
                PolymerEntityResponse,
                expected_stage="stage2_polymer_entity",
                expected_output_schema="polymer_entity_schema.v2",
            ),
        )

        self.assertEqual(result.polymer_entities[0].confidence.model_dump(), {"score": 0.9})
        warning = next(
            item for item in result.warnings
            if item["code"] == "confidence_fields_compacted"
        )
        self.assertIn(
            "entities[0].confidence.uncertain_fields",
            warning["fields"],
        )

    def test_missing_mention_is_marked_unresolved_without_retry(self) -> None:
        document = stage0_document()
        mentions = stage1_document()
        client = FakeClient()
        original_call = client.call_json

        def call_without_last_mention(*args, **kwargs):
            response = original_call(*args, **kwargs)
            response.data["entities"][0]["resolved_from_mentions"].remove("m002")
            return response

        client.call_json = call_without_last_mention
        result = extract_polymer_entities(
            document,
            mentions,
            client,
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertIn("m002", result.unresolved_mention_ids)
        self.assertTrue(any(
            warning["code"] == "missing_mentions_marked_unresolved"
            for warning in result.warnings
        ))

    def test_extracts_entities_remaps_ids_and_preserves_image_ref(self) -> None:
        result = extract_polymer_entities(
            stage0_document(),
            stage1_document(),
            FakeClient(),
            rendered_prompt(),
        )

        self.assertEqual(
            [entity.entity_id for entity in result.polymer_entities],
            ["pe001", "pe002"],
        )
        self.assertEqual(result.polymer_entities[1].variant_of, "pe001")
        self.assertEqual(
            result.polymer_entities[1].source_image_refs[0].image_path,
            "images/scheme_1.jpg",
        )
        self.assertEqual(
            result.polymer_entities[0].source_names,
            ["Polybutadiene", "PB"],
        )
        self.assertEqual(
            result.polymer_entities[0].representation_status,
            "expert_review_required",
        )
        self.assertEqual(result.provenance.model, "fake-actual")

    def test_missing_mention_coverage_does_not_retry(self) -> None:
        client = RetryClient()

        result = extract_polymer_entities(
            stage0_document(),
            stage1_document(),
            client,
            rendered_prompt(),
            max_validation_retries=1,
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(
            result.unresolved_mention_ids,
            ["m001", "m002", "m003"],
        )

    def test_nested_surface_mention_cannot_be_split(self) -> None:
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"][0]["text"] = (
            "The cis-Polybutadiene rubber was tested."
        )
        document = Stage0Document.model_validate(document_data)
        mention_data = stage1_document().model_dump(mode="json")
        mention_data["material_mentions"] = [
            {
                "mention_id": "m001",
                "text": "cis-Polybutadiene rubber",
                "mention_role": "polymer_name",
                "evidence": {
                    "block_id": "P_0_0",
                    "page": 0,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "text",
                    "source_sentence": (
                        "The cis-Polybutadiene rubber was tested."
                    ),
                },
            },
            {
                "mention_id": "m002",
                "text": "Polybutadiene",
                "mention_role": "polymer_name",
                "evidence": {
                    "block_id": "P_0_0",
                    "page": 0,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "text",
                    "source_sentence": (
                        "The cis-Polybutadiene rubber was tested."
                    ),
                },
            },
        ]
        mentions = Stage1Document.model_validate(mention_data)

        with self.assertRaises(Stage2Error):
            extract_polymer_entities(
                document,
                mentions,
                NestedSplitClient(),
                rendered_prompt(),
                max_validation_retries=0,
            )

        result = extract_polymer_entities(
            document,
            mentions,
            NestedSplitClient(),
            rendered_prompt(),
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(len(result.polymer_entities), 2)
        warning = next(
            item for item in result.warnings
            if item["code"] == "preview_nested_mentions_split_retained"
        )
        self.assertEqual(warning["items"][0]["shorter_mention_id"], "m002")
        self.assertEqual(warning["items"][0]["longer_mention_id"], "m001")

    def test_component_and_blend_nested_mentions_can_be_split(self) -> None:
        source_sentence = "GUR 415–PIR was tested."
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"][0]["text"] = source_sentence
        document = Stage0Document.model_validate(document_data)
        mention_data = stage1_document().model_dump(mode="json")
        mention_data["material_mentions"] = [
            {
                "mention_id": "m001",
                "text": "GUR 415",
                "mention_role": "commercial_name",
                "evidence": {
                    "block_id": "P_0_0",
                    "page": 0,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "text",
                    "source_sentence": source_sentence,
                },
            },
            {
                "mention_id": "m002",
                "text": "GUR 415–PIR",
                "mention_role": "sample_label",
                "evidence": {
                    "block_id": "P_0_0",
                    "page": 0,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "text",
                    "source_sentence": source_sentence,
                },
            },
        ]
        mentions = Stage1Document.model_validate(mention_data)

        result = extract_polymer_entities(
            document,
            mentions,
            BlendNestedSplitClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(len(result.polymer_entities), 2)
        warning = next(
            item for item in result.warnings
            if item["code"] == "component_blend_nested_mentions_split"
        )
        self.assertEqual(warning["items"][0]["shorter_mention_id"], "m001")
        self.assertEqual(warning["items"][0]["longer_mention_id"], "m002")

    def test_latex_whitespace_surface_is_restored_from_source(self) -> None:
        source_sentence = (
            r"Polybutadiene was heated at $7 5 \mathrm { { ^ \circ C } }$"
            "for 24 h."
        )
        model_sentence = (
            r"Polybutadiene was heated at $7 5 \mathrm { { ^\circ C } }$"
            "for 24 h."
        )
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"][0]["text"] = source_sentence
        document = Stage0Document.model_validate(document_data)

        result = extract_polymer_entities(
            document,
            stage1_document(),
            SurfaceEvidenceClient(model_sentence),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(
            result.polymer_entities[0].evidence.source_sentence,
            source_sentence,
        )
        self.assertTrue(any(
            item["code"] == "evidence_surface_whitespace_recovered"
            for item in result.warnings
        ))

    def test_non_whitespace_surface_difference_is_rejected(self) -> None:
        source_sentence = (
            r"Polybutadiene was heated at $7 5 \mathrm { { ^ \circ C } }$"
            "for 24 h."
        )
        model_sentence = (
            r"Polybutadiene was heated at $7 5 \mathrm { { ^\circ K } }$"
            "for 24 h."
        )
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"][0]["text"] = source_sentence
        document = Stage0Document.model_validate(document_data)

        with self.assertRaises(Stage2Error):
            extract_polymer_entities(
                document,
                stage1_document(),
                SurfaceEvidenceClient(model_sentence),
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_preview_inherits_entity_evidence_from_matching_mention(self) -> None:
        result = extract_polymer_entities(
            stage0_document(),
            stage1_document(),
            SurfaceEvidenceClient("This sentence is not in the source."),
            rendered_prompt(),
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(
            result.polymer_entities[0].evidence.source_sentence,
            stage1_document().material_mentions[0].evidence.source_sentence,
        )
        self.assertTrue(any(
            item["code"]
            == "preview_entity_evidence_inherited_from_mention"
            for item in result.warnings
        ))

    def test_preview_removes_entity_with_invented_name(self) -> None:
        result = extract_polymer_entities(
            stage0_document(),
            stage1_document(),
            InvalidNameClient("Polybutadiene (PB) was studied."),
            rendered_prompt(),
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(result.polymer_entities, [])
        self.assertIn("m001", result.unresolved_mention_ids)
        self.assertTrue(any(
            item["code"] == "preview_invalid_entities_removed"
            for item in result.warnings
        ))

    def test_materialize_prefers_specific_name_over_code_label(self) -> None:
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"][0]["text"] = (
            "Polycarbonate based on bisphenol A (PC-1) was studied."
        )
        document = Stage0Document.model_validate(document_data)

        mention_data = stage1_document().model_dump(mode="json")
        mention_data["material_mentions"] = [
            {
                "mention_id": "m001",
                "text": "Polycarbonate based on bisphenol A",
                "mention_role": "polymer_name",
                "evidence": {
                    "block_id": "P_0_0",
                    "page": 0,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "text",
                    "source_sentence": (
                        "Polycarbonate based on bisphenol A (PC-1) was studied."
                    ),
                },
            },
            {
                "mention_id": "m002",
                "text": "PC-1",
                "mention_role": "sample_label",
                "evidence": {
                    "block_id": "P_0_0",
                    "page": 0,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "text",
                    "source_sentence": (
                        "Polycarbonate based on bisphenol A (PC-1) was studied."
                    ),
                },
            },
        ]
        mentions = Stage1Document.model_validate(mention_data)
        parsed = PolymerEntityResponse.model_validate({
            "entities": [{
                "entity_id": "pe001",
                "polymer_name": "PC-1",
                "polymer_type": "homopolymer",
                "variant_of": None,
                "structural_features": [],
                "resolved_from_mentions": ["m001", "m002"],
                "evidence": {
                    "block_id": "P_0_0",
                    "source_sentence": (
                        "Polycarbonate based on bisphenol A (PC-1) was studied."
                    ),
                },
                "source_image_block_ids": [],
                "confidence": {"score": 0.9},
            }],
            "unresolved_mention_ids": [],
        })
        repairs: list[dict[str, str]] = []

        entities = _materialize_entities(
            parsed,
            document.elements,
            mentions,
            repairs,
        )

        self.assertEqual(
            entities[0].polymer_name,
            "Polycarbonate based on bisphenol A",
        )
        self.assertEqual(
            entities[0].evidence.source_sentence,
            "Polycarbonate based on bisphenol A (PC-1) was studied.",
        )
        self.assertEqual(repairs[0]["previous_name"], "PC-1")
        self.assertEqual(repairs[0]["mention_id"], "m001")

    def test_materialize_keeps_code_when_no_specific_name_exists(self) -> None:
        document = stage0_document()
        mention_data = stage1_document().model_dump(mode="json")
        mention_data["material_mentions"] = [{
            "mention_id": "m001",
            "text": "PC-1",
            "mention_role": "sample_label",
            "evidence": {
                "block_id": "P_0_0",
                "page": 0,
                "bbox": [1, 2, 3, 4],
                "source_type": "text",
                "source_sentence": "Polybutadiene (PB) was studied.",
            },
        }]
        mentions = Stage1Document.model_validate(mention_data)
        parsed = PolymerEntityResponse.model_validate({
            "entities": [{
                "entity_id": "pe001",
                "polymer_name": "PC-1",
                "polymer_type": None,
                "variant_of": None,
                "structural_features": [],
                "resolved_from_mentions": ["m001"],
                "evidence": {
                    "block_id": "P_0_0",
                    "source_sentence": "Polybutadiene (PB) was studied.",
                },
                "source_image_block_ids": [],
                "confidence": {"score": 0.8},
            }],
            "unresolved_mention_ids": [],
        })
        repairs: list[dict[str, str]] = []

        entities = _materialize_entities(
            parsed,
            document.elements,
            mentions,
            repairs,
        )

        self.assertEqual(entities[0].polymer_name, "PC-1")
        self.assertEqual(repairs, [])

    def test_preferred_name_supports_conservative_additional_code_forms(self) -> None:
        from types import SimpleNamespace

        cases = [
            ("PTh", "polythiophene", True),
            ("NBR", "アクリロニトリル-ブタジエンゴム", True),
            ("8b", "poly(4-vinyltriphenylamine)", True),
            ("9a", "polystyrene", True),
            ("PVC/ABS/SMIA", "PVC/ABS/SMIA composites", True),
            ("HS", "aliphatic DA-polyester", False),
            ("1AQA-PPDI", "polyurea", False),
            ("Ia", "polymer Ia", False),
            ("0-2-0-I", "ordered alternating polyhydrazide", False),
        ]
        for current_name, mention_name, should_replace in cases:
            with self.subTest(current_name=current_name):
                candidate = SimpleNamespace(
                    polymer_name=current_name,
                    resolved_from_mentions=["m001"],
                )
                mention = SimpleNamespace(
                    mention_id="m001",
                    mention_role="polymer_name",
                    text=mention_name,
                )

                preferred = _preferred_polymer_name_mention(
                    candidate,
                    {"m001": mention},
                )

                if should_replace:
                    self.assertIs(preferred, mention)
                else:
                    self.assertIsNone(preferred)

    def test_context_falls_back_to_mention_evidence(self) -> None:
        blocks, warnings, _ = select_context_blocks(
            stage0_document(method_section="Introduction"),
            stage1_document(),
        )

        self.assertEqual(
            [block.block_id for block in blocks],
            ["P_0_0", "P_1_0"],
        )
        self.assertEqual(warnings[0]["code"], "section_fallback")

    def test_context_limit_fails_without_silent_truncation(self) -> None:
        document = stage0_document()
        long_method = document.elements[1].model_copy(
            update={"text": "Polybutadiene " * 200}
        )
        document = document.model_copy(
            update={
                "elements": [
                    document.elements[0],
                    long_method,
                    document.elements[2],
                ]
            }
        )

        with self.assertRaises(Stage2Error):
            select_context_blocks(
                document,
                stage1_document(),
                max_input_chars=2000,
            )

    def test_compatible_output_cache_is_reused(self) -> None:
        document = stage0_document()
        mentions = stage1_document()
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage1_path = root / "stage1_mentions.json"
            output_path = root / "stage2_entities.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage1_path.write_text(
                json.dumps(mentions.model_dump(mode="json")),
                encoding="utf-8",
            )

            _, first_cached = run_stage2(
                stage0_path,
                stage1_path,
                output_path,
                client,
                rendered_prompt(),
            )
            calls_after_first = client.calls
            _, second_cached = run_stage2(
                stage0_path,
                stage1_path,
                output_path,
                client,
                rendered_prompt(),
            )

            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertEqual(client.calls, calls_after_first)

    def test_prompt_version_change_invalidates_cache(self) -> None:
        document = stage0_document()
        mentions = stage1_document()
        client = FakeClient()
        prompt = rendered_prompt()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage1_path = root / "stage1_mentions.json"
            output_path = root / "stage2_entities.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage1_path.write_text(
                json.dumps(mentions.model_dump(mode="json")),
                encoding="utf-8",
            )

            run_stage2(
                stage0_path,
                stage1_path,
                output_path,
                client,
                prompt,
            )
            calls_after_first = client.calls
            _, cached = run_stage2(
                stage0_path,
                stage1_path,
                output_path,
                client,
                replace(prompt, version="9.9.9"),
            )

            self.assertFalse(cached)
            self.assertGreater(client.calls, calls_after_first)

    def test_document_id_mismatch_fails(self) -> None:
        mentions = stage1_document().model_copy(
            update={"document_id": "reference_no_other"}
        )

        with self.assertRaises(Stage2Error):
            extract_polymer_entities(
                stage0_document(),
                mentions,
                FakeClient(),
                rendered_prompt(),
            )


if __name__ == "__main__":
    unittest.main()
