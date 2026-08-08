import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from llm_client import (
    DEFAULT_CONFIG_PATH,
    LLMCallRecord,
    LLMJSONResponse,
    LLMRawResponse,
    LLMTokenUsage,
    ResolvedLLMConfig,
    load_pipeline_config,
)
from prompt_loader import PromptLoader
from schema.polymer_schema import (
    CharacterizationStageResponse,
    Stage5PropertyCandidate,
    Stage0Document,
    Stage2Document,
    Stage3Document,
    Stage4Document,
)
from tests.helpers import add_model_confidence
from stages.stage5_characterization import (
    DEFAULT_VOCABULARY_PATH,
    Stage5Error,
    _failure_replay_client,
    _method_names_for_raw,
    _normalize_stage5_property,
    _repair_candidate_response_payload,
    _stage5_property_dedupe_key,
    extract_characterizations,
    load_characterization_vocabulary,
    run_stage5,
)
import stages.stage5_characterization as stage5_module


METHOD_SENTENCE = (
    "FTIR spectra were recorded over 4000-400 cm-1, and DSC measurements "
    "were conducted."
)
FTIR_SENTENCE = (
    "The FTIR spectrum of dried PB film showed an absorption band at "
    "1650 cm-1 assigned to C=C stretching."
)
DSC_SENTENCE = (
    "The glass transition temperature of dried PB film was -85 °C by DSC."
)
# 第二次 FTIR：同一 sample/entity，不同制样方式与不同吸收峰。
# 对应 reference_no_0071569 的薄膜 IR 与 KBr 压片 IR。
SECOND_FTIR_SENTENCE = (
    "The FTIR spectrum of the same sample as a KBr disk showed bands at "
    "1760 cm-1 and 840 cm-1."
)


def stage0_document() -> Stage0Document:
    return Stage0Document.model_validate({
        "schema_version": "1.0",
        "source_document_schema_version": "1.0",
        "document_id": "reference_no_0000002",
        "paper": {
            "ref_no": "reference_no_0000002",
            "pdf_filename": "uuid_origin.pdf",
            "source_pdf_path": "mineru_output/reference_no_0000002/uuid_origin.pdf",
            "organized_pdf_path": "wenxian/reference_no_0000002/origin.pdf",
            "doi": None,
            "title": "Characterization demo",
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
                "block_id": "P_1_0",
                "type": "text",
                "section": "Methods",
                "text": METHOD_SENTENCE,
                "page": 1,
                "bbox": [1, 2, 3, 4],
                "source_block_index": 0,
            },
            {
                "block_id": "P_2_0",
                "type": "text",
                "section": "Results",
                "text": FTIR_SENTENCE,
                "page": 2,
                "bbox": [5, 6, 7, 8],
                "source_block_index": 1,
            },
            {
                "block_id": "P_2_1",
                "type": "text",
                "section": "Results",
                "text": DSC_SENTENCE,
                "page": 2,
                "bbox": [9, 10, 11, 12],
                "source_block_index": 2,
            },
            {
                "block_id": "P_2_2",
                "type": "text",
                "section": "Results",
                "text": SECOND_FTIR_SENTENCE,
                "page": 2,
                "bbox": [13, 14, 15, 16],
                "source_block_index": 3,
            },
        ],
        "warnings": [],
    })


def stage2_document() -> Stage2Document:
    digest = "a" * 64
    return Stage2Document.model_validate({
        "schema_version": "1.0",
        "document_id": "reference_no_0000002",
        "polymer_entities": [{
            "entity_id": "pe001",
            "polymer_name": "polybutadiene",
            "polymer_type": None,
            "variant_of": None,
            "representation_status": "expert_review_required",
            "structural_features": [],
            "source_names": ["polybutadiene", "PB"],
            "resolved_from_mentions": ["m001"],
            "evidence": {
                "block_id": "P_2_0",
                "page": 2,
                "bbox": [5, 6, 7, 8],
                "source_type": "text",
                "source_sentence": FTIR_SENTENCE,
            },
            "source_image_refs": [],
        }],
        "unresolved_mention_ids": [],
        "provenance": {
            "provider": "test",
            "model": "fake",
            "models": ["fake"],
            "prompt_id": "polymer.stage2.polymer_entity",
            "prompt_version": "1.0.1",
            "prompt_sha256": digest,
            "input_hash": digest,
            "model_config_hash": digest,
            "cache_key": digest,
            "output_schema_version": "polymer_entity_schema.v1",
            "implementation_version": "1.1.0",
            "context_block_count": 1,
            "context_chars": 100,
            "call_count": 1,
        },
        "warnings": [],
    })


def stage3_document() -> Stage3Document:
    digest = "b" * 64
    return Stage3Document.model_validate({
        "schema_version": "1.0",
        "document_id": "reference_no_0000002",
        "samples": [{
            "sample_id": "s001",
            "sample_kind": "processed_material",
            "refers_to_entity": "pe001",
            "polymer_name": "polybutadiene",
            "sample_label_raw": "dried PB film",
            "state_description": None,
            "intended_use": [],
            "evidence": {
                "block_id": "P_2_0",
                "page": 2,
                "bbox": [5, 6, 7, 8],
                "source_type": "text",
                "source_sentence": FTIR_SENTENCE,
            },
        }],
        "process_steps": [],
        "unresolved_entity_ids": [],
        "provenance": {
            "provider": "test",
            "model": "fake",
            "models": ["fake"],
            "prompt_id": "polymer.stage3.sample_process",
            "prompt_version": "1.0.2",
            "prompt_sha256": digest,
            "input_hash": digest,
            "model_config_hash": digest,
            "cache_key": digest,
            "output_schema_version": "sample_process_schema.v1",
            "implementation_version": "1.0.0",
            "context_block_count": 1,
            "context_chars": 100,
            "call_count": 1,
        },
        "warnings": [],
    })


def stage4_document(
    *,
    determination_method_raw: str | None = "DSC",
) -> Stage4Document:
    digest = "c" * 64
    return Stage4Document.model_validate({
        "schema_version": "1.0",
        "document_id": "reference_no_0000002",
        "measurement_conditions": [{
            "condition_id": "mc001",
            "temperature": None,
            "frequency": None,
            "humidity": None,
            "pressure": None,
            "wavelength": None,
            "other_conditions": {},
            "condition_status": "not_reported",
            "evidence": {
                "block_id": "P_2_1",
                "page": 2,
                "bbox": [9, 10, 11, 12],
                "source_type": "text",
                "source_sentence": DSC_SENTENCE,
            },
        }],
        "properties": [{
            "property_id": "prop001",
            "sample_id": "s001",
            "property_name_raw": "glass transition temperature",
            "property_name_normalized": "glass_transition_temperature",
            "property_code": "P3110",
            "property_category": "thermal_property",
            "molecular_weight_type": None,
            "determination_method_raw": determination_method_raw,
            "observation_group_id": None,
            "value_raw": "-85",
            "value_min": -85,
            "value_max": -85,
            "unit_raw": "°C",
            "unit_normalized": "°C",
            "measurement_condition_id": "mc001",
            "source_type": "text",
            "evidence": [{
                "block_id": "P_2_1",
                "page": 2,
                "bbox": [9, 10, 11, 12],
                "source_type": "text",
                "source_sentence": DSC_SENTENCE,
            }],
        }],
        "unresolved_properties": [],
        "provenance": {
            "provider": "test",
            "model": "fake",
            "models": ["fake"],
            "prompt_id": "polymer.stage4.property",
            "prompt_version": "1.0.0",
            "prompt_sha256": digest,
            "vocabulary_sha256": digest,
            "input_hash": digest,
            "model_config_hash": digest,
            "cache_key": digest,
            "output_schema_version": "property_observation_schema.v1",
            "implementation_version": "1.0.0",
            "context_block_count": 3,
            "context_chars": 300,
            "call_count": 1,
        },
        "warnings": [],
    })


def stage4_with_unresolved_method(
    *,
    entity_id: str = "pe001",
) -> Stage4Document:
    data = stage4_document(
        determination_method_raw=None
    ).model_dump(mode="json")
    data["unresolved_properties"] = [{
        "unresolved_id": "uprop001",
        "entity_id": entity_id,
        "sample_id": None,
        "property_name_raw": "absorption band",
        "property_name_normalized": None,
        "property_code": None,
        "property_category": None,
        "molecular_weight_type": None,
        "determination_method_raw": "FTIR",
        "observation_group_id": None,
        "value_raw": "1650",
        "value_min": None,
        "value_max": None,
        "unit_raw": "cm-1",
        "unit_normalized": None,
        "measurement_condition_id": None,
        "reason": "sample_ambiguous",
        "evidence": [{
            "block_id": "P_2_0",
            "page": 2,
            "bbox": [5, 6, 7, 8],
            "source_type": "text",
            "source_sentence": FTIR_SENTENCE,
        }],
    }]
    return Stage4Document.model_validate(data)


def stage4_with_two_series(*, second_sample_id: str = "s001") -> Stage4Document:
    data = stage4_document().model_dump(mode="json")
    evidence = {
        "block_id": "P_2_1",
        "page": 2,
        "bbox": [9, 10, 11, 12],
        "source_type": "text",
        "source_sentence": DSC_SENTENCE,
    }
    data["property_series"] = [
        {
            "series_id": series_id,
            "sample_id": sample_id,
            "entity_id": "pe001",
            "sample_resolution_status": "resolved",
            "property_name_raw": property_name,
            "property_name_normalized": normalized,
            "property_code": property_code,
            "property_category": "thermal_property",
            "determination_method_raw": "DSC",
            "unit_raw": "°C",
            "unit_normalized": "°C",
            "measurement_context": {"condition_status": "not_reported"},
            "points": [{
                "point_id": point_id,
                "sample_id": sample_id,
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "coordinates": [],
                "value_raw": value,
                "value_min": float(value),
                "value_max": float(value),
                "unit_raw": "°C",
                "unit_normalized": "°C",
                "measurement_context": {"condition_status": "not_reported"},
                "coverage_status": "covered",
                "evidence": [evidence],
                "confidence": {"score": 0.9},
            }],
            "coverage": {
                "expected": 1,
                "covered": 1,
                "missing": 0,
                "not_applicable": 0,
                "ratio": 1.0,
            },
            "evidence": [evidence],
            "confidence": {"score": 0.9},
        }
        for series_id, sample_id, property_name, normalized, property_code,
        point_id, value in (
            (
                "series001", "s001", "glass transition temperature",
                "glass_transition_temperature", "P3110", "pt001", "-85",
            ),
            (
                "series002", second_sample_id, "melting temperature",
                "melting_temperature", "P3120", "pt002", "120",
            ),
        )
    ]
    return Stage4Document.model_validate(data)


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
                "characterizations": [
                    {
                        "characterization_id": "char010",
                        "method_raw": "FTIR",
                        "method_normalized": "FTIR",
                        "sample_id": "s001",
                        "entity_id": "pe001",
                        "sample_resolution_status": "resolved",
                        "instrument": None,
                        "parameters": {
                            "wavenumber_range": "4000-400 cm-1",
                        },
                        "result_summary": (
                            "An absorption band at 1650 cm-1 was assigned "
                            "to C=C stretching."
                        ),
                        "derived_property_ids": ["prop_s5_010"],
                        "evidence": [
                            {
                                "block_id": "P_1_0",
                                "source_sentence": METHOD_SENTENCE,
                                "table_locator": None,
                            },
                            {
                                "block_id": "P_2_0",
                                "source_sentence": FTIR_SENTENCE,
                                "table_locator": None,
                            },
                        ],
                    },
                    {
                        "characterization_id": "char020",
                        "method_raw": "DSC",
                        "method_normalized": "DSC",
                        "sample_id": "s001",
                        "entity_id": "pe001",
                        "sample_resolution_status": "resolved",
                        "instrument": None,
                        "parameters": {},
                        "result_summary": (
                            "The glass transition temperature was -85 °C."
                        ),
                        "derived_property_ids": ["prop001"],
                        "evidence": [{
                            "block_id": "P_2_1",
                            "source_sentence": DSC_SENTENCE,
                            "table_locator": None,
                        }],
                    },
                ],
                "properties": [{
                    "property_id": "prop_s5_010",
                    "characterization_id": "char010",
                    "sample_id": "s001",
                    "entity_id": "pe001",
                    "sample_resolution_status": "resolved",
                    "property_name_raw": "absorption band",
                    "property_name_normalized": "ftir_peak_wavenumber",
                    "property_category": "composition_structure",
                    "value_raw": "1650",
                    "value_min": 1650,
                    "value_max": 1650,
                    "unit_raw": "cm-1",
                    "unit_normalized": "cm⁻¹",
                    "spectral_assignment": "C=C stretching",
                    "solvent": None,
                    "source_stage": "stage5",
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": FTIR_SENTENCE,
                        "table_locator": None,
                    }],
                }],
            }),
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
        response = super().call_json(
            system_prompt,
            user_message,
            max_tokens=max_tokens,
        )
        if self.calls == 1:
            response.data["characterizations"][1][
                "derived_property_ids"
            ] = ["prop999"]
        return response


class InvalidMethodClient(FakeClient):
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
        response.data["properties"][0][
            "property_name_normalized"
        ] = "xrd_diffraction_peak_2theta"
        response.data["properties"][0]["property_category"] = "morphology"
        return response


class MissingRequiredFieldClient(FakeClient):
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
        del response.data["characterizations"][0]["method_raw"]
        return response


class UnknownEvidenceBlockClient(FakeClient):
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
        response.data["characterizations"][0]["evidence"][0][
            "block_id"
        ] = "P_missing"
        return response


class MissingStage4LinkClient(FakeClient):
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
        response.data["characterizations"][1]["derived_property_ids"] = []
        return response


class UnknownSeriesClient(FakeClient):
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
        response.data["characterizations"][0]["series_id"] = "series999"
        return response


class MultiSeriesClient(FakeClient):
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
        characterization = response.data["characterizations"][1]
        characterization["series_id"] = "series001"
        characterization["derived_property_ids"] = [
            "prop001",
            "series001",
            "series002",
        ]
        return response


class SameMethodTwicePayloadClient(FakeClient):
    """同一方法、同一 sample/entity，但证据块与派生 property 不同。

    对应 reference_no_0071569：薄膜 IR 与 KBr 压片 IR 是两次独立测量。
    """

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
        second = response.data["characterizations"][1]
        second["method_raw"] = "FTIR"
        second["method_normalized"] = "FTIR"
        second["result_summary"] = (
            "Bands at 1760 cm-1 and 840 cm-1 were observed."
        )
        second["evidence"] = [{
            "block_id": "P_2_2",
            "source_sentence": SECOND_FTIR_SENTENCE,
            "table_locator": None,
        }]
        # 原 fixture 里这条挂着 DSC 测定的 prop001；改成 FTIR 后不再回链它。
        second["derived_property_ids"] = []
        return response


class TrueDuplicatePayloadClient(FakeClient):
    """完全相同的两条 Characterization：方法、归属、证据、派生 property 均一致。"""

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
        characterizations = response.data["characterizations"]
        clone = json.loads(json.dumps(characterizations[0]))
        clone["characterization_id"] = "char011"
        characterizations[1] = clone
        for item in response.data["properties"]:
            item["characterization_id"] = "char010"
        return response


class UnresolvedStage4LinkClient(FakeClient):
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
        response.data["characterizations"][0][
            "derived_property_ids"
        ].append("uprop001")
        return response


class UnresolvedClient(FakeClient):
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
                "characterizations": [{
                    "characterization_id": "char010",
                    "method_raw": "FTIR",
                    "method_normalized": "FTIR",
                    "sample_id": None,
                    "entity_id": "pe001",
                    "sample_resolution_status": "unresolved",
                    "instrument": None,
                    "parameters": {},
                    "result_summary": None,
                    "derived_property_ids": [],
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": FTIR_SENTENCE,
                        "table_locator": None,
                    }],
                }],
                "properties": [],
            }),
            provider="test",
            model="fake-actual",
        )


def rendered_prompt():
    return PromptLoader().render_stage_prompt(
        "polymer.stage5.characterization",
        CharacterizationStageResponse,
        expected_stage="stage5_characterization",
        expected_output_schema="characterization_schema.v4",
    )


class Stage5Tests(unittest.TestCase):
    def test_same_value_in_different_table_rows_is_not_duplicate(self) -> None:
        base = add_model_confidence({
            "property_id": "prop_s5_001",
            "characterization_id": "char001",
            "sample_ids": ["s001", "s002"],
            "entity_ids": ["pe001", "pe002"],
            "sample_resolution_status": "multi_resolved",
            "property_name_raw": "IR peak",
            "property_name_normalized": "ftir_peak_wavenumber",
            "property_category": "composition_structure",
            "value_raw": "1589",
            "confidence": {"score": 0.9},
            "evidence": [{
                "block_id": "T_1_0",
                "source_sentence": "Table 1",
                "table_locator": {
                    "table_id": "T_1_0",
                    "row_label": "polymer A",
                    "column_label": "IR peak",
                    "cell_value": "1589",
                },
            }],
        })
        other = dict(base)
        other["property_id"] = "prop_s5_002"
        other["evidence"] = [dict(base["evidence"][0])]
        other["evidence"][0]["table_locator"] = dict(
            base["evidence"][0]["table_locator"]
        )
        other["evidence"][0]["table_locator"]["row_label"] = "polymer B"

        first = Stage5PropertyCandidate.model_validate(base)
        second = Stage5PropertyCandidate.model_validate(other)

        self.assertNotEqual(
            _stage5_property_dedupe_key(first),
            _stage5_property_dedupe_key(second),
        )

    def test_preview_removes_unscoped_characterization_and_property(self) -> None:
        payload = {
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "NMR",
                "method_normalized": "NMR",
                "sample_id": None,
                "entity_id": None,
                "sample_ids": None,
                "entity_ids": None,
                "sample_resolution_status": "unresolved",
                "derived_property_ids": ["prop_s5_001"],
                "evidence": [{
                    "block_id": "P_1_0",
                    "source_sentence": "NMR",
                }],
                "confidence": {"score": 0.5},
            }],
            "properties": [{
                "property_id": "prop_s5_001",
                "characterization_id": "char001",
            }],
        }

        strict, _ = _repair_candidate_response_payload(payload)
        self.assertEqual(len(strict["characterizations"]), 1)

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            preview_relaxed=True,
        )

        self.assertEqual(repaired["characterizations"], [])
        self.assertEqual(repaired["properties"], [])
        self.assertTrue(any(
            warning["code"] == "preview_unscoped_stage5_items_removed"
            for warning in warnings
        ))

    def test_symbolic_method_alias_requires_full_match(self) -> None:
        methods = {"viscometry": ("[\\eta]",)}

        self.assertEqual(
            _method_names_for_raw("[\\eta]", methods),
            {"viscometry"},
        )
        self.assertEqual(
            _method_names_for_raw("$\\eta_{inh}$ (DMSO)", methods),
            set(),
        )

    def test_ftir_hyphenated_alias_matches(self) -> None:
        self.assertEqual(
            _method_names_for_raw("FT-IR", {"FTIR": ("FTIR",)}),
            {"FTIR"},
        )

    def test_preview_synthesizes_characterization_from_unique_series_method(
        self,
    ) -> None:
        stage4_data = stage4_with_two_series().model_dump(mode="json")
        stage4_data["property_series"] = [stage4_data["property_series"][0]]
        stage4_data["property_series"][0][
            "determination_method_raw"
        ] = "inherent viscosities"
        stage4 = Stage4Document.model_validate(stage4_data)

        repaired, warnings = _repair_candidate_response_payload(
            {"characterizations": [], "properties": []},
            stage4=stage4,
            methods={"viscometry": ("inherent viscosities",)},
            preview_relaxed=True,
        )

        characterization = repaired["characterizations"][0]
        self.assertEqual(characterization["method_normalized"], "viscometry")
        self.assertEqual(characterization["series_id"], "series001")
        self.assertEqual(characterization["confidence"]["score"], 0.5)
        self.assertEqual(
            warnings[0]["code"],
            "preview_characterization_synthesized_from_series",
        )

    def test_preview_removes_only_additional_table_evidence_without_locator(
        self,
    ) -> None:
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"].append({
            "block_id": "T_2_0",
            "type": "table",
            "section": "Results",
            "caption": "Table 1. FTIR bands",
            "table_body": "<table><tr><td>1650</td></tr></table>",
            "page": 2,
            "bbox": [17, 18, 19, 20],
            "source_block_index": 4,
        })
        document = Stage0Document.model_validate(document_data)
        payload = {
            "characterizations": [{
                "characterization_id": "char001",
                "derived_property_ids": [],
                "evidence": [
                    {
                        "block_id": "P_2_0",
                        "source_sentence": FTIR_SENTENCE,
                    },
                    {
                        "block_id": "T_2_0",
                        "source_sentence": "Table 1. FTIR bands",
                        "table_locator": None,
                    },
                ],
            }],
            "properties": [],
        }

        strict, _ = _repair_candidate_response_payload(
            payload,
            blocks=document.elements,
        )
        self.assertEqual(len(strict["characterizations"][0]["evidence"]), 2)

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            blocks=document.elements,
            preview_relaxed=True,
        )

        self.assertEqual(
            [
                item["block_id"]
                for item in repaired["characterizations"][0]["evidence"]
            ],
            ["P_2_0"],
        )
        self.assertEqual(
            warnings[0]["code"],
            "preview_incomplete_table_evidence_removed",
        )

    def test_preview_recovers_method_surface_from_evidence(self) -> None:
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "FT-IR",
                "method_normalized": "FTIR",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": "IR spectra were recorded.",
                }],
            }],
            "properties": [],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            methods={"FTIR": ("FTIR", "IR spectra")},
            preview_relaxed=True,
        )

        self.assertEqual(
            repaired["characterizations"][0]["method_raw"],
            "IR spectra",
        )
        self.assertIn(
            "preview_characterization_method_surface_recovered",
            [warning["code"] for warning in warnings],
        )

    def test_preview_removes_unlocatable_additional_text_evidence(self) -> None:
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "FTIR",
                "method_normalized": "FTIR",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "derived_property_ids": [],
                "evidence": [
                    {
                        "block_id": "P_2_0",
                        "source_sentence": FTIR_SENTENCE,
                    },
                    {
                        "block_id": "P_2_1",
                        "source_sentence": "invented sentence",
                    },
                ],
            }],
            "properties": [],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            blocks=stage0_document().elements,
            preview_relaxed=True,
        )

        self.assertEqual(
            [
                item["block_id"]
                for item in repaired["characterizations"][0]["evidence"]
            ],
            ["P_2_0"],
        )
        self.assertIn(
            "preview_unlocatable_additional_evidence_removed",
            [warning["code"] for warning in warnings],
        )

    def test_preview_normalizes_singleton_property_subject_scope(self) -> None:
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "X-ray diffraction",
                "method_normalized": "XRD",
                "sample_ids": ["s005", "s006"],
                "entity_ids": ["pe005", "pe006"],
                "sample_resolution_status": "multi_resolved",
                "derived_property_ids": ["prop_s5_001"],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": "The polymers were examined by XRD.",
                }],
            }],
            "properties": [{
                "property_id": "prop_s5_001",
                "characterization_id": "char001",
                "sample_id": None,
                "sample_ids": ["s006"],
                "entity_id": None,
                "sample_resolution_status": "resolved",
                "property_name_raw": "amorphous",
                "property_name_normalized": "crystallinity",
                "property_category": "composition_structure",
                "value_raw": "amorphous",
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": "The polymer was amorphous.",
                }],
            }],
        })

        strict, _ = _repair_candidate_response_payload(payload)
        self.assertEqual(strict["properties"][0]["sample_ids"], ["s006"])
        with self.assertRaises(ValidationError):
            CharacterizationStageResponse.model_validate(strict)

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            preview_relaxed=True,
        )

        item = repaired["properties"][0]
        self.assertEqual(item["sample_id"], "s006")
        self.assertIsNone(item["sample_ids"])
        CharacterizationStageResponse.model_validate(repaired)
        self.assertIn(
            "preview_singleton_subject_scope_normalized",
            [warning["code"] for warning in warnings],
        )

    def test_preview_normalizes_conflicting_multi_subject_fields(self) -> None:
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "FTIR",
                "method_normalized": "FTIR",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_ids": ["s001", "s002"],
                "entity_ids": ["pe001", "pe002"],
                "sample_resolution_status": "multi_resolved",
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": FTIR_SENTENCE,
                }],
            }],
            "properties": [],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            preview_relaxed=True,
        )

        item = repaired["characterizations"][0]
        self.assertEqual(item["sample_resolution_status"], "resolved")
        self.assertEqual(item["sample_id"], "s001")
        self.assertIsNone(item["sample_ids"])
        self.assertIsNone(item["entity_ids"])
        CharacterizationStageResponse.model_validate(repaired)
        self.assertIn(
            "preview_subject_scope_normalized",
            [warning["code"] for warning in warnings],
        )

    def test_preview_corrects_plural_scope_status(self) -> None:
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "FTIR",
                "method_normalized": "FTIR",
                "entity_ids": ["pe001", "pe002"],
                "sample_resolution_status": "unresolved",
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": FTIR_SENTENCE,
                }],
            }],
            "properties": [],
        })

        repaired, _ = _repair_candidate_response_payload(
            payload,
            preview_relaxed=True,
        )

        self.assertEqual(
            repaired["characterizations"][0]["sample_resolution_status"],
            "multi_resolved",
        )
        CharacterizationStageResponse.model_validate(repaired)

    def test_preview_removes_unsupported_value_and_clears_unit(self) -> None:
        document = stage0_document()
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "FTIR",
                "method_normalized": "FTIR",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "derived_property_ids": ["prop_s5_001", "prop_s5_002"],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": FTIR_SENTENCE,
                }],
            }],
            "properties": [
                {
                    "property_id": "prop_s5_001",
                    "characterization_id": "char001",
                    "sample_id": "s001",
                    "entity_id": "pe001",
                    "sample_resolution_status": "resolved",
                    "property_name_raw": "band",
                    "property_name_normalized": "chemical_structure",
                    "property_category": "composition_structure",
                    "value_raw": "not in source",
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": FTIR_SENTENCE,
                    }],
                },
                {
                    "property_id": "prop_s5_002",
                    "characterization_id": "char001",
                    "sample_id": "s001",
                    "entity_id": "pe001",
                    "sample_resolution_status": "resolved",
                    "property_name_raw": "1650",
                    "property_name_normalized": "chemical_structure",
                    "property_category": "composition_structure",
                    "value_raw": "1650",
                    "unit_raw": "invented-unit",
                    "unit_normalized": "invented-unit",
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": FTIR_SENTENCE,
                    }],
                },
            ],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            blocks=document.elements,
            preview_relaxed=True,
        )

        self.assertEqual(
            [item["property_id"] for item in repaired["properties"]],
            ["prop_s5_002"],
        )
        self.assertIsNone(repaired["properties"][0]["unit_raw"])
        self.assertIsNone(repaired["properties"][0]["unit_normalized"])
        self.assertEqual(
            repaired["characterizations"][0]["derived_property_ids"],
            ["prop_s5_002"],
        )
        codes = [warning["code"] for warning in warnings]
        self.assertIn("preview_unsupported_stage5_property_removed", codes)
        self.assertIn(
            "preview_unsupported_stage5_optional_fields_cleared",
            codes,
        )

    def test_preview_clears_unsupported_characterization_details(self) -> None:
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "FTIR",
                "method_normalized": "FTIR",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "instrument": "invented instrument",
                "parameters": {"mode": "invented mode"},
                "measurement_context": {
                    "condition_status": "reported",
                    "other_conditions": {"curve": "second heating curve"},
                    "other_condition_evidence": {},
                },
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": FTIR_SENTENCE,
                }],
            }],
            "properties": [],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            blocks=stage0_document().elements,
            preview_relaxed=True,
        )

        item = repaired["characterizations"][0]
        self.assertIsNone(item["instrument"])
        self.assertEqual(item["parameters"], {})
        self.assertIsNone(item["measurement_context"])
        self.assertIn(
            "preview_unsupported_characterization_details_cleared",
            [warning["code"] for warning in warnings],
        )

    def test_preview_aligns_conflicting_property_scope_to_owner(self) -> None:
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "FTIR",
                "method_normalized": "FTIR",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "derived_property_ids": ["prop_s5_001"],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": FTIR_SENTENCE,
                }],
            }],
            "properties": [{
                "property_id": "prop_s5_001",
                "characterization_id": "char001",
                "sample_id": "s002",
                "entity_id": "pe002",
                "sample_resolution_status": "resolved",
                "property_name_raw": "1650",
                "property_name_normalized": "chemical_structure",
                "property_category": "composition_structure",
                "value_raw": "1650",
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": FTIR_SENTENCE,
                }],
            }],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            preview_relaxed=True,
        )

        item = repaired["properties"][0]
        self.assertEqual(item["sample_id"], "s001")
        self.assertEqual(item["entity_id"], "pe001")
        self.assertIn(
            "preview_property_subject_scope_aligned",
            [warning["code"] for warning in warnings],
        )

    def test_unique_method_block_is_added_to_characterization_evidence(
        self,
    ) -> None:
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "glass transition temperature",
                "method_normalized": "DSC",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_1_0",
                    "source_sentence": METHOD_SENTENCE,
                }],
            }],
            "properties": [],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            blocks=stage0_document().elements,
        )

        self.assertEqual(
            repaired["characterizations"][0]["evidence"][-1]["block_id"],
            "P_2_1",
        )
        self.assertIn(
            "characterization_method_evidence_supplemented",
            [warning["code"] for warning in warnings],
        )

    def test_ambiguous_method_block_is_not_added(self) -> None:
        document_data = stage0_document().model_dump(mode="json")
        elements = document_data["elements"]
        # 复制承载 DSC 语句的块，制造"两个块都能匹配"的歧义。
        # 不能取 elements[-1]——那是第二条 FTIR 语句。
        source = next(
            item for item in elements if item["text"] == DSC_SENTENCE
        )
        duplicate = dict(source)
        duplicate["block_id"] = "P_2_9"
        duplicate["source_block_index"] = len(elements)
        elements.append(duplicate)
        document = Stage0Document.model_validate(document_data)
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "glass transition temperature",
                "method_normalized": "DSC",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_1_0",
                    "source_sentence": METHOD_SENTENCE,
                }],
            }],
            "properties": [],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            blocks=document.elements,
        )

        self.assertEqual(
            [item["block_id"] for item in repaired["characterizations"][0]["evidence"]],
            ["P_1_0"],
        )
        self.assertNotIn(
            "characterization_method_evidence_supplemented",
            [warning["code"] for warning in warnings],
        )

    def test_cross_subject_series_scope_is_inherited(self) -> None:
        stage4_data = stage4_with_two_series(
            second_sample_id="s002"
        ).model_dump(mode="json")
        first, second = stage4_data["property_series"]
        first["sample_id"] = None
        first["entity_id"] = None
        first["sample_resolution_status"] = "unresolved"
        second_point = second["points"][0]
        second_point["entity_id"] = "pe002"
        first["points"].append(second_point)
        first["coverage"] = {
            "expected": 2,
            "covered": 2,
            "missing": 0,
            "not_applicable": 0,
            "ratio": 1.0,
        }
        stage4_data["property_series"] = [first]
        stage4 = Stage4Document.model_validate(stage4_data)
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "DSC",
                "method_normalized": "DSC",
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
                "series_id": "series001",
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_1",
                    "source_sentence": DSC_SENTENCE,
                }],
            }],
            "properties": [],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            stage4=stage4,
        )

        item = repaired["characterizations"][0]
        self.assertEqual(item["sample_resolution_status"], "multi_resolved")
        self.assertEqual(item["sample_ids"], ["s001", "s002"])
        self.assertEqual(item["entity_ids"], ["pe001", "pe002"])
        CharacterizationStageResponse.model_validate(repaired)
        self.assertIn(
            "series_point_subject_scope_inherited",
            [warning["code"] for warning in warnings],
        )

    def test_single_subject_series_does_not_expand_scope(self) -> None:
        stage4 = stage4_with_two_series()
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "DSC",
                "method_normalized": "DSC",
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
                "series_id": "series001",
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_1",
                    "source_sentence": DSC_SENTENCE,
                }],
            }],
            "properties": [],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            stage4=stage4,
        )

        self.assertNotIn(
            "series_point_subject_scope_inherited",
            [warning["code"] for warning in warnings],
        )
        with self.assertRaises(ValidationError):
            CharacterizationStageResponse.model_validate(repaired)

    def test_nested_multi_series_scopes_inherit_unique_superset(self) -> None:
        stage4_data = stage4_with_two_series(
            second_sample_id="s002"
        ).model_dump(mode="json")
        first, second = stage4_data["property_series"]
        for series in (first, second):
            series["sample_id"] = None
            series["entity_id"] = None
            series["sample_resolution_status"] = "unresolved"
            series["points"][0]["entity_id"] = (
                "pe002"
                if series["points"][0]["sample_id"] == "s002"
                else "pe001"
            )
            extra = dict(series["points"][0])
            extra["point_id"] = f"pt{100 + len(series['points']):03d}"
            extra["sample_id"] = "s002" if extra["sample_id"] == "s001" else "s001"
            extra["entity_id"] = "pe002" if extra["sample_id"] == "s002" else "pe001"
            series["points"].append(extra)
            series["coverage"] = {
                "expected": 2,
                "covered": 2,
                "missing": 0,
                "not_applicable": 0,
                "ratio": 1.0,
            }
        third = dict(first["points"][0])
        third["point_id"] = "pt200"
        third["sample_id"] = "s003"
        third["entity_id"] = "pe003"
        first["points"].append(third)
        first["coverage"] = {
            "expected": 3,
            "covered": 3,
            "missing": 0,
            "not_applicable": 0,
            "ratio": 1.0,
        }
        stage4 = Stage4Document.model_validate(stage4_data)
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "DSC",
                "method_normalized": "DSC",
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
                "series_ids": ["series001", "series002"],
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_1",
                    "source_sentence": DSC_SENTENCE,
                }],
            }],
            "properties": [],
        })

        repaired, _ = _repair_candidate_response_payload(
            payload,
            stage4=stage4,
            preview_relaxed=True,
        )

        item = repaired["characterizations"][0]
        self.assertEqual(item["sample_resolution_status"], "multi_resolved")
        self.assertEqual(set(item["sample_ids"]), {"s001", "s002", "s003"})
        self.assertEqual(set(item["entity_ids"]), {"pe001", "pe002", "pe003"})
        CharacterizationStageResponse.model_validate(repaired)

    def test_different_multi_series_scopes_are_not_inherited(self) -> None:
        stage4_data = stage4_with_two_series(
            second_sample_id="s002"
        ).model_dump(mode="json")
        for index, series in enumerate(stage4_data["property_series"]):
            series["sample_id"] = None
            series["entity_id"] = None
            series["sample_resolution_status"] = "unresolved"
            extra = dict(series["points"][0])
            extra["point_id"] = f"pt{100 + index:03d}"
            extra["sample_id"] = "s002" if index == 0 else "s003"
            extra["entity_id"] = "pe002" if index == 0 else "pe003"
            series["points"].append(extra)
            series["coverage"] = {
                "expected": 2,
                "covered": 2,
                "missing": 0,
                "not_applicable": 0,
                "ratio": 1.0,
            }
        stage4 = Stage4Document.model_validate(stage4_data)
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "DSC",
                "method_normalized": "DSC",
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
                "series_ids": ["series001", "series002"],
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_1",
                    "source_sentence": DSC_SENTENCE,
                }],
            }],
            "properties": [],
        })

        repaired, _ = _repair_candidate_response_payload(
            payload,
            stage4=stage4,
        )

        with self.assertRaises(ValidationError):
            CharacterizationStageResponse.model_validate(repaired)

    def test_explicit_all_synthesized_polymers_expands_multi_subject_scope(
        self,
    ) -> None:
        process_data = stage3_document().model_dump(mode="json")
        process_data["samples"] = [
            {
                **process_data["samples"][0],
                "sample_id": sample_id,
                "sample_kind": "synthesis_batch",
                "refers_to_entity": entity_id,
                "polymer_name": label,
                "sample_label_raw": label,
            }
            for sample_id, entity_id, label in (
                ("s001", "pe001", "PU1"),
                ("s002", "pe002", "PU2"),
            )
        ]
        process = Stage3Document.model_validate(process_data)
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "X-ray diffraction",
                "method_normalized": "XRD",
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
                "derived_property_ids": ["prop_s5_001"],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": (
                        "All synthesized polymers were examined by "
                        "X-ray diffraction."
                    ),
                }],
            }],
            "properties": [{
                "property_id": "prop_s5_001",
                "characterization_id": "char001",
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
                "property_name_raw": "amorphous",
                "property_name_normalized": "crystallinity",
                "property_category": "composition_structure",
                "value_raw": "amorphous",
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": (
                        "All synthesized polymers were amorphous."
                    ),
                }],
            }],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            process=process,
        )

        characterization = repaired["characterizations"][0]
        property_item = repaired["properties"][0]
        self.assertEqual(
            characterization["sample_resolution_status"],
            "multi_resolved",
        )
        self.assertEqual(characterization["sample_ids"], ["s001", "s002"])
        self.assertEqual(characterization["entity_ids"], ["pe001", "pe002"])
        self.assertEqual(property_item["sample_ids"], ["s001", "s002"])
        CharacterizationStageResponse.model_validate(repaired)
        self.assertIn(
            "explicit_all_samples_scope_expanded",
            [warning["code"] for warning in warnings],
        )

    def test_generic_polymers_does_not_expand_subject_scope(self) -> None:
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "X-ray diffraction",
                "method_normalized": "XRD",
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": "Polymers were examined by XRD.",
                }],
            }],
            "properties": [],
        })

        repaired, warnings = _repair_candidate_response_payload(
            payload,
            process=stage3_document(),
        )

        self.assertEqual(warnings, [])
        with self.assertRaises(ValidationError):
            CharacterizationStageResponse.model_validate(repaired)

    def test_explicit_all_scope_is_inherited_only_within_same_block(self) -> None:
        process_data = stage3_document().model_dump(mode="json")
        process_data["samples"] = [
            {
                **process_data["samples"][0],
                "sample_id": sample_id,
                "sample_kind": "synthesis_batch",
                "refers_to_entity": entity_id,
                "polymer_name": label,
                "sample_label_raw": label,
            }
            for sample_id, entity_id, label in (
                ("s001", "pe001", "PU1"),
                ("s002", "pe002", "PU2"),
            )
        ]
        process = Stage3Document.model_validate(process_data)
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"][1]["text"] = (
            "All synthesized polymers were amorphous. "
            "DSC thermograms of polymers showed glass transitions."
        )
        document = Stage0Document.model_validate(document_data)
        payload = add_model_confidence({
            "characterizations": [{
                "characterization_id": "char001",
                "method_raw": "DSC",
                "method_normalized": "DSC",
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
                "derived_property_ids": [],
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": (
                        "DSC thermograms of polymers showed glass transitions."
                    ),
                }],
            }],
            "properties": [],
        })

        repaired, _ = _repair_candidate_response_payload(
            payload,
            process=process,
            blocks=document.elements,
        )

        self.assertEqual(
            repaired["characterizations"][0]["sample_ids"],
            ["s001", "s002"],
        )

    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.methods,
            cls.vocabulary,
            cls.vocabulary_hash,
        ) = load_characterization_vocabulary(DEFAULT_VOCABULARY_PATH)

    def test_vocabulary_separates_stage4_measurements(self) -> None:
        self.assertIn("FTIR", self.methods)
        self.assertIn("DSC", self.methods)
        self.assertEqual(
            self.vocabulary["ftir_peak_wavenumber"],
            ("composition_structure", frozenset({"FTIR"})),
        )
        self.assertNotIn(
            "glass_transition_temperature",
            self.vocabulary,
        )
        self.assertIn("swelling", self.methods)
        self.assertIn(r"[\eta]", self.methods["viscometry"])
        self.assertIn("[Q]", self.methods["swelling"])
        self.assertIn(r"[\Phi]", self.methods["turbidimetry"])
        self.assertIn("dilatometry", self.methods)
        self.assertEqual(
            _method_names_for_raw("measured dilatometrically", self.methods),
            {"dilatometry"},
        )
        self.assertEqual(
            _method_names_for_raw("dila-tometric", self.methods),
            {"dilatometry"},
        )

    def test_prompt_contains_stage4_series_section(self) -> None:
        blocks, _, _ = stage5_module.select_context_blocks(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4_document(),
        )
        message = stage5_module._user_message(
            "reference_no_0000002",
            stage2_document(),
            stage3_document(),
            stage4_document(),
            blocks,
            self.methods,
            self.vocabulary,
        )

        self.assertIn("BEGIN EXISTING STAGE 4 SERIES", message)

    def test_unknown_series_reference_is_rejected(self) -> None:
        with self.assertRaises(Stage5Error):
            extract_characterizations(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                stage4_document(),
                UnknownSeriesClient(),
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_duplicate_series_derived_reference_is_removed(self) -> None:
        original = {
            "characterizations": [{
                "characterization_id": "char001",
                "series_id": "series001",
                "derived_property_ids": ["series001", "prop001"],
            }],
        }

        repaired, warnings = _repair_candidate_response_payload(original)

        self.assertEqual(
            repaired["characterizations"][0]["derived_property_ids"],
            ["prop001"],
        )
        self.assertEqual(
            original["characterizations"][0]["derived_property_ids"],
            ["series001", "prop001"],
        )
        self.assertEqual(
            warnings[0]["code"],
            "duplicate_series_derived_reference_removed",
        )

    def test_other_unknown_derived_reference_is_not_removed(self) -> None:
        repaired, warnings = _repair_candidate_response_payload({
            "characterizations": [{
                "characterization_id": "char001",
                "series_id": "series001",
                "derived_property_ids": ["series999"],
            }],
        })

        self.assertEqual(
            repaired["characterizations"][0]["derived_property_ids"],
            ["series999"],
        )
        self.assertEqual(warnings, [])

    def test_property_completes_missing_characterization_back_reference(
        self,
    ) -> None:
        repaired, warnings = _repair_candidate_response_payload({
            "characterizations": [{
                "characterization_id": "char001",
                "derived_property_ids": [],
            }],
            "properties": [{
                "property_id": "prop_s5_001",
                "characterization_id": "char001",
            }],
        })

        self.assertEqual(
            repaired["characterizations"][0]["derived_property_ids"],
            ["prop_s5_001"],
        )
        self.assertEqual(
            warnings[0]["code"],
            "derived_property_back_reference_completed",
        )

    def test_conflicting_property_owner_is_not_repaired(self) -> None:
        payload = {
            "characterizations": [
                {
                    "characterization_id": "char001",
                    "derived_property_ids": [],
                },
                {
                    "characterization_id": "char002",
                    "derived_property_ids": ["prop_s5_001"],
                },
            ],
            "properties": [{
                "property_id": "prop_s5_001",
                "characterization_id": "char001",
            }],
        }
        repaired, warnings = _repair_candidate_response_payload(payload)

        self.assertEqual(
            repaired["characterizations"][0]["derived_property_ids"], []
        )
        self.assertEqual(warnings, [])

        preview_repaired, preview_warnings = (
            _repair_candidate_response_payload(
                payload,
                preview_relaxed=True,
            )
        )
        self.assertEqual(
            preview_repaired["characterizations"][0]["derived_property_ids"],
            ["prop_s5_001"],
        )
        self.assertEqual(
            preview_repaired["characterizations"][1]["derived_property_ids"],
            [],
        )
        self.assertEqual(
            preview_warnings[0]["code"],
            "preview_property_back_reference_reassigned",
        )

    def test_known_series_references_move_to_series_ids(self) -> None:
        stage4 = stage4_with_two_series()
        response = MultiSeriesClient().call_json("", "")

        repaired, warnings = _repair_candidate_response_payload(
            response.data,
            stage4,
        )

        characterization = repaired["characterizations"][1]
        self.assertIsNone(characterization["series_id"])
        self.assertEqual(
            characterization["series_ids"],
            ["series001", "series002"],
        )
        self.assertEqual(characterization["derived_property_ids"], ["prop001"])
        self.assertEqual(
            warnings[0]["code"],
            "series_references_moved_from_derived_properties",
        )

    def test_multiple_series_are_validated_and_materialized(self) -> None:
        result = extract_characterizations(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4_with_two_series(),
            MultiSeriesClient(),
            rendered_prompt(),
            self.methods,
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        characterization = result.characterizations[1]
        self.assertIsNone(characterization.series_id)
        self.assertEqual(
            characterization.series_ids,
            ["series001", "series002"],
        )
        self.assertEqual(characterization.derived_property_ids, ["prop001"])

    def test_same_method_on_distinct_evidence_is_not_duplicate(self) -> None:
        """同一方法在不同位置的两次表征不是重复。

        真实案例 reference_no_0071569：薄膜 IR（P_8_66，N—H 3420/C=O 1685）
        与 KBr 压片 IR（P_8_72，1760/1168/840）是两次独立测量，
        旧去重键只看 (方法, sample, entity)，会把它们判成重复而整篇硬失败。
        """
        result = extract_characterizations(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4_document(determination_method_raw=None),
            SameMethodTwicePayloadClient(),
            rendered_prompt(),
            self.methods,
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        methods = [
            item.method_normalized for item in result.characterizations
        ]
        self.assertEqual(methods, ["FTIR", "FTIR"])
        self.assertEqual(
            [
                sorted(evidence.block_id for evidence in item.evidence)
                for item in result.characterizations
            ],
            [["P_1_0", "P_2_0"], ["P_2_2"]],
        )

    def test_identical_characterizations_still_fail(self) -> None:
        """放宽去重键之后，真正的重复输出仍须硬失败。"""
        with self.assertRaises(Stage5Error) as ctx:
            extract_characterizations(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                stage4_document(),
                TrueDuplicatePayloadClient(),
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

        self.assertIn("重复 Characterization", str(ctx.exception))

    def test_multiple_series_cannot_cross_samples(self) -> None:
        with self.assertRaises(Stage5Error):
            extract_characterizations(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                stage4_with_two_series(second_sample_id="s999"),
                MultiSeriesClient(),
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_series_id_and_series_ids_are_mutually_exclusive(self) -> None:
        payload = MultiSeriesClient().call_json("", "").data
        payload["characterizations"][1]["series_ids"] = [
            "series001",
            "series002",
        ]

        with self.assertRaises(ValidationError):
            CharacterizationStageResponse.model_validate(payload)

    def test_extracts_ftir_property_and_links_dsc_stage4_property(self) -> None:
        result = extract_characterizations(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4_document(),
            FakeClient(),
            rendered_prompt(),
            self.methods,
            self.vocabulary,
            self.vocabulary_hash,
        )

        self.assertEqual(
            [item.characterization_id for item in result.characterizations],
            ["char001", "char002"],
        )
        self.assertEqual(result.properties[0].property_id, "prop_s5_001")
        self.assertEqual(
            result.properties[0].characterization_id,
            "char001",
        )
        self.assertEqual(
            result.characterizations[0].derived_property_ids,
            ["prop_s5_001"],
        )
        self.assertEqual(
            result.characterizations[1].derived_property_ids,
            ["prop001"],
        )
        self.assertEqual(
            result.characterizations[0].parameters["wavenumber_range"],
            "4000-400 cm-1",
        )
        self.assertEqual(result.provenance.model, "fake-actual")

    def test_unknown_derived_property_is_retried(self) -> None:
        client = RetryClient()

        result = extract_characterizations(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4_document(),
            client,
            rendered_prompt(),
            self.methods,
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=1,
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual(len(result.characterizations), 2)

    def test_property_method_mismatch_is_rejected(self) -> None:
        with self.assertRaises(Stage5Error):
            extract_characterizations(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                stage4_document(),
                InvalidMethodClient(),
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_preview_preserves_schema_valid_semantic_mismatch(self) -> None:
        result = extract_characterizations(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4_document(),
            InvalidMethodClient(),
            rendered_prompt(),
            self.methods,
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(
            result.properties[0].property_name_normalized,
            "xrd_diffraction_peak_2theta",
        )
        self.assertTrue(any(
            item["code"] == "preview_semantic_validation_bypassed"
            for item in result.warnings
        ))

    def test_preview_uses_degraded_shell_for_invalid_schema(self) -> None:
        result = extract_characterizations(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4_document(),
            MissingRequiredFieldClient(),
            rendered_prompt(),
            self.methods,
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(result.characterizations, [])
        self.assertEqual(result.properties, [])
        warning = next(
            item
            for item in result.warnings
            if item["code"] == "preview_degraded_empty_shell"
        )
        self.assertTrue(warning["degraded"])

    def test_strict_rejects_invalid_schema(self) -> None:
        with self.assertRaises(Stage5Error):
            extract_characterizations(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                stage4_document(),
                MissingRequiredFieldClient(),
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_preview_degrades_when_materialization_is_unsafe(self) -> None:
        result = extract_characterizations(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4_document(),
            UnknownEvidenceBlockClient(),
            rendered_prompt(),
            self.methods,
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(result.characterizations, [])
        self.assertEqual(result.properties, [])
        self.assertTrue(any(
            item["code"] == "preview_degraded_empty_shell"
            and item["degraded"]
            for item in result.warnings
        ))

    def test_stage4_method_requires_characterization_backlink(self) -> None:
        with self.assertRaises(Stage5Error):
            extract_characterizations(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                stage4_document(),
                MissingStage4LinkClient(),
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_unresolved_stage4_method_links_by_entity(self) -> None:
        result = extract_characterizations(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4_with_unresolved_method(),
            UnresolvedStage4LinkClient(),
            rendered_prompt(),
            self.methods,
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertIn(
            "uprop001",
            result.characterizations[0].derived_property_ids,
        )

    def test_unresolved_stage4_method_requires_backlink(self) -> None:
        with self.assertRaises(Stage5Error):
            extract_characterizations(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                stage4_with_unresolved_method(),
                FakeClient(),
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_unresolved_stage4_link_cannot_cross_entity(self) -> None:
        with self.assertRaises(Stage5Error):
            extract_characterizations(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                stage4_with_unresolved_method(entity_id="pe999"),
                UnresolvedStage4LinkClient(),
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_unresolved_sample_generates_warning(self) -> None:
        result = extract_characterizations(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                stage4_document(determination_method_raw=None),
                UnresolvedClient(),
            rendered_prompt(),
            self.methods,
            self.vocabulary,
            self.vocabulary_hash,
        )

        self.assertEqual(
            result.characterizations[0].sample_resolution_status,
            "unresolved",
        )
        self.assertEqual(
            result.warnings[0]["code"],
            "unresolved_characterizations",
        )

    def test_compatible_output_cache_is_reused(self) -> None:
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0.json"
            stage2_path = root / "stage2.json"
            stage3_path = root / "stage3.json"
            stage4_path = root / "stage4.json"
            output_path = root / "stage5.json"
            for path, model in (
                (stage0_path, stage0_document()),
                (stage2_path, stage2_document()),
                (stage3_path, stage3_document()),
                (stage4_path, stage4_document()),
            ):
                path.write_text(
                    json.dumps(model.model_dump(mode="json")),
                    encoding="utf-8",
                )

            _, first_cached = run_stage5(
                stage0_path,
                stage2_path,
                stage3_path,
                stage4_path,
                output_path,
                client,
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
            )
            _, second_cached = run_stage5(
                stage0_path,
                stage2_path,
                stage3_path,
                stage4_path,
                output_path,
                client,
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
            )

        self.assertFalse(first_cached)
        self.assertTrue(second_cached)
        self.assertEqual(client.calls, 1)

    def test_failure_response_can_be_replayed_without_network(self) -> None:
        response = FakeClient().call_json("", "")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_paths = {
                "stage0": root / "stage0.json",
                "stage2": root / "stage2.json",
                "stage3": root / "stage3.json",
                "stage4": root / "stage4.json",
            }
            output_path = root / "stage5.json"
            failure_path = root / "stage5_failure.json"
            for name, model in (
                ("stage0", stage0_document()),
                ("stage2", stage2_document()),
                ("stage3", stage3_document()),
                ("stage4", stage4_document()),
            ):
                input_paths[name].write_text(
                    json.dumps(model.model_dump(mode="json")),
                    encoding="utf-8",
                )
            failure_path.write_text(
                json.dumps({
                    "raw_response": {
                        "provider": "saved-provider",
                        "model": "saved-model",
                        "content": json.dumps(response.data),
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                        "cost": {
                            "currency": "CNY",
                            "input_per_million": "6.75",
                            "output_per_million": "40.48",
                            "input_cost": "0.000675",
                            "output_cost": "0.002024",
                            "total_cost": "0.002699",
                        },
                    }
                }),
                encoding="utf-8",
            )
            client = _failure_replay_client(
                failure_path,
                load_pipeline_config(DEFAULT_CONFIG_PATH),
            )

            run_stage5(
                input_paths["stage0"],
                input_paths["stage2"],
                input_paths["stage3"],
                input_paths["stage4"],
                output_path,
                client,
                rendered_prompt(),
                self.methods,
                self.vocabulary,
                self.vocabulary_hash,
                force=True,
                max_validation_retries=0,
            )

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(client.calls, 1)
            self.assertEqual(result["provenance"]["model"], "saved-model")
            self.assertEqual(result["provenance"]["usage"]["input_tokens"], 100)
            self.assertEqual(result["provenance"]["cost"]["total_cost"], "0.002699")
            self.assertTrue(any(
                item["code"] == "failure_response_replayed"
                for item in result["warnings"]
            ))

    def test_failure_replay_requires_saved_raw_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            failure_path = Path(temp_dir) / "stage5_failure.json"
            failure_path.write_text(
                json.dumps({"raw_response": None}),
                encoding="utf-8",
            )

            with self.assertRaises(Stage5Error):
                _failure_replay_client(
                    failure_path,
                    load_pipeline_config(DEFAULT_CONFIG_PATH),
                )

    def test_cli_failure_writes_raw_response_audit(self) -> None:
        client = FakeClient()
        client.call_history = []
        client.last_raw_response = None
        usage = LLMTokenUsage(input_tokens=11, output_tokens=7)

        def fail_after_response(*args, **kwargs):
            client.call_history.append(LLMCallRecord(
                provider="test",
                model="fake-actual",
                usage=usage,
                cost=None,
                usage_available=True,
            ))
            client.last_raw_response = LLMRawResponse(
                provider="test",
                model="fake-actual",
                finish_reason="stop",
                content='{"characterizations": []}',
                usage=usage,
                cost=None,
            )
            raise Stage5Error("schema failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            argv = [
                "stage5_characterization.py",
                "--ref-no",
                "reference_no_0000002",
                "--input-root",
                str(root),
                "--output-root",
                str(root),
            ]
            with (
                patch.object(
                    stage5_module.LLMClient,
                    "from_pipeline_config",
                    return_value=client,
                ),
                patch.object(
                    stage5_module,
                    "run_stage5",
                    side_effect=fail_after_response,
                ),
                patch.object(sys, "argv", argv),
            ):
                status = stage5_module.main()

            audit_path = (
                root
                / "reference_no_0000002"
                / "stage5_failure.json"
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(status, 1)
            self.assertEqual(audit["usage"]["input_tokens"], 11)
            self.assertEqual(
                audit["raw_response"]["content"],
                '{"characterizations": []}',
            )

    def test_historical_method_aliases_map_uniquely(self) -> None:
        self.assertEqual(
            _method_names_for_raw(
                "scanning electron microscope",
                self.methods,
            ),
            {"SEM"},
        )
        self.assertEqual(
            _method_names_for_raw("Infrared Absorption", self.methods),
            {"FTIR"},
        )
        self.assertEqual(
            _method_names_for_raw("Thermal Degradation", self.methods),
            {"TGA"},
        )
        self.assertEqual(
            _method_names_for_raw(
                "thermogravimetric analyses conducted in nitrogen",
                self.methods,
            ),
            {"TGA"},
        )
        self.assertEqual(
            _method_names_for_raw(
                "Inherent viscosities were obtained from polymer solutions",
                self.methods,
            ),
            {"viscometry"},
        )
        self.assertEqual(
            _method_names_for_raw("x-ray diffractions", self.methods),
            {"XRD"},
        )
        self.assertEqual(
            _method_names_for_raw(
                "The IR spectra of polymers were scanned in KBr pellets",
                self.methods,
            ),
            {"FTIR"},
        )

    def test_short_method_alias_does_not_match_inside_word(self) -> None:
        raw = "polymer was placed on a metal block"

        self.assertNotIn(
            "viscometry",
            _method_names_for_raw(raw, self.methods),
        )

    def test_untraceable_property_name_falls_back_to_raw_value(self) -> None:
        response = FakeClient().call_json("", "").data
        candidate = CharacterizationStageResponse.model_validate(response)
        item = candidate.properties[0].model_copy(update={
            "property_name_raw": "invented peak label",
        })
        document = stage0_document()
        block_map = {block.block_id: block for block in document.elements}
        characterization = candidate.characterizations[0]

        normalized = _normalize_stage5_property(
            item,
            block_map,
            characterization,
            self.vocabulary,
        )

        self.assertEqual(normalized.property_name_raw, item.value_raw)


if __name__ == "__main__":
    unittest.main()
