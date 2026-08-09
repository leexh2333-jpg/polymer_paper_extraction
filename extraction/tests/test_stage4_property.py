import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


from llm_client import (
    DEFAULT_CONFIG_PATH,
    LLMCallCost,
    LLMJSONResponse,
    LLMRawResponse,
    LLMTokenUsage,
    ResolvedLLMConfig,
    load_pipeline_config,
)
from prompt_loader import PromptLoader
from schema.polymer_schema import (
    PropertyEvidenceCandidate,
    PropertyStageResponse,
    Stage0Document,
    Stage2Document,
    Stage3Document,
    UnresolvedPropertyObservation,
    _validate_aggregate_series_reference,
)
from tests.helpers import add_model_confidence
from stages.stage4_property import (
    DEFAULT_VOCABULARY_PATH,
    Stage4Error,
    _cache_components,
    _normalize_condition_field_evidence,
    _normalize_determination_method,
    _normalize_property_name,
    _candidate_repair_warnings,
    _failure_replay_client,
    _looks_like_series_property_header,
    _materialize,
    _preview_salvage_materialization,
    _recover_grouped_table_methods,
    _repair_candidate_response_payload,
    _resolve_surface_text,
    _stage4_raw_response_artifact,
    _validate_required_table_series,
    extract_properties,
    load_property_vocabulary,
    run_stage4,
    select_context_blocks,
)
from stages.table_grid import parse_table_cells


RESULT_SENTENCE = (
    "The solubility parameter of dried PB film was 8.5 to 8.6 "
    "(cal/ml)^1/2 at 25 °C."
)
TABLE_BODY = (
    "| Sample | Property | Value |\n"
    "|---|---|---|\n"
    "| dried PB film | solubility parameter | "
    "8.5 to 8.6 (cal/ml)^1/2 |"
)
MULTI_METHOD_SENTENCE = (
    "The solubility parameter values were 8.55, 8.60, and 8.55 "
    "(cal/ml)^1/2 by viscometry, turbidimetry, and swelling "
    "measurements, respectively."
)


def stage0_document() -> Stage0Document:
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
                "block_id": "P_1_0",
                "type": "text",
                "section": "Methods",
                "text": "Solubility parameters were measured at 25 °C.",
                "page": 1,
                "bbox": [1, 2, 3, 4],
                "source_block_index": 0,
            },
            {
                "block_id": "P_2_0",
                "type": "text",
                "section": "Results",
                "text": RESULT_SENTENCE,
                "page": 2,
                "bbox": [5, 6, 7, 8],
                "source_block_index": 1,
            },
            {
                "block_id": "T_2_0",
                "type": "table",
                "section": "Results",
                "table_body": TABLE_BODY,
                "caption": "Table 1. Solubility parameter.",
                "page": 2,
                "bbox": [9, 10, 11, 12],
                "source_block_index": 2,
            },
            {
                "block_id": "P_3_0",
                "type": "text",
                "section": "Methods",
                "text": "The specimens were stored in the dark.",
                "page": 3,
                "bbox": [13, 14, 15, 16],
                "source_block_index": 3,
            },
        ],
        "warnings": [],
    })


def tg_mn_table_document() -> Stage0Document:
    document = stage0_document().model_dump(mode="json")
    document["elements"].append({
        "block_id": "T_4_0",
        "type": "table",
        "section": "Results",
        "page": 4,
        "bbox": [1, 2, 3, 4],
        "source_block_index": 4,
        "caption": "Glass transition and molecular weight",
        "table_body": (
            "| Sample | Tg/K | Mn |\n"
            "| --- | --- | --- |\n"
            "| A | 301 | 1000 |\n"
            "| B | 302 | 2000 |"
        ),
        "table_cells": [
            {"cell_id": "T_4_0:r0000:c0000", "row_index": 0, "column_index": 0, "row_span": 1, "column_span": 1, "text": "Sample"},
            {"cell_id": "T_4_0:r0000:c0001", "row_index": 0, "column_index": 1, "row_span": 1, "column_span": 1, "text": "Tg/K"},
            {"cell_id": "T_4_0:r0000:c0002", "row_index": 0, "column_index": 2, "row_span": 1, "column_span": 1, "text": "Mn"},
            {"cell_id": "T_4_0:r0001:c0000", "row_index": 1, "column_index": 0, "row_span": 1, "column_span": 1, "text": "A"},
            {"cell_id": "T_4_0:r0001:c0001", "row_index": 1, "column_index": 1, "row_span": 1, "column_span": 1, "text": "301"},
            {"cell_id": "T_4_0:r0001:c0002", "row_index": 1, "column_index": 2, "row_span": 1, "column_span": 1, "text": "1000"},
            {"cell_id": "T_4_0:r0002:c0000", "row_index": 2, "column_index": 0, "row_span": 1, "column_span": 1, "text": "B"},
            {"cell_id": "T_4_0:r0002:c0001", "row_index": 2, "column_index": 1, "row_span": 1, "column_span": 1, "text": "302"},
            {"cell_id": "T_4_0:r0002:c0002", "row_index": 2, "column_index": 2, "row_span": 1, "column_span": 1, "text": "2000"},
        ],
    })
    return Stage0Document.model_validate(document)


def multi_method_stage0_document() -> Stage0Document:
    data = stage0_document().model_dump(mode="json")
    data["elements"][1]["text"] = MULTI_METHOD_SENTENCE
    return Stage0Document.model_validate(data)


def separate_method_stage0_document() -> Stage0Document:
    data = stage0_document().model_dump(mode="json")
    data["elements"][3]["text"] = (
        "The independent viscosity method was used."
    )
    return Stage0Document.model_validate(data)


def method_header_stage0_document() -> Stage0Document:
    data = stage0_document().model_dump(mode="json")
    data["elements"][2]["table_body"] = TABLE_BODY.replace(
        "| Sample | Property | Value |",
        "| Sample | Property | [Q] |",
    )
    return Stage0Document.model_validate(data)


def stage2_document() -> Stage2Document:
    digest = "c" * 64
    return Stage2Document.model_validate({
        "schema_version": "1.0",
        "document_id": "reference_no_0000001",
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
                "source_sentence": RESULT_SENTENCE,
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
    digest = "d" * 64
    return Stage3Document.model_validate({
        "schema_version": "1.0",
        "document_id": "reference_no_0000001",
        "samples": [{
            "sample_id": "s001",
            "sample_kind": "processed_material",
            "refers_to_entity": "pe001",
            "polymer_name": "dried PB film",
            "evidence": {
                "block_id": "P_2_0",
                "page": 2,
                "bbox": [5, 6, 7, 8],
                "source_type": "text",
                "source_sentence": RESULT_SENTENCE,
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
                "measurement_conditions": [{
                    "condition_id": "mc010",
                    "temperature": {
                        "raw": "25 °C",
                        "value": 25,
                        "unit": "°C",
                    },
                    "frequency": None,
                    "humidity": None,
                    "pressure": None,
                    "wavelength": None,
                    "other_conditions": {},
                    "condition_status": "reported",
                    "evidence": {
                        "block_id": "P_2_0",
                        "source_sentence": RESULT_SENTENCE,
                        "table_locator": None,
                    },
                }],
                "properties": [{
                    "property_id": "prop010",
                    "sample_id": "s001",
                    "property_name_raw": "solubility parameter",
                    "property_name_normalized": "solubility_parameter",
                    "property_code": "P5110",
                    "property_category": "physicochemical_property",
                    "value_raw": "8.5 to 8.6",
                    "value_min": 8.5,
                    "value_max": 8.6,
                    "unit_raw": "(cal/ml)^1/2",
                    "unit_normalized": "(cal/mL)^0.5",
                    "measurement_condition_id": "mc010",
                    "evidence": [
                        {
                            "block_id": "P_2_0",
                            "source_sentence": RESULT_SENTENCE,
                            "table_locator": None,
                        },
                        {
                            "block_id": "T_2_0",
                            "source_sentence": (
                                "dried PB film | solubility parameter | "
                                "8.5 to 8.6 (cal/ml)^1/2"
                            ),
                            "table_locator": {
                                "table_id": "T_2_0",
                                "row_label": "dried PB film",
                                "column_label": "Value",
                                "cell_value": "8.5 to 8.6 (cal/ml)^1/2",
                            },
                        },
                    ],
                }],
                "unresolved_properties": [],
            }),
            provider="test",
            model="fake-actual",
        )


class SeriesClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        row = (
            "dried PB film | solubility parameter | "
            "8.5 to 8.6 (cal/ml)^1/2"
        )
        return LLMJSONResponse(
            data=add_model_confidence({
                "measurement_conditions": [],
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "series_id": "series010",
                    "sample_id": "s001",
                    "entity_id": "pe001",
                    "sample_resolution_status": "resolved",
                    "property_name_raw": "solubility parameter",
                    "property_name_normalized": "solubility_parameter",
                    "property_code": "P5110",
                    "property_category": "physicochemical_property",
                    "determination_method_raw": None,
                    "observation_group_id": None,
                    "unit_raw": "(cal/ml)^1/2",
                    "unit_normalized": "(cal/mL)^0.5",
                    "measurement_context": {
                        "condition_status": "not_reported",
                    },
                    "points": [{
                        "point_id": "pt010",
                        "coordinates": [{
                            "name_raw": "Sample",
                            "value_raw": "dried PB film",
                            "unit_raw": None,
                            "evidence": {
                                "block_id": "T_2_0",
                                "source_sentence": row,
                                "table_locator": {
                                    "table_id": "T_2_0",
                                    "row_label": "dried PB film",
                                    "column_label": "Sample",
                                    "cell_value": "dried PB film",
                                },
                            },
                        }],
                        "value_raw": "8.5 to 8.6 (cal/ml)^1/2",
                        "value_min": 8.5,
                        "value_max": 8.6,
                        "unit_raw": "(cal/ml)^1/2",
                        "unit_normalized": "(cal/mL)^0.5",
                        "measurement_context": None,
                        "coverage_status": "covered",
                        "evidence": [{
                            "block_id": "T_2_0",
                            "source_sentence": row,
                            "table_locator": {
                                "table_id": "T_2_0",
                                "row_label": "dried PB film",
                                "column_label": "Value",
                                "cell_value": (
                                    "8.5 to 8.6 (cal/ml)^1/2"
                                ),
                            },
                        }],
                    }],
                    "coverage": None,
                    "evidence": [],
                }],
            }),
            provider="test",
            model="fake-actual",
        )


class CoordinateOnlyMnClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        table = tg_mn_table_document().elements[-1].table_body
        points = []
        for index, (sample, tg, mn) in enumerate(
            (("A", "301", "1000"), ("B", "302", "2000")),
            start=1,
        ):
            points.append({
                "point_id": f"pt{index:03d}",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "coordinates": [{
                    "name_raw": "Mn",
                    "value_raw": mn,
                    "unit_raw": None,
                    "evidence": {
                        "block_id": "T_4_0",
                        "source_sentence": table,
                        "table_locator": {
                            "table_id": "T_4_0",
                            "row_label": sample,
                            "column_label": "Mn",
                            "cell_value": mn,
                            "cell_id": f"T_4_0:r{index:04d}:c0002",
                            "row_index": index,
                            "column_index": 2,
                        },
                    },
                }],
                "value_raw": tg,
                "value_min": float(tg),
                "value_max": float(tg),
                "unit_raw": "K",
                "unit_normalized": "K",
                "measurement_context": None,
                "coverage_status": "covered",
                "evidence": [{
                    "block_id": "T_4_0",
                    "source_sentence": table,
                    "table_locator": {
                        "table_id": "T_4_0",
                        "row_label": sample,
                        "column_label": "Tg/K",
                        "cell_value": tg,
                        "cell_id": f"T_4_0:r{index:04d}:c0001",
                        "row_index": index,
                        "column_index": 1,
                    },
                }],
            })
        return LLMJSONResponse(
            data=add_model_confidence({
                "measurement_conditions": [],
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "series_id": "series001",
                    "sample_id": "s001",
                    "entity_id": "pe001",
                    "sample_resolution_status": "resolved",
                    "property_name_raw": "Tg",
                    "property_name_normalized": "glass_transition_temperature",
                    "property_code": "P3110",
                    "property_category": "thermal_property",
                    "determination_method_raw": None,
                    "observation_group_id": None,
                    "unit_raw": "K",
                    "unit_normalized": "K",
                    "measurement_context": {"condition_status": "not_reported"},
                    "points": points,
                    "coverage": None,
                    "evidence": [],
                }],
            }),
            provider="test",
            model="fake-actual",
        )


class PartiallyRepresentedMnClient(CoordinateOnlyMnClient):
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
        response.data["property_series"][0]["points"][1]["coordinates"] = []
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
            response = FakeClient().call_json(
                system_prompt,
                user_message,
                max_tokens=max_tokens,
            )
            response.data["properties"][0]["sample_id"] = "s999"
            return response
        self.calls -= 1
        return super().call_json(
            system_prompt,
            user_message,
            max_tokens=max_tokens,
        )


class InvalidVocabularyClient(FakeClient):
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
        response.data["properties"][0]["property_code"] = "P1110"
        return response


class NumericPropertyNameClient(FakeClient):
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
        response.data["properties"][0]["property_name_raw"] = "8.5 to 8.6"
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
        del response.data["properties"][0]["property_name_raw"]
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
        response.data["properties"][0]["evidence"][0]["block_id"] = (
            "P_missing"
        )
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
                "measurement_conditions": [],
                "properties": [],
                "unresolved_properties": [{
                    "unresolved_id": "uprop010",
                    "entity_id": "pe001",
                    "property_name_raw": "solubility parameter",
                    "value_raw": "8.5 to 8.6",
                    "unit_raw": "(cal/ml)^1/2",
                    "reason": "sample_ambiguous",
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": RESULT_SENTENCE,
                        "table_locator": None,
                    }],
                }],
            }),
            provider="test",
            model="fake-actual",
        )


class UnresolvedSampleConfidenceClient(UnresolvedClient):
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
        response.data["unresolved_properties"][0]["confidence"][
            "uncertain_fields"
        ] = ["sample_id", "property_name_normalized"]
        response.data["unresolved_properties"][0]["confidence"][
            "field_scores"
        ] = {"sample_id": 0.2, "property_name_normalized": 0.4}
        return response


class UnresolvedMethodClient(UnresolvedClient):
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
        unresolved = response.data["unresolved_properties"][0]
        unresolved["determination_method_raw"] = "measured"
        unresolved["observation_group_id"] = "pog010"
        unresolved["evidence"].append({
            "block_id": "P_1_0",
            "source_sentence": (
                "Solubility parameters were measured at 25 °C."
            ),
            "table_locator": None,
        })
        return response


class ParaphrasedTableMethodClient(UnresolvedClient):
    column_label = "[Q]"

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
        unresolved = response.data["unresolved_properties"][0]
        unresolved["determination_method_raw"] = (
            f"computed from {self.column_label}"
        )
        unresolved["evidence"] = [{
            "block_id": "T_2_0",
            "source_sentence": (
                "dried PB film | solubility parameter | "
                "8.5 to 8.6 (cal/ml)^1/2"
            ),
            "table_locator": {
                "table_id": "T_2_0",
                "row_label": "dried PB film",
                "column_label": self.column_label,
                "cell_value": "8.5 to 8.6 (cal/ml)^1/2",
            },
        }]
        return response


class ParaphrasedGenericTableMethodClient(ParaphrasedTableMethodClient):
    column_label = "Value"


class UnresolvedNumericNameClient(UnresolvedClient):
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
        response.data["unresolved_properties"][0][
            "property_name_raw"
        ] = "8.5 to 8.6"
        return response


class UnresolvedInvalidTableLocatorClient(UnresolvedClient):
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
        unresolved = response.data["unresolved_properties"][0]
        unresolved["evidence"] = [{
            "block_id": "T_2_0",
            "source_sentence": (
                "dried PB film | solubility parameter | "
                "8.5 to 8.6 (cal/ml)^1/2"
            ),
            "table_locator": {
                "table_id": "T_2_0",
                "row_label": "invented row",
                "column_label": "Value",
                "cell_value": "8.5 to 8.6 (cal/ml)^1/2",
            },
        }]
        return response


class MultiMethodClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        methods_and_values = [
            ("viscometry", "8.55"),
            ("turbidimetry", "8.60"),
            ("swelling measurements", "8.55"),
        ]
        properties = []
        for index, (method, value) in enumerate(
            methods_and_values,
            start=10,
        ):
            properties.append({
                "property_id": f"prop{index:03d}",
                "sample_id": "s001",
                "property_name_raw": "solubility parameter",
                "property_name_normalized": "solubility_parameter",
                "property_code": "P5110",
                "property_category": "physicochemical_property",
                "molecular_weight_type": None,
                "determination_method_raw": method,
                "observation_group_id": "pog010",
                "value_raw": value,
                "value_min": float(value),
                "value_max": float(value),
                "unit_raw": "(cal/ml)^1/2",
                "unit_normalized": "(cal/mL)^0.5",
                "measurement_condition_id": "mc010",
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": MULTI_METHOD_SENTENCE,
                    "table_locator": None,
                }],
            })
        return LLMJSONResponse(
            data=add_model_confidence({
                "measurement_conditions": [{
                    "condition_id": "mc010",
                    "temperature": None,
                    "frequency": None,
                    "humidity": None,
                    "pressure": None,
                    "wavelength": None,
                    "other_conditions": {},
                    "condition_status": "not_reported",
                    "evidence": {
                        "block_id": "P_2_0",
                        "source_sentence": MULTI_METHOD_SENTENCE,
                        "table_locator": None,
                    },
                }],
                "properties": properties,
                "unresolved_properties": [],
            }),
            provider="test",
            model="fake-actual",
        )


class InventedDeterminationMethodClient(MultiMethodClient):
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
        response.data["properties"][0]["determination_method_raw"] = (
            "osmometry"
        )
        return response


class SeparateDeterminationMethodClient(FakeClient):
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
        response.data["properties"][0]["determination_method_raw"] = (
            "independent viscosity method"
        )
        return response


class MissingAdditionalTableLocatorClient(FakeClient):
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
        response.data["properties"][0]["evidence"][1]["table_locator"] = None
        return response


class OnlyMissingTableLocatorClient(MissingAdditionalTableLocatorClient):
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
        response.data["properties"][0]["evidence"] = [
            response.data["properties"][0]["evidence"][1]
        ]
        return response


class OnlyInvalidTableLocatorClient(FakeClient):
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
        table_evidence = response.data["properties"][0]["evidence"][1]
        table_evidence["table_locator"]["row_label"] = "invented row"
        response.data["properties"][0]["evidence"] = [table_evidence]
        return response


class UnusedConditionClient(FakeClient):
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
        unused = json.loads(json.dumps(
            response.data["measurement_conditions"][0]
        ))
        unused["condition_id"] = "mc020"
        response.data["measurement_conditions"].append(unused)
        return response


class UnanchoredAdditionalEvidenceClient(FakeClient):
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
        response.data["properties"][0]["evidence"].append({
            "block_id": "P_3_0",
            "source_sentence": "This sentence was invented.",
            "table_locator": None,
        })
        return response


class TableConditionWithoutLocatorClient(FakeClient):
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
        condition = response.data["measurement_conditions"][0]
        condition["temperature"] = None
        condition["condition_status"] = "not_reported"
        condition["evidence"] = {
            "block_id": "T_2_0",
            "source_sentence": (
                "dried PB film | solubility parameter | "
                "8.5 to 8.6 (cal/ml)^1/2"
            ),
            "table_locator": None,
        }
        return response


class WrongConditionEvidenceClient(FakeClient):
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
        response.data["measurement_conditions"][0]["evidence"] = {
            "block_id": "P_3_0",
            "source_sentence": "The specimens were stored in the dark.",
            "table_locator": None,
        }
        return response


class CrossBlockConditionClient(FakeClient):
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
        condition = response.data["measurement_conditions"][0]
        condition["evidence"] = {
            "block_id": "P_3_0",
            "source_sentence": "The specimens were stored in the dark.",
            "table_locator": None,
        }
        prop = response.data["properties"][0]
        prop["determination_method_raw"] = "measured"
        prop["evidence"] = [
            {
                "block_id": "P_2_0",
                "source_sentence": (
                    "The solubility parameter of dried PB film was "
                    "8.5 to 8.6 (cal/ml)^1/2."
                ),
                "table_locator": None,
            },
            {
                "block_id": "P_1_0",
                "source_sentence": (
                    "Solubility parameters were measured at 25 °C."
                ),
                "table_locator": None,
            },
        ]
        return response


class WrongPropertyEvidenceClient(FakeClient):
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
        response.data["properties"][0]["evidence"] = [{
            "block_id": "P_3_0",
            "source_sentence": "The specimens were stored in the dark.",
            "table_locator": None,
        }]
        return response


def rendered_prompt():
    return PromptLoader().render_stage_prompt(
            "polymer.stage4.property",
            PropertyStageResponse,
            expected_stage="stage4_property",
        expected_output_schema="property_observation_schema.v7",
        )


class Stage4Tests(unittest.TestCase):
    @staticmethod
    def _salvage_condition(condition_id: str = "mc001", *, block_id: str = "P_2_0") -> dict:
        return {
            "condition_id": condition_id,
            "temperature": None,
            "frequency": None,
            "humidity": None,
            "pressure": None,
            "wavelength": None,
            "other_conditions": {},
            "other_condition_evidence": {},
            "condition_status": "not_reported",
            "evidence": {
                "block_id": block_id,
                "source_sentence": RESULT_SENTENCE,
                "table_locator": None,
            },
        }

    @staticmethod
    def _salvage_property(
        property_id: str,
        condition_id: str,
        *,
        block_id: str = "P_2_0",
    ) -> dict:
        return {
            "property_id": property_id,
            "sample_id": "s001",
            "property_name_raw": "solubility parameter",
            "property_name_normalized": "solubility_parameter",
            "property_code": "P5110",
            "property_category": "physicochemical_property",
            "molecular_weight_type": None,
            "determination_method_raw": None,
            "observation_group_id": None,
            "observation_role": "single",
            "series_id": None,
            "series_ids": None,
            "value_raw": "8.5 to 8.6",
            "value_min": 8.5,
            "value_max": 8.6,
            "unit_raw": "(cal/ml)^1/2",
            "unit_normalized": "(cal/mL)^0.5",
            "measurement_condition_id": condition_id,
            "measurement_context": None,
            "evidence": [{
                "block_id": block_id,
                "source_sentence": RESULT_SENTENCE,
                "table_locator": None,
            }],
        }

    @staticmethod
    def _salvage_series() -> dict:
        row = (
            "dried PB film | solubility parameter | "
            "8.5 to 8.6 (cal/ml)^1/2"
        )

        def point(point_id: str, block_id: str, value: str) -> dict:
            return {
                "point_id": point_id,
                "observation_role": "series_point",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "coordinates": [],
                "value_raw": value,
                "value_min": None,
                "value_max": None,
                "unit_raw": "(cal/ml)^1/2",
                "unit_normalized": "(cal/mL)^0.5",
                "measurement_context": None,
                "coverage_status": "covered",
                "evidence": [{
                    "block_id": block_id,
                    "source_sentence": row,
                    "table_locator": None,
                }],
            }

        return {
            "series_id": "series001",
            "sample_id": "s001",
            "entity_id": "pe001",
            "sample_resolution_status": "resolved",
            "property_name_raw": "solubility parameter",
            "property_name_normalized": "solubility_parameter",
            "property_code": "P5110",
            "property_category": "physicochemical_property",
            "determination_method_raw": None,
            "observation_group_id": None,
            "unit_raw": "(cal/ml)^1/2",
            "unit_normalized": "(cal/mL)^0.5",
            "measurement_context": {"condition_status": "not_reported"},
            "points": [
                point("pt001", "T_2_0", "8.5"),
                point("pt002", "missing_block", "8.6"),
            ],
            "coverage": {
                "expected": 2,
                "covered": 2,
                "missing": 0,
                "not_applicable": 0,
                "ratio": 1.0,
            },
            "evidence": [{
                "block_id": "T_2_0",
                "source_sentence": row,
                "table_locator": None,
            }],
        }

    def test_preview_salvage_drops_only_bad_property(self) -> None:
        payload = add_model_confidence({
            "measurement_conditions": [self._salvage_condition()],
            "properties": [
                self._salvage_property("prop001", "mc001"),
                self._salvage_property(
                    "prop002", "mc001", block_id="missing_block"
                ),
            ],
            "unresolved_properties": [],
            "property_series": [],
        })
        parsed = PropertyStageResponse.model_validate(payload)

        with self.assertRaises(KeyError):
            _materialize(parsed, stage0_document().elements)
        salvaged, materialized, report = _preview_salvage_materialization(
            parsed,
            stage0_document().elements,
        )

        self.assertEqual(
            [item.property_id for item in salvaged.properties],
            ["prop001"],
        )
        self.assertEqual(len(materialized[1]), 1)
        self.assertEqual(report["dropped_properties"], ["prop002"])

    def test_preview_salvage_drops_only_bad_series_point(self) -> None:
        payload = add_model_confidence({
            "measurement_conditions": [],
            "properties": [],
            "unresolved_properties": [],
            "property_series": [self._salvage_series()],
        })
        parsed = PropertyStageResponse.model_validate(payload)

        salvaged, materialized, report = _preview_salvage_materialization(
            parsed,
            stage0_document().elements,
        )

        self.assertEqual(len(salvaged.property_series), 1)
        self.assertEqual(
            [point.point_id for point in salvaged.property_series[0].points],
            ["pt001"],
        )
        self.assertEqual(salvaged.property_series[0].coverage.expected, 1)
        self.assertEqual(len(materialized[3][0].points), 1)
        self.assertEqual(
            report["dropped_points"],
            [{"series_id": "series001", "point_id": "pt002"}],
        )

    def test_preview_salvage_cleans_property_reference_to_bad_condition(self) -> None:
        payload = add_model_confidence({
            "measurement_conditions": [
                self._salvage_condition("mc001", block_id="missing_block"),
                self._salvage_condition("mc002"),
            ],
            "properties": [
                self._salvage_property("prop001", "mc001"),
                self._salvage_property("prop002", "mc002"),
            ],
            "unresolved_properties": [],
            "property_series": [],
        })
        parsed = PropertyStageResponse.model_validate(payload)

        salvaged, materialized, report = _preview_salvage_materialization(
            parsed,
            stage0_document().elements,
        )

        self.assertEqual(
            [item.condition_id for item in salvaged.measurement_conditions],
            ["mc002"],
        )
        self.assertEqual(
            [item.property_id for item in salvaged.properties],
            ["prop002"],
        )
        self.assertEqual(len(materialized[0]), 1)
        self.assertEqual(len(materialized[1]), 1)
        self.assertEqual(report["dropped_conditions"], ["mc001"])
        self.assertEqual(report["dropped_properties"], ["prop001"])

    def test_preview_removes_unscoped_resolved_scalar_property(self) -> None:
        payload = {
            "properties": [{
                "property_id": "prop001",
                "sample_id": None,
                "property_name_raw": "conductivity",
                "value_raw": "10^-2 S/cm",
            }],
            "unresolved_properties": [],
            "property_series": [],
        }

        strict, _ = _repair_candidate_response_payload(
            payload,
            stage3_document(),
        )
        self.assertEqual(len(strict["properties"]), 1)

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            preview_relaxed=True,
        )
        self.assertEqual(repaired["properties"], [])
        self.assertEqual(
            repairs["preview_unscoped_scalar_properties_removed"],
            1,
        )

    def test_preview_moves_legacy_property_out_of_series(self) -> None:
        legacy = {
            "property_id": "prop001",
            "sample_id": "s001",
            "series_id": None,
            "value_raw": "250 to 300 °C",
            "coverage": None,
        }

        repaired, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [legacy],
            },
            stage3_document(),
            preview_relaxed=True,
        )

        self.assertEqual(repaired["property_series"], [])
        self.assertNotIn("coverage", repaired["properties"][0])
        self.assertEqual(
            repaired["properties"][0].get("observation_role", "single"),
            "single",
        )
        self.assertEqual(
            repairs["preview_legacy_properties_moved_from_series"], 1
        )

    def test_preview_clears_and_drops_invalid_unresolved_fields(self) -> None:
        repaired, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "property_series": [],
                "unresolved_properties": [
                    {
                        "property_name_raw": "elongation",
                        "value_raw": "9%",
                        "property_category": "tensile_property",
                    },
                    {
                        "property_name_raw": "soluble",
                        "value_raw": "soluble",
                    },
                ],
            },
            stage3_document(),
            preview_relaxed=True,
        )

        self.assertIsNone(
            repaired["unresolved_properties"][0]["property_category"]
        )
        self.assertEqual(len(repaired["unresolved_properties"]), 1)
        self.assertEqual(
            repairs["preview_invalid_unresolved_properties_removed"], 1
        )

    def test_preview_clears_unlocatable_unresolved_unit_surface(self) -> None:
        payload = {
            "properties": [],
            "property_series": [],
            "unresolved_properties": [{
                "property_name_raw": "temperature",
                "value_raw": r"100 ^{\circ}\mathrm C",
                "unit_raw": "°C",
                "evidence": [{"block_id": "P_3_0"}],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=True,
        )

        self.assertIsNone(repaired["unresolved_properties"][0]["unit_raw"])
        self.assertEqual(
            repairs["preview_unresolved_unit_surfaces_cleared"], 1
        )

    def test_preview_removes_unanchored_other_condition(self) -> None:
        payload = {
            "properties": [],
            "property_series": [],
            "unresolved_properties": [],
            "measurement_conditions": [{
                "condition_id": "mc001",
                "condition_status": "reported",
                "other_conditions": {"atmosphere": r"$\Nu_2$"},
                "other_condition_evidence": {
                    "atmosphere": [{"block_id": "P_1_0"}],
                },
                "evidence": {"block_id": "P_1_0"},
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=True,
        )

        condition = repaired["measurement_conditions"][0]
        self.assertEqual(condition["other_conditions"], {})
        self.assertEqual(condition["other_condition_evidence"], {})
        self.assertEqual(
            repairs["preview_unanchored_other_conditions_removed"], 1
        )

    def test_preview_marks_coordinate_only_points_unresolved(self) -> None:
        payload = {
            "properties": [],
            "unresolved_properties": [],
            "property_series": [{
                "series_id": "series001",
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
                "points": [{
                    "point_id": "pt001",
                    "sample_id": None,
                    "entity_id": None,
                    "sample_resolution_status": None,
                    "coordinates": [{"name_raw": "sample", "value_raw": "A"}],
                    "evidence": [{"block_id": "T_1"}],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            preview_relaxed=True,
        )

        self.assertEqual(
            repaired["property_series"][0]["points"][0][
                "sample_resolution_status"
            ],
            "unresolved",
        )
        self.assertEqual(
            repairs["preview_unresolved_point_status_filled"], 1
        )

    def test_polymer_mw_subject_is_not_a_property_header(self) -> None:
        self.assertFalse(_looks_like_series_property_header(
            "Poly acrylic acid (PAA)(MW:2000)"
        ))
        self.assertTrue(_looks_like_series_property_header("Mw (g/mol)"))

    def test_prompt_requires_base_and_component_series_coverage(self) -> None:
        prompt = rendered_prompt()

        self.assertEqual(prompt.version, "1.7.2")
        self.assertIn("分量 Series 不得视为已覆盖普通端点", prompt.text)

    def test_prompt_disambiguates_aggregate_binding_from_guessing(self) -> None:
        # 规则 23 同时要求"必须绑定"与"不得猜测"，二者在覆盖范围不可核实时
        # 互相矛盾。23a 用"原文是否指名来源"消解该矛盾，并把不可核实的情况
        # 导向 unresolved，而不是无绑定的 aggregate。
        prompt = rendered_prompt()

        self.assertIn("可以核实", prompt.text)
        self.assertIn("改为输出 `unresolved_properties` 条目", prompt.text)
        self.assertIn("输出两个 Series 字段皆为 null 的", prompt.text)
        self.assertIn("不依据数值范围是否恰好吻合", prompt.text)

    def test_prompt_requires_minimal_verbatim_property_name(self) -> None:
        prompt = rendered_prompt()

        self.assertEqual(prompt.version, "1.7.2")
        self.assertIn("足以标识该结果的最短", prompt.text)
        self.assertIn("不得把不同句子的片段拼接", prompt.text)
        self.assertIn("所有 `*_raw` 字段必须", prompt.text)
        self.assertIn("不得静默省略", prompt.text)

    def test_multivalue_tg_table_requires_property_series(self) -> None:
        document = stage0_document()
        table = Stage0Document.model_validate({
            **document.model_dump(mode="json"),
            "elements": [
                *document.model_dump(mode="json")["elements"],
                {
                    "block_id": "T_4_0",
                    "type": "table",
                    "section": "Results",
                    "page": 4,
                    "bbox": [1, 2, 3, 4],
                    "source_block_index": 4,
                    "caption": "Glass transition data",
                    "table_body": (
                        "<table><tr><td>Sample</td><td>Tg, °C</td></tr>"
                        "<tr><td>A</td><td>-31</td></tr>"
                        "<tr><td>B</td><td>-43</td></tr></table>"
                    ),
                    "table_cells": [
                        {"cell_id": "T_4_0:r0000:c0000", "row_index": 0, "column_index": 0, "row_span": 1, "column_span": 1, "text": "Sample"},
                        {"cell_id": "T_4_0:r0000:c0001", "row_index": 0, "column_index": 1, "row_span": 1, "column_span": 1, "text": "Tg, °C"},
                        {"cell_id": "T_4_0:r0001:c0000", "row_index": 1, "column_index": 0, "row_span": 1, "column_span": 1, "text": "A"},
                        {"cell_id": "T_4_0:r0001:c0001", "row_index": 1, "column_index": 1, "row_span": 1, "column_span": 1, "text": "-31"},
                        {"cell_id": "T_4_0:r0002:c0000", "row_index": 2, "column_index": 0, "row_span": 1, "column_span": 1, "text": "B"},
                        {"cell_id": "T_4_0:r0002:c0001", "row_index": 2, "column_index": 1, "row_span": 1, "column_span": 1, "text": "-43"},
                    ],
                },
            ],
        })

        with self.assertRaisesRegex(
            ValueError,
            "既未作为 PropertySeries point，也未作为 PropertySeries coordinate 输出",
        ):
            _validate_required_table_series(table.elements, [])

    def test_surface_text_maps_unicode_macron_to_latex_overline(self) -> None:
        source = r"$\overline{Mn}^*$"

        self.assertEqual(
            _resolve_surface_text(source, "Mn\u0304*"),
            r"\overline{Mn}^*",
        )

    def test_surface_text_maps_html_entity_to_literal_character(self) -> None:
        source = "<td>BTDA/4,4&#x27;-BABBP</td>"

        self.assertEqual(
            _resolve_surface_text(source, "BTDA/4,4'-BABBP"),
            "BTDA/4,4&#x27;-BABBP",
        )

    def test_empty_reported_context_is_downgraded(self) -> None:
        payload = SeriesClient().call_json("", "").data
        payload["property_series"][0]["measurement_context"] = {
            "condition_status": "reported",
            "other_conditions": {},
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
        )

        self.assertEqual(
            repaired["property_series"][0]["measurement_context"][
                "condition_status"
            ],
            "not_reported",
        )
        self.assertEqual(repairs["empty_reported_contexts_downgraded"], 1)

    def test_preview_clears_controlled_fields_from_unresolved_property(
        self,
    ) -> None:
        payload = {
            "properties": [],
            "property_series": [],
            "measurement_conditions": [],
            "unresolved_properties": [{
                "unresolved_id": "u001",
                "property_name_raw": "thermal property",
                "property_name_normalized": "glass_transition_temperature",
                "property_code": "P2100",
                "property_category": "thermal",
                "value_raw": "high",
            }],
        }

        strict, _ = _repair_candidate_response_payload(
            payload,
            stage3_document(),
        )
        self.assertEqual(
            strict["unresolved_properties"][0]["property_name_normalized"],
            "glass_transition_temperature",
        )

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            preview_relaxed=True,
        )

        unresolved = repaired["unresolved_properties"][0]
        self.assertIsNone(unresolved["property_name_normalized"])
        self.assertIsNone(unresolved["property_code"])
        self.assertEqual(
            repairs["preview_unresolved_controlled_fields_cleared"],
            1,
        )

    def test_preview_synthesizes_missing_numeric_point_inside_series_rows(
        self,
    ) -> None:
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"].append({
            "block_id": "T_9_0",
            "type": "table",
            "section": "Results",
            "table_body": (
                "<table><tr><td>Sample</td><td>PMT</td></tr>"
                "<tr><td>A</td><td>100</td></tr>"
                "<tr><td>B</td><td>200</td></tr>"
                "<tr><td>C</td><td>300</td></tr></table>"
            ),
            "page": 9,
            "bbox": [1, 2, 3, 4],
            "source_block_index": 99,
        })
        document = Stage0Document.model_validate(document_data)

        def point(point_id: str, row: int, label: str, value: str) -> dict:
            return {
                "point_id": point_id,
                "coverage_status": "covered",
                "value_raw": value,
                "coordinates": [],
                "evidence": [{
                    "block_id": "T_9_0",
                    "source_sentence": (
                        f"<tr><td>{label}</td><td>{value}</td></tr>"
                    ),
                    "table_locator": {
                        "table_id": "T_9_0",
                        "row_label": label,
                        "column_label": "PMT",
                        "cell_value": value,
                        "cell_id": f"T_9_0:r{row:04d}:c0001",
                        "row_index": row,
                        "column_index": 1,
                    },
                }],
                "confidence": {"score": 0.9},
            }

        payload = {
            "properties": [],
            "unresolved_properties": [],
            "measurement_conditions": [],
            "property_series": [{
                "series_id": "series001",
                "sample_resolution_status": "unresolved",
                "entity_id": "pe001",
                "unit_raw": "°C",
                "unit_normalized": "°C",
                "points": [
                    point("pt001", 1, "A", "100"),
                    point("pt002", 3, "C", "300"),
                ],
                "confidence": {"score": 0.9},
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
            preview_relaxed=True,
        )

        points = repaired["property_series"][0]["points"]
        synthesized = next(item for item in points if item["value_raw"] == "200")
        self.assertEqual(synthesized["confidence"]["score"], 0.5)
        self.assertEqual(synthesized["sample_resolution_status"], "unresolved")
        self.assertEqual(synthesized["coordinates"][0]["value_raw"], "B")
        self.assertEqual(
            synthesized["evidence"][0]["table_locator"]["cell_id"],
            "T_9_0:r0002:c0001",
        )
        self.assertEqual(
            repairs["preview_unresolved_series_points_synthesized"],
            1,
        )

    def test_reported_context_with_value_is_not_downgraded(self) -> None:
        payload = SeriesClient().call_json("", "").data
        payload["property_series"][0]["measurement_context"] = {
            "condition_status": "reported",
            "temperature": {"raw": "25 °C"},
            "other_conditions": {},
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
        )

        self.assertEqual(
            repaired["property_series"][0]["measurement_context"][
                "condition_status"
            ],
            "reported",
        )
        self.assertEqual(repairs["empty_reported_contexts_downgraded"], 0)

    def test_surface_text_does_not_join_hierarchical_table_headers(self) -> None:
        source = (
            "LOSS AT 1st STEP (200-300°C) "
            "200 300 400 500 600 700 Calcd. Found"
        )

        self.assertIsNone(
            _resolve_surface_text(
                source,
                "LOSS AT 1st STEP (200-300°C) Calcd.",
            )
        )

    def test_surface_text_does_not_delete_semantic_words(self) -> None:
        source = "The electrical conductivity was measured."

        self.assertIsNone(
            _resolve_surface_text(source, "electrical conductivity range")
        )

    def test_blank_table_cell_value_is_normalized_to_null(self) -> None:
        response = FakeClient().call_json("", "")
        locator = response.data["properties"][0]["evidence"][1][
            "table_locator"
        ]
        locator["cell_value"] = "  "

        repaired, repairs = _repair_candidate_response_payload(
            response.data,
            stage3_document(),
            stage0_document().elements,
        )

        normalized = repaired["properties"][0]["evidence"][1][
            "table_locator"
        ]["cell_value"]
        self.assertIsNone(normalized)
        self.assertEqual(repairs["blank_table_cell_values_normalized"], 1)
        PropertyStageResponse.model_validate(repaired)

    def test_blank_coordinate_cell_uses_coordinate_value(self) -> None:
        payload = {
            "property_series": [{
                "points": [{
                    "coordinates": [{
                        "name_raw": "Sample",
                        "value_raw": "dried PB film",
                        "unit_raw": None,
                        "evidence": {
                            "block_id": "T_2_0",
                            "source_sentence": TABLE_BODY,
                            "table_locator": {
                                "table_id": "T_2_0",
                                "row_label": "dried PB film",
                                "column_label": "Sample",
                                "cell_value": "  ",
                            },
                        },
                    }],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        locator = repaired["property_series"][0]["points"][0][
            "coordinates"
        ][0]["evidence"]["table_locator"]
        self.assertEqual(locator["cell_value"], "dried PB film")
        self.assertIsNotNone(locator["cell_id"])
        self.assertEqual(repairs["blank_table_cell_values_normalized"], 1)
        self.assertEqual(repairs["coordinate_table_locators_synthesized"], 1)

    def test_condition_evidence_uses_related_property_anchor(self) -> None:
        payload = {
            "measurement_conditions": [{
                "condition_id": "mc010",
                "condition_status": "not_reported",
                "evidence": {
                    "block_id": "P_2_0",
                    "source_sentence": "A summarized condition sentence.",
                },
            }],
            "properties": [{
                "property_id": "prop010",
                "measurement_condition_id": "mc010",
                "property_name_raw": "solubility parameter",
                "value_raw": "8.5 to 8.6",
                "determination_method_raw": None,
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": RESULT_SENTENCE,
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        evidence = repaired["measurement_conditions"][0]["evidence"]
        self.assertEqual(evidence["source_sentence"], RESULT_SENTENCE)
        self.assertEqual(repairs["condition_evidence_surfaces_repaired"], 1)
        warnings = _candidate_repair_warnings(repairs)
        warning = next(
            item
            for item in warnings
            if item["code"] == "condition_evidence_surfaces_repaired"
        )
        self.assertEqual(warning["evidence"], 1)

    def test_embedded_measurement_context_promotes_missing_condition(
        self,
    ) -> None:
        payload = FakeClient().call_json("", "").data
        payload["measurement_conditions"] = []
        payload["properties"][0]["measurement_context"] = {
            "condition_status": "reported",
            "temperature": {
                "raw": "25 °C",
                "value": 25,
                "unit": "°C",
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": RESULT_SENTENCE,
                    "table_locator": None,
                }],
            },
            "frequency": None,
            "humidity": None,
            "pressure": None,
            "wavelength": None,
            "other_conditions": {},
            "other_condition_evidence": {},
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        self.assertEqual(
            repaired["measurement_conditions"][0]["condition_id"],
            "mc010",
        )
        self.assertEqual(
            repaired["measurement_conditions"][0]["evidence"]["block_id"],
            "P_2_0",
        )
        self.assertEqual(
            repairs["embedded_measurement_conditions_promoted"], 1
        )
        PropertyStageResponse.model_validate(repaired)

    def test_conflicting_embedded_context_does_not_promote_condition(
        self,
    ) -> None:
        payload = FakeClient().call_json("", "").data
        payload["measurement_conditions"] = []
        first = payload["properties"][0]
        first["measurement_context"] = {
            "condition_status": "reported",
            "temperature": {
                "raw": "25 °C",
                "value": 25,
                "unit": "°C",
                "evidence": [{
                    "block_id": "P_2_0",
                    "source_sentence": RESULT_SENTENCE,
                }],
            },
            "other_conditions": {},
            "other_condition_evidence": {},
        }
        second = json.loads(json.dumps(first))
        second["property_id"] = "prop011"
        second["measurement_context"]["temperature"]["value"] = 30
        payload["properties"].append(second)

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        self.assertEqual(repaired["measurement_conditions"], [])
        self.assertEqual(
            repairs["embedded_measurement_conditions_promoted"], 0
        )

    def test_missing_context_with_unique_evidence_becomes_not_reported(
        self,
    ) -> None:
        payload = FakeClient().call_json("", "").data
        payload["measurement_conditions"] = []
        payload["properties"][0]["measurement_context"] = None
        payload["properties"][0]["evidence"] = [
            payload["properties"][0]["evidence"][0]
        ]

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        condition = repaired["measurement_conditions"][0]
        self.assertEqual(condition["condition_id"], "mc010")
        self.assertEqual(condition["condition_status"], "not_reported")
        self.assertEqual(
            repairs["missing_conditions_marked_not_reported"], 1
        )
        PropertyStageResponse.model_validate(repaired)

    def test_coordinate_locator_id_is_aligned_to_evidence_block(self) -> None:
        payload = SeriesClient().call_json("", "").data
        coordinate_evidence = payload["property_series"][0]["points"][0][
            "coordinates"
        ][0]["evidence"]
        coordinate_evidence["table_locator"]["table_id"] = "Table 1"

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        locator = repaired["property_series"][0]["points"][0][
            "coordinates"
        ][0]["evidence"]["table_locator"]
        self.assertEqual(locator["table_id"], "T_2_0")
        self.assertGreaterEqual(
            repairs["table_locator_ids_aligned_to_evidence"], 1
        )

    def test_condition_raw_synthesizes_unique_table_header_locator(self) -> None:
        payload = {
            "property_series": [{
                "measurement_context": {
                    "condition_status": "reported",
                    "temperature": {
                        "raw": "Value",
                        "evidence": [{
                            "block_id": "T_2_0",
                            "source_sentence": "Value",
                        }],
                    },
                },
                "points": [],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        locator = repaired["property_series"][0]["measurement_context"][
            "temperature"
        ]["evidence"][0]["table_locator"]
        self.assertEqual(locator["cell_value"], "Value")
        self.assertIsNotNone(locator["cell_id"])
        self.assertEqual(repairs["condition_table_locators_synthesized"], 1)

    def test_legacy_measurement_condition_uses_unique_evidence_header(self) -> None:
        payload = {
            "measurement_conditions": [{
                "condition_id": "mc001",
                "condition_status": "reported",
                "temperature": {
                    "raw": "Value",
                    "evidence": [{
                        "block_id": "T_2_0",
                        "source_sentence": "Value",
                    }],
                },
                "evidence": {
                    "block_id": "T_2_0",
                    "source_sentence": "Value",
                },
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        condition = repaired["measurement_conditions"][0]
        self.assertIsNotNone(condition["evidence"]["table_locator"]["cell_id"])
        self.assertIsNotNone(
            condition["temperature"]["evidence"][0]["table_locator"]["cell_id"]
        )
        self.assertEqual(repairs["condition_table_locators_synthesized"], 2)

    def test_placeholder_cell_is_not_used_as_condition_anchor(self) -> None:
        """占位单元格不得成为条件 locator 的锚点。

        真实案例 reference_no_0071569：表格 T_3_35 含一个表示「无数据」的
        "-" 单元格，而条件文本在 caption 里（"Measured in m-cresol at
        30.0°C."）。反向包含匹配会让 "-" 命中 m-cresol 的连字符，把测量条件
        锚定到不含任何条件信息的占位格上——违反契约 §5 不变量 1
        （*_raw 必须是合法 evidence 中可定位的最小原文片段）。
        正确行为是不合成 locator，交由既有降级路径保留块级证据。
        """
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.caption = "Measured in m-cresol at 30.0 C."
        table.table_body = (
            "| Sample | Value | Note |\n"
            "|---|---|---|\n"
            "| dried PB film | 8.5 | - |"
        )
        table.table_cells = None
        payload = {
            "measurement_conditions": [{
                "condition_id": "mc001",
                "condition_status": "reported",
                "temperature": {
                    "raw": "30.0 C",
                    "evidence": [{
                        "block_id": "T_2_0",
                        "source_sentence": "Measured in m-cresol at 30.0 C.",
                    }],
                },
                "evidence": {
                    "block_id": "T_2_0",
                    "source_sentence": "Measured in m-cresol at 30.0 C.",
                },
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
        )

        condition = repaired["measurement_conditions"][0]
        self.assertIsNone(condition["evidence"].get("table_locator"))
        self.assertIsNone(
            condition["temperature"]["evidence"][0].get("table_locator")
        )
        self.assertEqual(repairs["condition_table_locators_synthesized"], 0)

    def test_condition_parent_reuses_unique_field_header_locator(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Density | Alcohol immersion (at 27 C) |\n"
            "| --- | --- |\n| sample | 0.9 |"
        )
        table.table_cells = None
        payload = {
            "measurement_conditions": [{
                "condition_id": "mc001",
                "condition_status": "reported",
                "temperature": {
                    "raw": "27 C",
                    "evidence": [{
                        "block_id": table.block_id,
                        "source_sentence": "Alcohol immersion (at 27 C)",
                    }],
                },
                "evidence": {
                    "block_id": table.block_id,
                    "source_sentence": "Density Alcohol immersion (at 27 C)",
                },
            }],
        }

        repaired, _ = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
        )

        condition = repaired["measurement_conditions"][0]
        self.assertEqual(
            condition["evidence"]["table_locator"]["cell_id"],
            condition["temperature"]["evidence"][0]["table_locator"]["cell_id"],
        )

    def test_condition_header_match_prefers_full_anchor_over_short_cell(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| $n^30_D$ | Result |\n"
            "| --- | --- |\n| sample | 0 |"
        )
        table.table_cells = None
        payload = {
            "measurement_conditions": [{
                "condition_id": "mc001",
                "condition_status": "reported",
                "temperature": {
                    "raw": "30",
                    "evidence": [{
                        "block_id": table.block_id,
                        "source_sentence": "$n^30_D$",
                    }],
                },
                "evidence": {
                    "block_id": table.block_id,
                    "source_sentence": "$n^30_D$",
                },
            }],
        }

        repaired, _ = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
        )

        locator = repaired["measurement_conditions"][0]["temperature"][
            "evidence"
        ][0]["table_locator"]
        self.assertEqual(locator["cell_value"], "$n^30_D$")

    def test_property_locator_prefers_specific_header_over_group_row(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "<table><tr><th>Sample</th><th>$T_g$ (dila-tometric)</th>"
            "<th>Other</th></tr><tr><th colspan='3'>Vinyl ethers</th></tr>"
            "<tr><td>Sample A</td><td>-55</td><td>1</td></tr></table>"
        )
        table.table_cells = parse_table_cells(table.table_body, table.block_id)
        payload = {
            "properties": [{
                "property_name_raw": "$T_g$ (dilatometric)",
                "determination_method_raw": "dilatometric",
                "evidence": [{
                    "block_id": table.block_id,
                    "source_sentence": "Sample A -55",
                    "table_locator": {
                        "table_id": table.block_id,
                        "row_label": "Sample A",
                        "column_label": "$T_g$ (dilatometric)",
                        "cell_value": "-55",
                    },
                }],
            }],
        }

        repaired, _ = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
        )

        locator = repaired["properties"][0]["evidence"][0]["table_locator"]
        self.assertEqual(locator["column_label"], "$T_g$ (dila-tometric)")

    def test_series_unit_restores_unique_parenthesized_header_surface(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        header = (
            r"$\Delta C_p$  (cal g $^{-1}$  °C $^{-1}$ ) "
            r"( $\times 10^2$ )"
        )
        table.table_body = (
            f"<table><tr><td>Composition</td><td>{header}</td></tr>"
            "<tr><td>0.00</td><td>11.10</td></tr></table>"
        )
        table.table_cells = parse_table_cells(table.table_body, table.block_id)
        model_unit = r"cal g $^{-1}$  °C $^{-1}$ ( $\times 10^2$ )"
        evidence = {
            "block_id": table.block_id,
            "source_sentence": "11.10",
            "table_locator": {
                "table_id": table.block_id,
                "row_label": "0.00",
                "column_label": header,
                "cell_value": "11.10",
            },
        }
        payload = {
            "property_series": [{
                "series_id": "series001",
                "property_name_raw": r"$\Delta C_p$",
                "unit_raw": model_unit,
                "evidence": [],
                "points": [{
                    "point_id": "pt001",
                    "unit_raw": model_unit,
                    "coordinates": [],
                    "evidence": [evidence],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
        )

        series = repaired["property_series"][0]
        repaired_header = series["points"][0]["evidence"][0][
            "table_locator"
        ]["column_label"]
        resolved_name = _resolve_surface_text(
            repaired_header,
            r"$\Delta C_p$",
        )
        self.assertIsNotNone(resolved_name)
        expected = repaired_header[
            repaired_header.find(resolved_name) + len(resolved_name):
        ].strip()
        self.assertEqual(series["unit_raw"], expected)
        self.assertEqual(series["points"][0]["unit_raw"], expected)
        self.assertEqual(repairs["series_unit_surfaces_repaired"], 2)
        warning_codes = {
            item["code"] for item in _candidate_repair_warnings(repairs)
        }
        self.assertIn("series_unit_surface_repaired", warning_codes)

    def test_property_name_recovers_unique_hyphenated_header_surface(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Sample | $T_g$ (dila-tometric) |\n"
            "| --- | --- |\n| Sample A | -55 |"
        )
        table.table_cells = None
        evidence = [PropertyEvidenceCandidate.model_validate({
            "block_id": table.block_id,
            "source_sentence": "Sample A -55",
            "table_locator": {
                "table_id": table.block_id,
                "row_label": "Sample A",
                "column_label": "$T_g$ (dila-tometric)",
                "cell_value": "-55",
            },
        })]

        normalized = _normalize_property_name(
            "$T_g$ (dilatometric)",
            evidence,
            {table.block_id: table},
            "prop001.property_name_raw",
        )

        self.assertEqual(normalized, "$T_g$ (dila-tometric)")

    def test_property_name_accepts_exact_peak_but_rejects_interpretation(self) -> None:
        document = stage0_document().model_copy(deep=True)
        block = next(item for item in document.elements if item.block_id == "P_2_0")
        block.text = (
            "A flat peak was detected for PTS at 198 ± 2 K. "
            "These Cp peaks might be due to phase transitions."
        )
        evidence = [PropertyEvidenceCandidate.model_validate({
            "block_id": block.block_id,
            "source_sentence": "A flat peak was detected for PTS at 198 ± 2 K.",
        })]

        self.assertEqual(
            _normalize_property_name(
                "flat peak",
                evidence,
                {block.block_id: block},
                "prop001.property_name_raw",
            ),
            "flat peak",
        )
        with self.assertRaises(ValueError):
            _normalize_property_name(
                "flat peak in Cp (phase transition)",
                evidence,
                {block.block_id: block},
                "prop001.property_name_raw",
            )

    def test_property_name_recovers_header_with_latex_footnote(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        header = "Solubility parameter, $^a$ cal. $^{1/2}$ /cc. $^{1/2}$"
        table.table_body = (
            f"| Sample | {header} |\n"
            "| --- | --- |\n| Sample A | 8.02 |"
        )
        table.table_cells = None
        evidence = [PropertyEvidenceCandidate.model_validate({
            "block_id": table.block_id,
            "source_sentence": "Sample A 8.02",
            "table_locator": {
                "table_id": table.block_id,
                "row_label": "Sample A",
                "column_label": header,
                "cell_value": "8.02",
            },
        })]

        normalized = _normalize_property_name(
            "Solubility parameter, cal.$^{1/2}$/cc.$^{1/2}$",
            evidence,
            {table.block_id: table},
            "prop001.property_name_raw",
        )

        self.assertEqual(normalized, header)

    def test_property_name_falls_back_to_unique_raw_symbol(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Sample | Result |\n"
            "| --- | --- |\n| Sample A | $T_m = 30^e$ |"
        )
        table.table_cells = None
        evidence = [PropertyEvidenceCandidate.model_validate({
            "block_id": table.block_id,
            "source_sentence": "Sample A $T_m = 30^e$",
            "table_locator": {
                "table_id": table.block_id,
                "row_label": "Sample A",
                "column_label": "Result",
                "cell_value": "$T_m = 30^e$",
            },
        })]

        normalized = _normalize_property_name(
            "Melting temperature ($T_m$, side chain crystallization)",
            evidence,
            {table.block_id: table},
            "prop001.property_name_raw",
        )

        self.assertEqual(normalized, "T_m")

    def test_surface_text_maps_compact_fraction_unit_to_latex(self) -> None:
        source = (
            "Solubility parameter, $^a$ "
            "cal. $^{1/2}$ /cc. $^{1/2}$"
        )

        normalized = _resolve_surface_text(source, "cal.1/2/cc.1/2")

        self.assertEqual(normalized, "cal. $^{1/2}$ /cc. $^{1/2}")

    def test_surface_text_maps_operatorname_star_unit(self) -> None:
        source = r"heating rate of 10 K $\operatorname* { m i n } ^ { - 1 }$"

        normalized = _resolve_surface_text(source, "10 K min^-1")

        self.assertEqual(
            normalized,
            r"10 K $\operatorname* { m i n } ^ { - 1 }",
        )

    def test_method_recovers_best_noncontiguous_source_fragment(self) -> None:
        document = stage0_document().model_copy(deep=True)
        text_block = next(item for item in document.elements if item.type == "text")
        text_block.text = (
            "The low temperature torsion flex test was carried out "
            "according to ASTM D 1053–54T."
        )
        evidence = [PropertyEvidenceCandidate.model_validate({
            "block_id": text_block.block_id,
            "source_sentence": text_block.text,
        })]

        normalized, recovered = _normalize_determination_method(
            "torsion flex test, ASTM D 1053–54T",
            evidence,
            {text_block.block_id: text_block},
            "prop001.determination_method_raw",
        )

        self.assertEqual(normalized, "torsion flex test")
        self.assertTrue(recovered)

    def test_method_recovers_fragment_split_by_and(self) -> None:
        document = stage0_document().model_copy(deep=True)
        text_block = next(item for item in document.elements if item.type == "text")
        text_block.text = (
            "Values were obtained from linear compressibility data of "
            "Lochner et al. as well as from a formula used by Choy."
        )
        evidence = [PropertyEvidenceCandidate.model_validate({
            "block_id": text_block.block_id,
            "source_sentence": text_block.text,
        })]

        normalized, recovered = _normalize_determination_method(
            "linear compressibility data of Lochner et al. and formula used by Choy",
            evidence,
            {text_block.block_id: text_block},
            "prop001.determination_method_raw",
        )

        self.assertEqual(
            normalized,
            "linear compressibility data of Lochner et al.",
        )
        self.assertTrue(recovered)

    def test_confidence_array_paths_are_normalized_before_schema(self) -> None:
        payload = {
            "property_series": [{
                "confidence": {
                    "score": 0.8,
                    "uncertain_fields": ["points[].evidence"],
                    "field_scores": {"points[].value_raw": 0.5},
                },
                "points": [],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        confidence = repaired["property_series"][0]["confidence"]
        self.assertEqual(confidence, {"score": 0.8})
        self.assertEqual(repairs["confidence_paths_normalized"], 0)

    def test_known_confidence_aliases_and_description_are_normalized(self) -> None:
        payload = {
            "measurement_conditions": [{
                "evidence": [{"block_id": "P_2_0"}],
            }],
            "properties": [{
                "confidence": {
                    "score": 0.8,
                    "field_scores": {"value": 0.8, "unit": 0.7},
                },
            }],
            "property_series": [{
                "confidence": {
                    "score": 0.7,
                    "uncertain_fields": [
                        "sample_id for methyl and n-Pentyl rows",
                    ],
                },
                "points": [],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        self.assertIsInstance(
            repaired["measurement_conditions"][0]["evidence"],
            dict,
        )
        self.assertEqual(repaired["properties"][0]["confidence"], {"score": 0.8})
        self.assertEqual(repaired["property_series"][0]["confidence"], {"score": 0.7})
        self.assertEqual(repairs["singleton_condition_evidence_unwrapped"], 1)
        self.assertEqual(repairs["confidence_field_aliases_normalized"], 0)
        self.assertEqual(repairs["confidence_field_descriptions_normalized"], 0)

    def test_other_confidence_descriptions_remain_invalid(self) -> None:
        payload = SeriesClient().call_json("", "").data
        payload["property_series"][0]["confidence"][
            "uncertain_fields"
        ] = ["entity_id inferred from nearby discussion"]

        repaired, _ = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        self.assertEqual(
            repaired["property_series"][0]["confidence"],
            {"score": 0.9},
        )

    def test_redundant_indirect_relation_description_is_removed(self) -> None:
        payload = {
            "measurement_conditions": [{
                "confidence": {
                    "score": 0.8,
                    "uncertain_fields": [
                        "association with melting temperature",
                    ],
                    "uncertainty_codes": ["indirect_relation"],
                },
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        self.assertEqual(
            repaired["measurement_conditions"][0]["confidence"],
            {"score": 0.8},
        )
        self.assertEqual(repairs["redundant_confidence_descriptions_removed"], 0)

    def test_series_confidence_drops_direct_point_only_fields(self) -> None:
        payload = SeriesClient().call_json("", "").data
        confidence = payload["property_series"][0]["confidence"]
        confidence["uncertain_fields"] = [
            "value_min",
            "value_max",
            "points.value_min",
            "property_name_raw",
        ]
        confidence["field_scores"] = {
            "value_min": 0.4,
            "value_max": 0.4,
            "points.value_max": 0.5,
            "property_name_raw": 0.6,
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        repaired_confidence = repaired["property_series"][0]["confidence"]
        self.assertEqual(repaired_confidence, {"score": 0.9})
        self.assertEqual(repairs["series_point_confidence_fields_removed"], 0)
        PropertyStageResponse.model_validate(repaired)
        self.assertFalse(any(
            item["code"] == "series_point_confidence_fields_removed"
            for item in _candidate_repair_warnings(repairs)
        ))

    def test_series_confidence_keeps_other_unknown_fields_invalid(self) -> None:
        payload = SeriesClient().call_json("", "").data
        payload["property_series"][0]["confidence"][
            "uncertain_fields"
        ] = ["input_relation"]

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
        )

        self.assertEqual(
            repaired["property_series"][0]["confidence"],
            {"score": 0.9},
        )
        self.assertEqual(
            repairs["series_point_confidence_fields_removed"],
            0,
        )
        PropertyStageResponse.model_validate(repaired)

    def test_duplicate_point_value_is_aligned_to_unique_coordinate_row(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Sample | Value | x |\n| --- | --- | --- |\n"
            "| A | 3.2 | 1 |\n| B | 3.2 | 2 |"
        )
        table.table_cells = None
        payload = {
            "property_series": [{
                "points": [{
                    "evidence": [{
                        "block_id": table.block_id,
                        "table_locator": {
                            "table_id": table.block_id,
                            "row_label": "group",
                            "column_label": "Value",
                            "cell_value": "3.2",
                            "cell_id": f"{table.block_id}:r0001:c0001",
                            "row_index": 1,
                            "column_index": 1,
                        },
                    }],
                    "coordinates": [{
                        "name_raw": "x",
                        "value_raw": "2",
                        "evidence": {
                            "block_id": table.block_id,
                            "table_locator": {
                                "table_id": table.block_id,
                                "row_label": "B",
                                "column_label": "x",
                                "cell_value": "2",
                            },
                        },
                    }],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
        )

        locator = repaired["property_series"][0]["points"][0][
            "evidence"
        ][0]["table_locator"]
        self.assertEqual(locator["row_index"], 2)
        self.assertEqual(repairs["point_locators_aligned_to_coordinates"], 1)

    def test_duplicate_coordinate_is_aligned_to_point_rowspan(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "<table><tr><td>Group</td><td>Tg</td></tr>"
            "<tr><td rowspan=\"2\">Methyl</td><td>-34</td></tr>"
            "<tr><td>-31</td></tr>"
            "<tr><td>Methyl</td><td>-1</td></tr></table>"
        )
        table.table_cells = None
        payload = {
            "property_series": [{
                "points": [{
                    "evidence": [{
                        "block_id": table.block_id,
                        "table_locator": {
                            "table_id": table.block_id,
                            "row_label": "Methyl",
                            "column_label": "Tg",
                            "cell_value": "-31",
                        },
                    }],
                    "coordinates": [{
                        "name_raw": "Group",
                        "value_raw": "Methyl",
                        "evidence": {
                            "block_id": table.block_id,
                            "table_locator": {
                                "table_id": table.block_id,
                                "row_label": "Methyl",
                                "column_label": "Group",
                                "cell_value": "Methyl",
                            },
                        },
                    }],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
        )

        locator = repaired["property_series"][0]["points"][0][
            "coordinates"
        ][0]["evidence"]["table_locator"]
        self.assertEqual(locator["cell_id"], f"{table.block_id}:r0001:c0000")
        self.assertEqual(repairs["coordinate_locators_aligned_to_point"], 1)

    def test_compound_locator_is_aligned_to_exact_value_in_rowspan(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "<table><tr><td>Group</td><td>Tg</td><td>Other</td></tr>"
            "<tr><td rowspan=\"2\">Methyl</td><td>-34</td>"
            "<td>-10, -31</td></tr>"
            "<tr><td>-31</td><td></td></tr></table>"
        )
        table.table_cells = None
        payload = {
            "property_series": [{
                "points": [{
                    "value_raw": "-31",
                    "evidence": [{
                        "block_id": table.block_id,
                        "table_locator": {
                            "table_id": table.block_id,
                            "row_label": "Methyl",
                            "column_label": "Tg",
                            "cell_value": "-10, -31",
                            "cell_id": f"{table.block_id}:r0001:c0002",
                            "row_index": 1,
                            "column_index": 2,
                        },
                    }],
                    "coordinates": [{
                        "name_raw": "Group",
                        "value_raw": "Methyl",
                        "evidence": {
                            "block_id": table.block_id,
                            "table_locator": {
                                "table_id": table.block_id,
                                "row_label": "Methyl",
                                "column_label": "Group",
                                "cell_value": "Methyl",
                                "cell_id": f"{table.block_id}:r0001:c0000",
                                "row_index": 1,
                                "column_index": 0,
                            },
                        },
                    }],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
        )

        locator = repaired["property_series"][0]["points"][0][
            "evidence"
        ][0]["table_locator"]
        self.assertEqual(locator["cell_id"], f"{table.block_id}:r0002:c0001")
        self.assertEqual(locator["cell_value"], "-31")
        self.assertEqual(
            repairs["point_compound_locators_aligned_to_coordinates"],
            1,
        )

    def test_compound_locator_is_not_aligned_when_exact_value_is_ambiguous(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "<table><tr><td>Group</td><td>Tg</td><td>Other</td></tr>"
            "<tr><td rowspan=\"2\">Methyl</td><td>-31</td>"
            "<td>-10, -31</td></tr>"
            "<tr><td>-31</td><td></td></tr></table>"
        )
        table.table_cells = None
        original_cell_id = f"{table.block_id}:r0001:c0002"
        payload = {
            "property_series": [{
                "points": [{
                    "value_raw": "-31",
                    "evidence": [{
                        "block_id": table.block_id,
                        "table_locator": {
                            "table_id": table.block_id,
                            "row_label": "Methyl",
                            "column_label": "Other",
                            "cell_value": "-10, -31",
                            "cell_id": original_cell_id,
                            "row_index": 1,
                            "column_index": 2,
                        },
                    }],
                    "coordinates": [{
                        "name_raw": "Group",
                        "value_raw": "Methyl",
                        "evidence": {
                            "block_id": table.block_id,
                            "table_locator": {
                                "table_id": table.block_id,
                                "row_label": "Methyl",
                                "column_label": "Group",
                                "cell_value": "Methyl",
                                "cell_id": f"{table.block_id}:r0001:c0000",
                                "row_index": 1,
                                "column_index": 0,
                            },
                        },
                    }],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            document.elements,
        )

        locator = repaired["property_series"][0]["points"][0][
            "evidence"
        ][0]["table_locator"]
        self.assertEqual(locator["cell_id"], original_cell_id)
        self.assertEqual(
            repairs["point_compound_locators_aligned_to_coordinates"],
            0,
        )

    def test_normalize_evidence_preserves_valid_stable_duplicate_cell(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Sample | Value |\n| --- | --- |\n"
            "| Polystyrene | |\n| | 3.2 |\n| | 3.2 |"
        )
        table.table_cells = None
        candidate = PropertyEvidenceCandidate.model_validate({
            "block_id": table.block_id,
            "source_sentence": "3.2",
            "table_locator": {
                "table_id": table.block_id,
                "row_label": "Polystyrene",
                "column_label": "Value",
                "cell_value": "3.2",
                "cell_id": f"{table.block_id}:r0003:c0001",
                "row_index": 3,
                "column_index": 1,
            },
        })
        block_map = {item.block_id: item for item in document.elements}

        from stages.stage4_property import _normalize_evidence

        normalized = _normalize_evidence(candidate, block_map, ["3.2"])

        self.assertEqual(
            normalized.table_locator.cell_id,
            f"{table.block_id}:r0003:c0001",
        )

    def test_blank_table_cell_uses_row_label_as_evidence_anchor(self) -> None:
        candidate = PropertyEvidenceCandidate.model_validate({
            "block_id": "T_2_0",
            "source_sentence": "synthetic empty cell",
            "table_locator": {
                "table_id": "T_2_0",
                "row_label": "dried PB film",
                "column_label": "Value",
                "cell_value": None,
            },
        })
        block_map = {
            block.block_id: block for block in stage0_document().elements
        }

        from stages.stage4_property import _normalize_evidence

        normalized = _normalize_evidence(candidate, block_map, [])

        self.assertIsNone(normalized.table_locator.cell_value)
        self.assertIn("dried PB film", normalized.source_sentence)

    def test_explicit_condition_field_evidence_does_not_fall_back(self) -> None:
        document = stage0_document()
        block_map = {item.block_id: item for item in document.elements}
        explicit = PropertyEvidenceCandidate.model_validate({
            "block_id": "T_2_0",
            "source_sentence": TABLE_BODY,
            "table_locator": None,
        })
        fallback = PropertyEvidenceCandidate.model_validate({
            "block_id": "P_2_0",
            "source_sentence": RESULT_SENTENCE,
            "table_locator": None,
        })

        with self.assertRaisesRegex(ValueError, "字段专属 evidence"):
            _normalize_condition_field_evidence(
                [explicit],
                [fallback],
                "25 °C",
                block_map,
                "series001.measurement_context.temperature",
            )

    def test_candidate_repairs_unique_series_and_point_confidence(self) -> None:
        confidence = {"score": 0.8}
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [{
                    "property_id": "prop001",
                    "sample_id": "s001",
                    "property_name_raw": "$\\chi$",
                    "property_name_normalized": None,
                    "unit_raw": "unit",
                    "observation_role": "aggregate",
                    "series_id": "series002",
                    "evidence": [{"block_id": "T_2_0"}],
                    "confidence": add_model_confidence({
                        "properties": [{}],
                    })["properties"][0]["confidence"],
                }],
                "unresolved_properties": [{
                    "unresolved_id": "uprop001",
                    "entity_id": "pe001",
                    "property_name_raw": "$\\delta_p$",
                    "property_name_normalized": None,
                    "observation_role": "aggregate",
                    "series_id": None,
                    "confidence": add_model_confidence({
                        "unresolved_properties": [{}],
                    })["unresolved_properties"][0]["confidence"],
                }],
                "property_series": [
                    {
                        "series_id": "series001",
                        "sample_id": "s001",
                        "entity_id": "pe001",
                        "property_name_raw": "$\\chi$",
                        "property_name_normalized": None,
                        "unit_raw": "unit",
                        "evidence": [{"block_id": "T_2_0"}],
                        "confidence": confidence,
                        "points": [{"point_id": "pt001"}],
                    },
                    {
                        "series_id": "series002",
                        "entity_id": "pe001",
                        "property_name_raw": "$[\\Phi]$",
                        "property_name_normalized": None,
                        "confidence": confidence,
                        "points": [],
                    },
                ],
            },
            stage3_document(),
        )

        prop = payload["properties"][0]
        unresolved = payload["unresolved_properties"][0]
        point_confidence = payload["property_series"][0]["points"][0][
            "confidence"
        ]
        self.assertEqual(prop["series_id"], "series001")
        self.assertEqual(unresolved["observation_role"], "aggregate")
        self.assertIsNone(unresolved["series_id"])
        self.assertEqual(point_confidence, {"score": 0.5})
        self.assertEqual(repairs["aggregate_linked"], 1)
        self.assertEqual(repairs["point_confidence_inherited"], 1)

    def test_candidate_does_not_guess_between_multiple_series(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [{
                    "property_id": "prop001",
                    "sample_id": "s001",
                    "property_name_raw": "$\\chi$",
                    "property_name_normalized": None,
                    "unit_raw": "unit",
                    "observation_role": "aggregate",
                    "series_id": None,
                    "evidence": [{"block_id": "T_2_0"}],
                    "confidence": add_model_confidence({
                        "properties": [{}],
                    })["properties"][0]["confidence"],
                }],
                "unresolved_properties": [],
                "property_series": [
                    {
                        "series_id": series_id,
                        "sample_id": "s001",
                        "entity_id": "pe001",
                        "property_name_raw": "$\\chi$",
                        "property_name_normalized": None,
                        "unit_raw": "unit",
                        "evidence": [{"block_id": "T_2_0"}],
                        "confidence": add_model_confidence({
                            "property_series": [{}],
                        })["property_series"][0]["confidence"],
                        "points": [],
                    }
                    for series_id in ("series001", "series002")
                ],
            },
            stage3_document(),
        )

        self.assertIsNone(payload["properties"][0]["series_id"])
        self.assertNotIn("series_ids", payload["properties"][0])
        self.assertEqual(repairs["aggregate_linked"], 0)
        self.assertEqual(repairs["aggregate_multi_linked"], 0)

    def test_candidate_links_unique_series_by_exact_value_bounds(self) -> None:
        confidence = {"score": 0.8}
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [{
                    "property_id": "prop001",
                    "sample_id": "s001",
                    "property_name_raw": "Tg",
                    "property_name_normalized": "glass_transition_temperature",
                    "property_code": "P3110",
                    "property_category": "thermal_property",
                    "value_raw": "192 and 247 °C",
                    "value_min": 192,
                    "value_max": 247,
                    "unit_raw": "°C",
                    "observation_role": "aggregate",
                    "evidence": [{"block_id": "P_0_0"}],
                    "confidence": confidence,
                }],
                "unresolved_properties": [{
                    "unresolved_id": "uprop001",
                    "entity_id": "pe001",
                    "property_name_raw": "Tg",
                    "value_raw": "192 and 247 °C",
                    "unit_raw": "°C",
                    "observation_role": "aggregate",
                    "evidence": [{"block_id": "P_0_0"}],
                    "confidence": confidence,
                }],
                "property_series": [{
                    "series_id": "series001",
                    "property_name_raw": "Tg",
                    "property_name_normalized": "glass_transition_temperature",
                    "property_code": "P3110",
                    "property_category": "thermal_property",
                    "unit_raw": "°C",
                    "points": [
                        {"point_id": "pt001", "value_min": 247, "value_max": 247},
                        {"point_id": "pt002", "value_min": 192, "value_max": 192},
                    ],
                }],
            },
            stage3_document(),
        )

        self.assertEqual(payload["properties"][0]["series_id"], "series001")
        self.assertEqual(payload["unresolved_properties"], [])
        self.assertEqual(repairs["aggregate_range_linked"], 1)
        self.assertEqual(repairs["duplicate_unresolved_aggregates_removed"], 1)

    def test_candidate_does_not_guess_duplicate_exact_value_bounds(self) -> None:
        confidence = {"score": 0.8}
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [{
                    "property_id": "prop001",
                    "sample_id": "s001",
                    "property_name_raw": "Tg",
                    "property_name_normalized": "glass_transition_temperature",
                    "value_raw": "192 and 247 °C",
                    "value_min": 192,
                    "value_max": 247,
                    "unit_raw": "°C",
                    "observation_role": "aggregate",
                    "evidence": [{"block_id": "P_0_0"}],
                    "confidence": confidence,
                }],
                "unresolved_properties": [],
                "property_series": [
                    {
                        "series_id": series_id,
                        "property_name_raw": "Tg",
                        "property_name_normalized": "glass_transition_temperature",
                        "unit_raw": "°C",
                        "points": [
                            {"point_id": f"{series_id}_a", "value_min": 192, "value_max": 192},
                            {"point_id": f"{series_id}_b", "value_min": 247, "value_max": 247},
                        ],
                    }
                    for series_id in ("series001", "series002")
                ],
            },
            stage3_document(),
        )

        self.assertNotIn("series_id", payload["properties"][0])
        self.assertEqual(repairs["aggregate_range_linked"], 0)

    def test_candidate_links_explicit_group_to_multiple_series(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [{
                    "property_id": "prop001",
                    "sample_id": "s001",
                    "property_name_raw": "$\\chi$",
                    "property_name_normalized": None,
                    "unit_raw": "unit",
                    "observation_group_id": "pog001",
                    "observation_role": "aggregate",
                    "series_id": None,
                    "evidence": [{"block_id": "P_1_0"}],
                    "confidence": add_model_confidence({
                        "properties": [{}],
                    })["properties"][0]["confidence"],
                }],
                "unresolved_properties": [],
                "property_series": [
                    {
                        "series_id": series_id,
                        "sample_id": "s001",
                        "entity_id": "pe001",
                        "property_name_raw": "$\\chi$",
                        "property_name_normalized": None,
                        "unit_raw": "unit",
                        "observation_group_id": "pog001",
                        "evidence": [{"block_id": block_id}],
                        "confidence": add_model_confidence({
                            "property_series": [{}],
                        })["property_series"][0]["confidence"],
                        "points": [],
                    }
                    for series_id, block_id in (
                        ("series001", "T_2_0"),
                        ("series002", "P_2_0"),
                    )
                ],
            },
            stage3_document(),
        )

        prop = payload["properties"][0]
        self.assertNotIn("series_id", prop)
        self.assertEqual(prop["series_ids"], ["series001", "series002"])
        self.assertEqual(repairs["aggregate_multi_linked"], 1)

    def test_candidate_does_not_cross_resolved_samples_in_group(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [{
                    "property_id": "prop001",
                    "sample_id": "s001",
                    "property_name_raw": "$\\chi$",
                    "property_name_normalized": None,
                    "unit_raw": "unit",
                    "observation_group_id": "pog001",
                    "observation_role": "aggregate",
                    "series_id": None,
                    "evidence": [{"block_id": "P_1_0"}],
                    "confidence": add_model_confidence({
                        "properties": [{}],
                    })["properties"][0]["confidence"],
                }],
                "unresolved_properties": [],
                "property_series": [
                    {
                        "series_id": series_id,
                        "sample_id": sample_id,
                        "entity_id": "pe001",
                        "property_name_raw": "$\\chi$",
                        "property_name_normalized": None,
                        "unit_raw": "unit",
                        "observation_group_id": "pog001",
                        "evidence": [{"block_id": "T_2_0"}],
                        "confidence": add_model_confidence({
                            "property_series": [{}],
                        })["property_series"][0]["confidence"],
                        "points": [],
                    }
                    for series_id, sample_id in (
                        ("series001", "s001"),
                        ("series002", "s002"),
                    )
                ],
            },
            stage3_document(),
        )

        prop = payload["properties"][0]
        self.assertEqual(prop["series_id"], "series001")
        self.assertNotIn("series_ids", prop)
        self.assertEqual(repairs["aggregate_linked"], 1)

    def test_series_entity_is_relinked_to_resolved_sample(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "series_id": "series001",
                    "sample_id": "s001",
                    "entity_id": "pe999",
                    "points": [{
                        "point_id": "pt001",
                        "sample_id": "s001",
                        "entity_id": "pe999",
                    }],
                }],
            },
            stage3_document(),
        )

        series = payload["property_series"][0]
        self.assertEqual(series["entity_id"], "pe001")
        self.assertEqual(series["points"][0]["entity_id"], "pe001")
        self.assertEqual(repairs["series_entity_relinked_to_sample"], 1)
        self.assertEqual(repairs["point_entity_relinked_to_sample"], 1)

    def test_series_entity_is_not_guessed_for_unknown_sample(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "series_id": "series001",
                    "sample_id": "s999",
                    "entity_id": "pe999",
                    "points": [],
                }],
            },
            stage3_document(),
        )

        self.assertEqual(
            payload["property_series"][0]["entity_id"],
            "pe999",
        )
        self.assertEqual(repairs["series_entity_relinked_to_sample"], 0)

    def test_multi_subject_series_clears_incorrect_top_level_subject(self) -> None:
        payload = SeriesClient().call_json("", "").data
        series = payload["property_series"][0]
        first_point = series["points"][0]
        first_point.update({"sample_id": "s001", "entity_id": "pe001"})
        second_point = json.loads(json.dumps(first_point))
        second_point.update({
            "point_id": "pt011",
            "sample_id": "s002",
            "entity_id": "pe002",
        })
        series["points"].append(second_point)

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
        )

        repaired_series = repaired["property_series"][0]
        self.assertIsNone(repaired_series["sample_id"])
        self.assertIsNone(repaired_series["entity_id"])
        self.assertEqual(
            repaired_series["sample_resolution_status"],
            "unresolved",
        )
        self.assertEqual(repairs["multi_subject_series_normalized"], 1)
        PropertyStageResponse.model_validate(repaired)

    def test_single_subject_series_inherits_subject_from_points(self) -> None:
        payload = SeriesClient().call_json("", "").data
        series = payload["property_series"][0]
        series.update({
            "sample_id": None,
            "entity_id": None,
            "sample_resolution_status": "unresolved",
        })
        series["points"][0].update({
            "sample_id": "s001",
            "entity_id": "pe001",
        })

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
        )

        repaired_series = repaired["property_series"][0]
        self.assertEqual(repaired_series["sample_id"], "s001")
        self.assertEqual(repaired_series["entity_id"], "pe001")
        self.assertEqual(
            repaired_series["sample_resolution_status"],
            "resolved",
        )
        self.assertEqual(
            repairs["single_subject_series_inherited_from_points"],
            1,
        )
        PropertyStageResponse.model_validate(repaired)

    def test_coordinate_series_can_remain_explicitly_unresolved(self) -> None:
        payload = SeriesClient().call_json("", "").data
        series = payload["property_series"][0]
        series.update({
            "sample_id": None,
            "entity_id": None,
            "sample_resolution_status": "unresolved",
        })
        for point in series["points"]:
            point.update({
                "sample_id": None,
                "entity_id": None,
                "sample_resolution_status": "unresolved",
            })

        parsed = PropertyStageResponse.model_validate(payload)

        self.assertEqual(
            parsed.property_series[0].sample_resolution_status,
            "unresolved",
        )

    def test_series_with_subjectless_point_remains_invalid(self) -> None:
        payload = SeriesClient().call_json("", "").data
        series = payload["property_series"][0]
        series.update({
            "sample_id": None,
            "entity_id": None,
            "sample_resolution_status": "unresolved",
        })
        first_point = series["points"][0]
        first_point.update({"sample_id": "s001", "entity_id": "pe001"})
        second_point = json.loads(json.dumps(first_point))
        second_point.update({
            "point_id": "pt011",
            "sample_id": None,
            "entity_id": None,
        })
        series["points"].append(second_point)

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
        )

        self.assertEqual(repairs["multi_subject_series_normalized"], 0)
        self.assertEqual(
            repairs["single_subject_series_inherited_from_points"],
            0,
        )
        with self.assertRaisesRegex(ValueError, "至少关联"):
            PropertyStageResponse.model_validate(repaired)

    def test_missing_series_confidence_uses_lowest_point_score(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "series_id": "series001",
                    "points": [
                        {"confidence": {"score": 0.9}},
                        {"confidence": {"score": 0.7}},
                    ],
                }],
            },
            stage3_document(),
        )

        self.assertEqual(
            payload["property_series"][0]["confidence"],
            {"score": 0.7},
        )
        self.assertEqual(
            repairs["series_confidence_inherited_from_points"],
            1,
        )

    def test_missing_point_score_does_not_invent_series_confidence(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "series_id": "series001",
                    "points": [
                        {"confidence": {"score": 0.9}},
                        {},
                    ],
                }],
            },
            stage3_document(),
        )

        self.assertNotIn("confidence", payload["property_series"][0])
        self.assertEqual(
            repairs["series_confidence_inherited_from_points"],
            0,
        )

    def test_existing_series_confidence_is_not_overwritten(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "series_id": "series001",
                    "confidence": {"score": 0.8},
                    "points": [
                        {"confidence": {"score": 0.6}},
                    ],
                }],
            },
            stage3_document(),
        )

        self.assertEqual(
            payload["property_series"][0]["confidence"],
            {"score": 0.8},
        )
        self.assertEqual(
            repairs["series_confidence_inherited_from_points"],
            0,
        )

    def test_invalid_property_name_maps_from_unique_code_category(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [{
                    "property_name_normalized": "glass transition temperature",
                    "property_code": "P3110",
                    "property_category": "thermal_property",
                }],
                "unresolved_properties": [],
                "property_series": [],
            },
            stage3_document(),
            vocabulary={
                "glass_transition_temperature": (
                    "P3110",
                    "thermal_property",
                ),
            },
        )

        self.assertEqual(
            payload["properties"][0]["property_name_normalized"],
            "glass_transition_temperature",
        )
        self.assertEqual(
            repairs["property_names_mapped_from_code_category"],
            1,
        )

    def test_ambiguous_code_category_does_not_map_property_name(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [{
                    "property_name_normalized": "thermal value",
                    "property_code": "P3140",
                    "property_category": "thermal_property",
                }],
                "unresolved_properties": [],
                "property_series": [],
            },
            stage3_document(),
            vocabulary={
                "crystallization_temperature": (
                    "P3140",
                    "thermal_property",
                ),
                "heat_of_crystallization": (
                    "P3140",
                    "thermal_property",
                ),
            },
        )

        self.assertEqual(
            payload["properties"][0]["property_name_normalized"],
            "thermal value",
        )
        self.assertEqual(
            repairs["property_names_mapped_from_code_category"],
            0,
        )

    def test_valid_property_name_is_not_overwritten_by_code(self) -> None:
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [{
                    "property_name_normalized": "melting_temperature",
                    "property_code": "P3110",
                    "property_category": "thermal_property",
                }],
                "unresolved_properties": [],
                "property_series": [],
            },
            stage3_document(),
            vocabulary={
                "glass_transition_temperature": (
                    "P3110",
                    "thermal_property",
                ),
                "melting_temperature": (
                    "P3120",
                    "thermal_property",
                ),
            },
        )

        self.assertEqual(
            payload["properties"][0]["property_name_normalized"],
            "melting_temperature",
        )
        self.assertEqual(
            repairs["property_names_mapped_from_code_category"],
            0,
        )

    def test_aggregate_series_reference_schema_rules(self) -> None:
        _validate_aggregate_series_reference(
            "aggregate", None, ["series001", "series002"], "test"
        )
        # 两种违规形态的修法不同，报错必须可区分：都没填要改用
        # unresolved property，都填了只需删掉一个。
        with self.assertRaisesRegex(ValueError, "必须填写 series_id 或 series_ids"):
            _validate_aggregate_series_reference("aggregate", None, None, "test")
        with self.assertRaisesRegex(ValueError, "unresolved property"):
            _validate_aggregate_series_reference("aggregate", None, None, "test")
        with self.assertRaisesRegex(ValueError, "不得同时填写"):
            _validate_aggregate_series_reference(
                "aggregate", "series001", ["series001", "series002"], "test"
            )
        with self.assertRaisesRegex(ValueError, "不得重复"):
            _validate_aggregate_series_reference(
                "aggregate", None, ["series001", "series001"], "test"
            )
        with self.assertRaisesRegex(ValueError, "不得引用 Series"):
            _validate_aggregate_series_reference(
                "single", None, ["series001", "series002"], "test"
            )

    def test_candidate_requires_compatible_series_relation(self) -> None:
        base_property = {
            "property_id": "prop001",
            "sample_id": "s001",
            "property_name_raw": "conductivity",
            "property_name_normalized": "electric_conductivity",
            "unit_raw": "S/cm",
            "observation_role": "aggregate",
            "series_id": None,
            "evidence": [{"block_id": "P_1_0"}],
            "confidence": add_model_confidence({
                "properties": [{}],
            })["properties"][0]["confidence"],
        }
        base_series = {
            "series_id": "series001",
            "sample_id": "s001",
            "entity_id": "pe001",
            "property_name_raw": "conductivity",
            "property_name_normalized": "electric_conductivity",
            "unit_raw": "S/cm",
            "evidence": [{"block_id": "P_1_0"}],
            "confidence": add_model_confidence({
                "property_series": [{}],
            })["property_series"][0]["confidence"],
            "points": [],
        }
        incompatible_updates = (
            {"sample_id": "s002"},
            {"unit_raw": "Pa"},
            {"evidence": [{"block_id": "T_2_0"}]},
        )

        for updates in incompatible_updates:
            with self.subTest(updates=updates):
                series = {**base_series, **updates}
                repaired, repairs = _repair_candidate_response_payload(
                    {
                        "properties": [base_property],
                        "unresolved_properties": [],
                        "property_series": [series],
                    },
                    stage3_document(),
                )

                self.assertIsNone(repaired["properties"][0]["series_id"])
                self.assertEqual(repairs["aggregate_linked"], 0)

    def test_candidate_repairs_locator_from_property_and_method(self) -> None:
        document = method_header_stage0_document()
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [{
                    "property_name_raw": "solubility parameter",
                    "determination_method_raw": "computed from [Q]",
                    "evidence": [{
                        "block_id": "T_2_0",
                        "table_locator": {
                            "table_id": "T_2_0",
                            "row_label": "corrupt row",
                            "column_label": "corrupt column",
                            "cell_value": "8.5 to 8.6 (cal/ml)^1/2",
                        },
                    }],
                }],
                "property_series": [],
            },
            stage3_document(),
            document.elements,
        )

        locator = payload["unresolved_properties"][0]["evidence"][0][
            "table_locator"
        ]
        self.assertEqual(locator["row_label"], "solubility parameter")
        self.assertEqual(locator["column_label"], "[Q]")
        self.assertEqual(repairs["table_locator_surfaces_repaired"], 1)

    def test_candidate_aligns_point_table_id_to_evidence_block(self) -> None:
        document = stage0_document()
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "property_name_raw": "Property",
                    "evidence": [],
                    "points": [{
                        "evidence": [{
                            "block_id": "T_2_0",
                            "table_locator": {
                                "table_id": "Table 1",
                                "row_index": 1,
                                "column_index": 2,
                            },
                        }],
                        "coordinates": [],
                    }],
                }],
            },
            stage3_document(),
            document.elements,
        )

        locator = payload["property_series"][0]["points"][0]["evidence"][
            0
        ]["table_locator"]
        self.assertEqual(locator["table_id"], "T_2_0")
        self.assertEqual(repairs["table_locator_ids_aligned_to_evidence"], 1)

    def test_candidate_does_not_align_table_id_for_non_table_block(self) -> None:
        document = stage0_document()
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "property_name_raw": "Property",
                    "evidence": [],
                    "points": [{
                        "evidence": [{
                            "block_id": "P_2_0",
                            "table_locator": {"table_id": "Table 1"},
                        }],
                        "coordinates": [],
                    }],
                }],
            },
            stage3_document(),
            document.elements,
        )

        locator = payload["property_series"][0]["points"][0]["evidence"][
            0
        ]["table_locator"]
        self.assertEqual(locator["table_id"], "Table 1")
        self.assertEqual(repairs["table_locator_ids_aligned_to_evidence"], 0)

    def test_candidate_repairs_pm_mp_context_to_exact_caption(self) -> None:
        document = stage0_document()
        document.elements[2].caption = (
            r"Table 1 at $35 \mp 0.01^{\circ}\mathrm{C}$."
        )
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "measurement_context": {
                        "temperature": {
                            "raw": r"35 \pm 0.01^{\circ}C",
                            "value": 35,
                            "unit": "°C",
                        },
                        "condition_status": "reported",
                    },
                    "evidence": [],
                    "points": [{
                        "measurement_context": None,
                        "evidence": [{"block_id": "T_2_0"}],
                        "coordinates": [],
                    }],
                }],
            },
            stage3_document(),
            document.elements,
        )

        raw = payload["property_series"][0]["measurement_context"][
            "temperature"
        ]["raw"]
        self.assertEqual(raw, r"35 \mp 0.01^{\circ}\mathrm{C}")
        self.assertEqual(
            repairs["measurement_context_surfaces_repaired"],
            1,
        )

    def test_candidate_supplements_exact_series_method_evidence(self) -> None:
        document = stage0_document()
        method = "independent viscosity method"
        document.elements[3].text = f"The {method} was used."
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "determination_method_raw": method,
                    "evidence": [],
                    "points": [],
                }],
            },
            stage3_document(),
            document.elements,
        )

        evidence = payload["property_series"][0]["evidence"]
        self.assertEqual([item["block_id"] for item in evidence], ["P_3_0"])
        self.assertEqual(repairs["series_method_evidence_supplemented"], 1)

    def test_candidate_prefers_unique_methods_block_for_series_method(
        self,
    ) -> None:
        document = stage0_document()
        document.elements[1].text = "These DSC results confirmed the trend."
        document.elements[3].text = "Thermal properties were measured by DSC."
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "determination_method_raw": "DSC",
                    "evidence": [],
                    "points": [],
                }],
            },
            stage3_document(),
            document.elements,
        )

        evidence = payload["property_series"][0]["evidence"]
        self.assertEqual([item["block_id"] for item in evidence], ["P_3_0"])
        self.assertEqual(repairs["series_method_evidence_supplemented"], 1)

    def test_candidate_does_not_choose_between_multiple_methods_blocks(
        self,
    ) -> None:
        document = stage0_document()
        document.elements[0].text = "Initial DSC measurements were performed."
        document.elements[3].text = "Thermal properties were measured by DSC."
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "determination_method_raw": "DSC",
                    "evidence": [],
                    "points": [],
                }],
            },
            stage3_document(),
            document.elements,
        )

        self.assertEqual(payload["property_series"][0]["evidence"], [])
        self.assertEqual(repairs["series_method_evidence_supplemented"], 0)

    def test_candidate_uses_series_header_for_repeated_missing_values(
        self,
    ) -> None:
        document = stage0_document()
        table_body = (
            "<table><tr><td>Sample</td><td>[Q]</td><td>[Phi]</td></tr>"
            "<tr><td>S1</td><td>-</td><td>-</td></tr></table>"
        )
        document.elements[2].table_body = table_body
        document.elements[2].table_cells = parse_table_cells(
            table_body,
            "T_2_0",
        )
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "property_name_raw": "[Phi]",
                    "evidence": [],
                    "points": [{
                        "evidence": [{
                            "block_id": "T_2_0",
                            "table_locator": {
                                "table_id": "T_2_0",
                                "row_label": "S1",
                                "column_label": "corrupt",
                                "cell_value": "-",
                            },
                        }],
                        "coordinates": [],
                    }],
                }],
            },
            stage3_document(),
            document.elements,
        )

        locator = payload["property_series"][0]["points"][0]["evidence"][
            0
        ]["table_locator"]
        self.assertEqual(locator["column_label"], "[Phi]")
        self.assertEqual(locator["cell_id"], "T_2_0:r0001:c0002")
        self.assertEqual(repairs["table_locator_surfaces_repaired"], 1)

    def test_candidate_supplements_common_series_context_evidence(self) -> None:
        document = stage0_document()
        payload, repairs = _repair_candidate_response_payload(
            {
                "properties": [],
                "unresolved_properties": [],
                "property_series": [{
                    "property_name_raw": "dried PB film",
                    "measurement_context": {
                        "temperature": {
                            "raw": "25 °C",
                            "value": 25,
                            "unit": "°C",
                        },
                        "other_conditions": {},
                        "condition_status": "reported",
                    },
                    "evidence": [],
                    "points": [{
                        "measurement_context": None,
                        "evidence": [{"block_id": "T_2_0"}],
                        "coordinates": [],
                    }],
                }],
            },
            stage3_document(),
            document.elements,
        )

        evidence = payload["property_series"][0]["evidence"]
        self.assertEqual([item["block_id"] for item in evidence], ["P_2_0"])
        self.assertEqual(repairs["series_context_evidence_supplemented"], 1)

    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary, cls.vocabulary_hash = load_property_vocabulary(
            DEFAULT_VOCABULARY_PATH
        )

    def test_vocabulary_has_97_authoritative_entries(self) -> None:
        self.assertEqual(len(self.vocabulary), 97)

    def test_surface_text_resolves_latex_formatting_to_source(self) -> None:
        source = (
            r"Table 1. Measurements at "
            r"$35 \mp 0.01^{\circ}\mathrm{C}$"
        )
        candidate = r"35 \mp 0.01^{\circ}C"

        self.assertEqual(
            _resolve_surface_text(source, candidate),
            r"35 \mp 0.01^{\circ}\mathrm{C}",
        )

    def test_surface_text_resolves_rendered_subscript_to_source(self) -> None:
        self.assertEqual(
            _resolve_surface_text(r"Results of $\delta_p$", "δp"),
            r"\delta_p",
        )

    def test_surface_text_resolves_split_math_unit_to_source(self) -> None:
        self.assertEqual(
            _resolve_surface_text(
                r"8.55 (cal/ml)  $^{1/2}$",
                r"(cal/ml)^{1/2}",
            ),
            r"(cal/ml)  $^{1/2}",
        )

    def test_surface_text_resolves_rendered_times_power_to_source(self) -> None:
        self.assertEqual(
            _resolve_surface_text(
                r"velocity $v = 1.64 \times 10^{5}$ cm/sec",
                "1.64 × 10^5",
            ),
            r"1.64 \times 10^{5}",
        )

    def test_surface_text_resolves_unicode_minus_power_to_source(self) -> None:
        self.assertEqual(
            _resolve_surface_text(
                r"$5.00 \times 10^{-4}$",
                "5.00 × 10−4",
            ),
            r"5.00 \times 10^{-4}",
        )

    def test_surface_text_resolves_escaped_latex_percent(self) -> None:
        self.assertEqual(
            _resolve_surface_text(r"$T_{3\%} (air)$", "T_{3%} (air)"),
            r"T_{3\%} (air)",
        )

    def test_surface_text_resolves_text_with_spaced_latex_temperature(self) -> None:
        source = (
            r"run from room temperature to "
            r"$8 0 0 ^ { \circ } \mathrm { C }$"
            r"at 10°C/min."
        )

        normalized = _resolve_surface_text(
            source,
            "room temperature to 800°C",
        )

        self.assertEqual(
            normalized,
            r"room temperature to $8 0 0 ^ { \circ } \mathrm { C }",
        )

    def test_surface_text_resolves_spaced_ocr_latex_range(self) -> None:
        source = (
            r"range of $4 . 1 \times 1 0 ^ { - 1 2 }$to "
            r"$9 . 4 \times 1 0 ^ { - 9 } ~ \Omega { \cdot } "
            r"\mathrm { c m } ^ { - 1 }$ depending"
        )
        candidate = "4.1 × 10^-12 to 9.4 × 10^-9 Ω·cm^-1"

        self.assertEqual(
            _resolve_surface_text(source, candidate),
            (
                r"4 . 1 \times 1 0 ^ { - 1 2 }$to $9 . 4 \times "
                r"1 0 ^ { - 9 } ~ \Omega { \cdot } "
                r"\mathrm { c m } ^ { - 1 }"
            ),
        )

    def test_surface_text_resolves_rendered_mn_star_to_source(self) -> None:
        self.assertEqual(
            _resolve_surface_text(r"header $Mn^*$ value", "Mn*"),
            r"Mn^*",
        )

    def test_surface_text_resolves_contextual_ocr_degree_mark(self) -> None:
        self.assertEqual(
            _resolve_surface_text("the second stage starts from 380～C", "380°C"),
            "380～C",
        )

    def test_surface_text_does_not_treat_range_tilde_as_degree_mark(self) -> None:
        self.assertIsNone(
            _resolve_surface_text("range 200～300 C", "200°300 C")
        )

    def test_surface_text_resolves_beta_subscript(self) -> None:
        resolved = _resolve_surface_text(
            r"values of $\beta_{T}$ were estimated",
            "βT",
        )

        self.assertEqual(resolved, r"\beta_{T}")

    def test_surface_text_resolves_short_math_subscript(self) -> None:
        resolved = _resolve_surface_text(
            r"a value of $K_{t}$ was used",
            "Kt",
        )

        self.assertEqual(resolved, r"K_{t}")

    def test_surface_text_resolves_unicode_superscript(self) -> None:
        resolved = _resolve_surface_text(
            r"a value in m $^{2}$ /N",
            "m²/N",
        )

        self.assertEqual(resolved, r"m $^{2}$ /N")

    def test_grouped_table_columns_recover_unresolved_methods(self) -> None:
        items = [
            UnresolvedPropertyObservation.model_validate({
                "unresolved_id": f"uprop{index:03d}",
                "entity_id": "pe001",
                "property_name_raw": r"\delta_p",
                "observation_group_id": "pog001",
                "value_raw": value,
                "reason": "sample_ambiguous",
                "evidence": [{
                    "block_id": "T_1_0",
                    "page": 1,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "table",
                    "source_sentence": "Table 1",
                    "table_locator": {
                        "table_id": "T_1_0",
                        "row_label": r"\delta_p",
                        "column_label": column,
                        "cell_value": value,
                    },
                }],
                "confidence": {"score": 0.8},
            })
            for index, (column, value) in enumerate(
                ((r"$[\eta]$", "8.55"), ("[Q]", "8.60")),
                start=1,
            )
        ]

        recovered_items, recovered = _recover_grouped_table_methods(items)

        self.assertEqual(
            [item.determination_method_raw for item in recovered_items],
            [r"$[\eta]$", "[Q]"],
        )
        self.assertEqual(len(recovered), 2)
        self.assertEqual(recovered_items[0].confidence.score, 0.5)

    def test_generic_table_columns_do_not_recover_methods(self) -> None:
        items = [
            UnresolvedPropertyObservation.model_validate({
                "unresolved_id": f"uprop{index:03d}",
                "entity_id": "pe001",
                "property_name_raw": "modulus",
                "observation_group_id": "pog001",
                "value_raw": value,
                "reason": "sample_ambiguous",
                "evidence": [{
                    "block_id": "T_1_0",
                    "page": 1,
                    "bbox": [1, 2, 3, 4],
                    "source_type": "table",
                    "source_sentence": "Table 1",
                    "table_locator": {
                        "table_id": "T_1_0",
                        "row_label": "modulus",
                        "column_label": column,
                        "cell_value": value,
                    },
                }],
            })
            for index, (column, value) in enumerate(
                (("Value", "1.0"), ("Result", "2.0")),
                start=1,
            )
        ]

        recovered_items, recovered = _recover_grouped_table_methods(items)

        self.assertFalse(recovered)
        self.assertTrue(all(
            item.determination_method_raw is None
            for item in recovered_items
        ))
        self.assertEqual(
            self.vocabulary["solubility_parameter"],
            ("P5110", "physicochemical_property"),
        )

    def test_extracts_property_condition_and_table_evidence(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            FakeClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
        )

        self.assertEqual(result.measurement_conditions[0].condition_id, "mc001")
        temperature = result.measurement_conditions[0].temperature
        self.assertIsNotNone(temperature)
        assert temperature is not None
        self.assertEqual(len(temperature.evidence), 1)
        self.assertIn("25 °C", temperature.evidence[0].source_sentence)
        self.assertEqual(result.properties[0].property_id, "prop001")
        self.assertEqual(
            result.properties[0].measurement_condition_id,
            "mc001",
        )
        self.assertEqual(len(result.properties[0].evidence), 2)
        self.assertEqual(
            result.properties[0].evidence[1].table_locator["row_label"],
            "dried PB film",
        )
        self.assertEqual(
            result.properties[0].evidence[1].table_locator["cell_id"],
            "T_2_0:r0001:c0002",
        )
        self.assertEqual(
            result.properties[0].evidence[1].table_locator["row_index"],
            1,
        )
        self.assertEqual(
            result.properties[0].evidence[1].table_locator["column_index"],
            2,
        )
        self.assertEqual(result.provenance.model, "fake-actual")

    def test_extracts_property_series_with_coordinates_and_coverage(
        self,
    ) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            SeriesClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
        )

        self.assertEqual(len(result.property_series), 1)
        series = result.property_series[0]
        self.assertEqual(series.series_id, "series001")
        self.assertEqual(series.coverage.covered, 1)
        self.assertEqual(series.coverage.missing, 0)
        self.assertEqual(series.coverage.ratio, 1.0)
        self.assertEqual(series.points[0].point_id, "pt001")
        self.assertEqual(
            series.points[0].evidence[0].table_locator["cell_id"],
            "T_2_0:r0001:c0002",
        )
        self.assertEqual(
            series.points[0].coordinates[0].evidence.table_locator["cell_id"],
            "T_2_0:r0001:c0000",
        )
        self.assertEqual(
            series.points[0].measurement_context.condition_status,
            "not_reported",
        )
        self.assertTrue(series.points[0].confidence)

    def test_coordinate_only_property_column_is_accepted_with_warning(
        self,
    ) -> None:
        result = extract_properties(
            tg_mn_table_document(),
            stage2_document(),
            stage3_document(),
            CoordinateOnlyMnClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        warning = next(
            item
            for item in result.warnings
            if item["code"]
            == "table_property_column_represented_as_coordinate"
        )
        self.assertEqual(warning["columns"], [{
            "table_id": "T_4_0",
            "column_index": 2,
            "column_label": "Mn",
            "value_count": 2,
            "coordinate_cell_count": 2,
            "unrepresented_cell_count": 0,
        }])

    def test_coordinate_column_warning_reports_partial_representation(
        self,
    ) -> None:
        result = extract_properties(
            tg_mn_table_document(),
            stage2_document(),
            stage3_document(),
            PartiallyRepresentedMnClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        warning = next(
            item
            for item in result.warnings
            if item["code"]
            == "table_property_column_represented_as_coordinate"
        )
        self.assertEqual(warning["columns"][0]["value_count"], 2)
        self.assertEqual(warning["columns"][0]["coordinate_cell_count"], 1)
        self.assertEqual(warning["columns"][0]["unrepresented_cell_count"], 1)

    def test_property_series_cannot_silently_skip_table_rows(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            TABLE_BODY
            + "\n| second PB film | solubility parameter | "
            "9.1 (cal/ml)^1/2 |"
        )

        with self.assertRaisesRegex(Stage4Error, "未覆盖同一表格行/列"):
            extract_properties(
                document,
                stage2_document(),
                stage3_document(),
                SeriesClient(),
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_property_series_coverage_stays_within_material_group(self) -> None:
        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Sample | Property | Value |\n"
            "| --- | --- | --- |\n"
            "| Polybutadiene | | |\n"
            "| dried PB film | solubility parameter | "
            "8.5 to 8.6 (cal/ml)^1/2 |\n"
            "| Polystyrene | | |\n"
            "| PS film | solubility parameter | 9.1 (cal/ml)^1/2 |"
        )
        table.table_cells = None

        result = extract_properties(
            document,
            stage2_document(),
            stage3_document(),
            SeriesClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertEqual(result.property_series[0].coverage.covered, 1)

    def test_coverage_stops_when_blank_label_group_reaches_new_material(self) -> None:
        from schema.polymer_schema import PropertySeriesPointCandidate
        from stages.stage4_property import _validate_series_table_coverage

        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Polymer | Tg/K |\n| --- | --- |\n"
            "| Polystyrene | |\n| | 340 |\n| Polybutadiene | 192 |"
        )
        table.table_cells = None
        point = PropertySeriesPointCandidate.model_validate({
                "point_id": "pt001",
                "observation_role": "series_point",
                "coordinates": [],
                "value_raw": "340",
                "value_min": 340,
                "value_max": 340,
                "coverage_status": "covered",
                "evidence": [{
                    "block_id": table.block_id,
                    "source_sentence": "340",
                    "table_locator": {
                        "table_id": table.block_id,
                        "row_label": "Polystyrene",
                        "column_label": "Tg/K",
                        "cell_value": "340",
                        "cell_id": f"{table.block_id}:r0002:c0001",
                        "row_index": 2,
                        "column_index": 1,
                    },
                }],
                "confidence": {"score": 0.8},
            })

        _validate_series_table_coverage(
            "series001",
            [point],
            {table.block_id: table},
        )

    def test_coverage_excludes_explicit_other_property_assignment(self) -> None:
        from schema.polymer_schema import PropertySeriesPointCandidate
        from stages.stage4_property import _validate_series_table_coverage

        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Alkyl | To, °C (dilatometer) |\n| --- | --- |\n"
            "| Dodecyl | -55 |\n| Tetradecyl | -72 |\n"
            "| Hexadecyl | Tm = 22 |"
        )
        table.table_cells = None

        def point(point_id: str, row: int, label: str, value: str):
            return PropertySeriesPointCandidate.model_validate({
                "point_id": point_id,
                "coordinates": [],
                "value_raw": value,
                "value_min": float(value),
                "value_max": float(value),
                "coverage_status": "covered",
                "evidence": [{
                    "block_id": table.block_id,
                    "source_sentence": value,
                    "table_locator": {
                        "table_id": table.block_id,
                        "row_label": label,
                        "column_label": "To, °C (dilatometer)",
                        "cell_value": value,
                        "cell_id": f"{table.block_id}:r{row:04d}:c0001",
                        "row_index": row,
                        "column_index": 1,
                    },
                }],
                "confidence": add_model_confidence({
                    "property_series": [{"points": [{}]}],
                })["property_series"][0]["points"][0]["confidence"],
            })

        _validate_series_table_coverage(
            "series001",
            [
                point("pt001", 1, "Dodecyl", "-55"),
                point("pt002", 2, "Tetradecyl", "-72"),
            ],
            {table.block_id: table},
        )

        table.table_body = table.table_body.replace("Tm = 22", "22")
        table.table_cells = None
        with self.assertRaisesRegex(ValueError, "未覆盖同一表格"):
            _validate_series_table_coverage(
                "series001",
                [
                    point("pt001", 1, "Dodecyl", "-55"),
                    point("pt002", 2, "Tetradecyl", "-72"),
                ],
                {table.block_id: table},
            )

    def test_coverage_ignores_nonnumeric_ocr_placeholder(self) -> None:
        from schema.polymer_schema import PropertySeriesPointCandidate
        from stages.stage4_property import _validate_series_table_coverage

        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Alkyl | Tg, °C |\n| --- | --- |\n"
            "| Octyl | -80 |\n| Decyl | бк |\n| Dodecyl | -72 |"
        )
        table.table_cells = None

        def point(point_id: str, row: int, label: str, value: str):
            return PropertySeriesPointCandidate.model_validate({
                "point_id": point_id,
                "observation_role": "series_point",
                "coordinates": [],
                "value_raw": value,
                "value_min": float(value),
                "value_max": float(value),
                "coverage_status": "covered",
                "evidence": [{
                    "block_id": table.block_id,
                    "source_sentence": value,
                    "table_locator": {
                        "table_id": table.block_id,
                        "row_label": label,
                        "column_label": "Tg, °C",
                        "cell_value": value,
                        "cell_id": f"{table.block_id}:r{row:04d}:c0001",
                        "row_index": row,
                        "column_index": 1,
                    },
                }],
                "confidence": add_model_confidence({
                    "property_series": [{"points": [{}]}],
                })["property_series"][0]["points"][0]["confidence"],
            })

        _validate_series_table_coverage(
            "series001",
            [
                point("pt001", 1, "Octyl", "-80"),
                point("pt002", 3, "Dodecyl", "-72"),
            ],
            {table.block_id: table},
        )

    def test_coverage_starts_at_first_labeled_material_row(self) -> None:
        from schema.polymer_schema import PropertySeriesPointCandidate
        from stages.stage4_property import _validate_series_table_coverage

        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Polymer | Tg/K |\n| --- | --- |\n"
            "| Polystyrene | 340 |\n| | 349 |\n"
            "| Polybutadiene | 192 |\n| | 191 |"
        )
        table.table_cells = None

        def point(point_id: str, row: int, value: str):
            return PropertySeriesPointCandidate.model_validate({
                "point_id": point_id,
                "observation_role": "series_point",
                "coordinates": [],
                "value_raw": value,
                "coverage_status": "covered",
                "evidence": [{
                    "block_id": table.block_id,
                    "source_sentence": value,
                    "table_locator": {
                        "table_id": table.block_id,
                        "row_label": "Polybutadiene",
                        "column_label": "Tg/K",
                        "cell_value": value,
                        "cell_id": f"{table.block_id}:r{row:04d}:c0001",
                        "row_index": row,
                        "column_index": 1,
                    },
                }],
                "confidence": {"score": 0.8},
            })

        _validate_series_table_coverage(
            "series001",
            [point("pt001", 3, "192"), point("pt002", 4, "191")],
            {table.block_id: table},
        )

    def test_coverage_stops_before_scalar_row_with_own_unit(self) -> None:
        from schema.polymer_schema import PropertySeriesPointCandidate
        from stages.stage4_property import _validate_series_table_coverage

        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| liquid | state | poly(MPC) |\n| --- | --- | --- |\n"
            "| Water | static | 3 |\n| Air | static | 170 |\n"
            "| gamma | (mN/m) | 73 |"
        )
        table.table_cells = None

        def point(point_id: str, row: int, label: str, value: str):
            return PropertySeriesPointCandidate.model_validate({
                "point_id": point_id,
                "coordinates": [],
                "value_raw": value,
                "coverage_status": "covered",
                "evidence": [{
                    "block_id": table.block_id,
                    "source_sentence": f"{label} {value}",
                    "table_locator": {
                        "table_id": table.block_id,
                        "row_label": label,
                        "column_label": "poly(MPC)",
                        "cell_value": value,
                        "cell_id": f"{table.block_id}:r{row:04d}:c0002",
                        "row_index": row,
                        "column_index": 2,
                    },
                }],
                "confidence": {"score": 0.8},
            })

        _validate_series_table_coverage(
            "series001",
            [
                point("pt001", 1, "Water", "3"),
                point("pt002", 2, "Air", "170"),
            ],
            {table.block_id: table},
        )

    def test_preview_allows_missing_table_property_column(self) -> None:
        from stages.stage4_property import _validate_required_table_series
        from stages.table_grid import table_cells_for

        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Polymer | Mw (g/mol) |\n| --- | --- |\n"
            "| A | 1000 |\n| B | 2000 |"
        )
        table.table_cells = None
        table.table_cells = table_cells_for(table)

        gaps = _validate_required_table_series(
            document.elements,
            [],
            allow_missing=True,
        )

        self.assertEqual(gaps[0]["representation"], "missing")
        with self.assertRaisesRegex(ValueError, "未作为 PropertySeries"):
            _validate_required_table_series(document.elements, [])

    def test_coverage_uses_common_numbered_label_prefix(self) -> None:
        from schema.polymer_schema import PropertySeriesPointCandidate
        from stages.stage4_property import _validate_series_table_coverage

        document = stage0_document().model_copy(deep=True)
        table = next(item for item in document.elements if item.type == "table")
        table.table_body = (
            "| Polymer | Tg/K |\n| --- | --- |\n"
            "| DTE-2 | 207 |\n| DTE-3 | 208 |\n| Squalene | 176 |"
        )
        table.table_cells = None

        def point(point_id: str, row: int, label: str, value: str):
            return PropertySeriesPointCandidate.model_validate({
                "point_id": point_id,
                "observation_role": "series_point",
                "coordinates": [],
                "value_raw": value,
                "coverage_status": "covered",
                "evidence": [{
                    "block_id": table.block_id,
                    "source_sentence": f"{label} {value}",
                    "table_locator": {
                        "table_id": table.block_id,
                        "row_label": label,
                        "column_label": "Tg/K",
                        "cell_value": value,
                        "cell_id": f"{table.block_id}:r{row:04d}:c0001",
                        "row_index": row,
                        "column_index": 1,
                    },
                }],
                "confidence": {"score": 0.8},
            })

        _validate_series_table_coverage(
            "series001",
            [
                point("pt001", 1, "DTE-2", "207"),
                point("pt002", 2, "DTE-3", "208"),
            ],
            {table.block_id: table},
        )

    def test_unknown_sample_is_retried(self) -> None:
        client = RetryClient()

        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            client,
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=1,
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual(len(result.properties), 1)

    def test_mismatched_controlled_vocabulary_is_rejected(self) -> None:
        with self.assertRaises(Stage4Error):
            extract_properties(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                InvalidVocabularyClient(),
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_numeric_resolved_property_name_is_rejected(self) -> None:
        with self.assertRaises(Stage4Error):
            extract_properties(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                NumericPropertyNameClient(),
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_unresolved_property_is_preserved_with_warning(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            UnresolvedClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
        )

        self.assertEqual(result.unresolved_properties[0].unresolved_id, "uprop001")
        self.assertEqual(result.warnings[0]["code"], "unresolved_properties")

    def test_unresolved_confidence_can_reference_null_sample_id(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            UnresolvedSampleConfidenceClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
        )

        self.assertIsNone(result.unresolved_properties[0].sample_id)
        self.assertEqual(
            result.unresolved_properties[0].confidence.model_dump(),
            {"score": 0.9},
        )

    def test_unresolved_method_and_group_are_preserved(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            UnresolvedMethodClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
        )

        unresolved = result.unresolved_properties[0]
        self.assertEqual(unresolved.determination_method_raw, "measured")
        self.assertEqual(unresolved.observation_group_id, "pog001")

    def test_paraphrased_method_recovers_exact_non_generic_column_header(
        self,
    ) -> None:
        result = extract_properties(
            method_header_stage0_document(),
            stage2_document(),
            stage3_document(),
            ParaphrasedTableMethodClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
        )

        unresolved = result.unresolved_properties[0]
        self.assertEqual(unresolved.determination_method_raw, "[Q]")
        self.assertEqual(unresolved.confidence.score, 0.5)

    def test_paraphrased_method_does_not_recover_generic_column_header(
        self,
    ) -> None:
        with self.assertRaises(Stage4Error):
            extract_properties(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                ParaphrasedGenericTableMethodClient(),
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_unresolved_numeric_property_name_is_rejected(self) -> None:
        with self.assertRaises(Stage4Error):
            extract_properties(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                UnresolvedNumericNameClient(),
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_unresolved_invalid_table_locator_is_repaired(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            UnresolvedInvalidTableLocatorClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
        )

        evidence = result.unresolved_properties[0].evidence[0]
        self.assertEqual(evidence.source_type, "table")
        self.assertEqual(
            evidence.table_locator["cell_id"],
            "T_2_0:r0001:c0002",
        )
        self.assertIn(
            "candidate_table_locator_surface_repaired",
            [warning["code"] for warning in result.warnings],
        )

    def test_multi_method_values_are_split_and_grouped(self) -> None:
        result = extract_properties(
            multi_method_stage0_document(),
            stage2_document(),
            stage3_document(),
            MultiMethodClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
        )

        self.assertEqual(len(result.properties), 3)
        self.assertEqual(
            [item.determination_method_raw for item in result.properties],
            ["viscometry", "turbidimetry", "swelling measurements"],
        )
        self.assertEqual(
            {item.observation_group_id for item in result.properties},
            {"pog001"},
        )

    def test_determination_method_must_come_from_evidence(self) -> None:
        with self.assertRaises(Stage4Error):
            extract_properties(
                multi_method_stage0_document(),
                stage2_document(),
                stage3_document(),
                InventedDeterminationMethodClient(),
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_preview_preserves_schema_valid_semantic_mismatch(self) -> None:
        result = extract_properties(
            multi_method_stage0_document(),
            stage2_document(),
            stage3_document(),
            InventedDeterminationMethodClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(
            result.properties[0].determination_method_raw,
            "osmometry",
        )
        self.assertTrue(any(
            item["code"] == "preview_semantic_validation_bypassed"
            for item in result.warnings
        ))

    def test_preview_uses_degraded_shell_for_invalid_schema(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            MissingRequiredFieldClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(result.measurement_conditions, [])
        self.assertEqual(result.properties, [])
        warning = next(
            item
            for item in result.warnings
            if item["code"] == "preview_degraded_empty_shell"
        )
        self.assertTrue(warning["degraded"])

    def test_strict_rejects_invalid_schema(self) -> None:
        with self.assertRaises(Stage4Error):
            extract_properties(
                stage0_document(),
                stage2_document(),
                stage3_document(),
                MissingRequiredFieldClient(),
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
                max_validation_retries=0,
            )

    def test_preview_salvages_property_with_one_valid_evidence(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            UnknownEvidenceBlockClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(len(result.properties), 1)
        self.assertEqual(
            [item.block_id for item in result.properties[0].evidence],
            ["T_2_0"],
        )
        salvage_warning = next(
            item for item in result.warnings
            if item["code"] == "preview_objects_salvaged"
        )
        self.assertEqual(salvage_warning["details"]["dropped_evidence"], 1)
        self.assertNotIn(
            "preview_degraded_empty_shell",
            [item["code"] for item in result.warnings],
        )

    def test_determination_method_can_use_separate_exact_evidence(self) -> None:
        result = extract_properties(
            separate_method_stage0_document(),
            stage2_document(),
            stage3_document(),
            SeparateDeterminationMethodClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
        )

        prop = result.properties[0]
        self.assertEqual(
            prop.determination_method_raw,
            "independent viscosity method",
        )
        self.assertIn("P_3_0", [item.block_id for item in prop.evidence])
        self.assertIn(
            "supplemented_property_evidence",
            [warning["code"] for warning in result.warnings],
        )

    def test_missing_additional_table_locator_is_dropped(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            MissingAdditionalTableLocatorClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertEqual(len(result.properties[0].evidence), 1)
        self.assertEqual(
            result.warnings[0]["code"],
            "dropped_table_evidence",
        )

    def test_only_table_evidence_can_degrade_without_locator(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            OnlyMissingTableLocatorClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertEqual(len(result.properties[0].evidence), 1)
        self.assertIsNone(result.properties[0].evidence[0].table_locator)
        self.assertEqual(
            result.warnings[0]["code"],
            "table_evidence_without_locator",
        )

    def test_invalid_table_locator_is_repaired_from_property_raw(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            OnlyInvalidTableLocatorClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertEqual(len(result.properties[0].evidence), 1)
        self.assertEqual(
            result.properties[0].evidence[0].table_locator["cell_id"],
            "T_2_0:r0001:c0002",
        )
        self.assertIn(
            "candidate_table_locator_surface_repaired",
            [warning["code"] for warning in result.warnings],
        )

    def test_unused_condition_is_dropped_with_warning(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            UnusedConditionClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertEqual(len(result.measurement_conditions), 1)
        self.assertEqual(
            result.warnings[0]["code"],
            "dropped_unused_conditions",
        )
        self.assertEqual(
            result.warnings[0]["candidate_condition_ids"],
            ["mc020"],
        )

    def test_unanchored_additional_evidence_is_dropped(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            UnanchoredAdditionalEvidenceClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertEqual(len(result.properties[0].evidence), 2)
        self.assertEqual(
            result.warnings[0]["code"],
            "dropped_unanchored_evidence",
        )

    def test_table_condition_can_degrade_without_locator(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            TableConditionWithoutLocatorClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        condition = result.measurement_conditions[0]
        self.assertEqual(condition.evidence.source_type, "table")
        self.assertIsNone(condition.evidence.table_locator)
        self.assertEqual(
            result.warnings[0]["code"],
            "table_condition_evidence_without_locator",
        )

    def test_property_evidence_can_be_supplemented_from_exact_text(self) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            WrongPropertyEvidenceClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertEqual(
            result.properties[0].evidence[0].block_id,
            "P_2_0",
        )
        self.assertIn(
            "supplemented_property_evidence",
            [warning["code"] for warning in result.warnings],
        )

    def test_condition_evidence_can_be_supplemented_from_linked_property(
        self,
    ) -> None:
        result = extract_properties(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            WrongConditionEvidenceClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertEqual(
            result.measurement_conditions[0].evidence.block_id,
            "P_2_0",
        )
        self.assertIn(
            "supplemented_condition_evidence",
            [warning["code"] for warning in result.warnings],
        )

    def test_cross_block_condition_can_use_determination_method_anchor(
        self,
    ) -> None:
        document = stage0_document()
        document.elements[1].text = (
            "The solubility parameter of dried PB film was "
            "8.5 to 8.6 (cal/ml)^1/2."
        )
        result = extract_properties(
            document,
            stage2_document(),
            stage3_document(),
            CrossBlockConditionClient(),
            rendered_prompt(),
            self.vocabulary,
            self.vocabulary_hash,
            max_validation_retries=0,
        )

        self.assertEqual(
            result.measurement_conditions[0].evidence.block_id,
            "P_1_0",
        )
        self.assertIn(
            "supplemented_condition_evidence",
            [warning["code"] for warning in result.warnings],
        )

    def test_compatible_output_cache_is_reused(self) -> None:
        document = stage0_document()
        entities = stage2_document()
        process = stage3_document()
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage2_path = root / "stage2_entities.json"
            stage3_path = root / "stage3_process.json"
            output_path = root / "stage4_properties.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage2_path.write_text(
                json.dumps(entities.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage3_path.write_text(
                json.dumps(process.model_dump(mode="json")),
                encoding="utf-8",
            )

            _, first_cached = run_stage4(
                stage0_path,
                stage2_path,
                stage3_path,
                output_path,
                client,
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
            )
            calls_after_first = client.calls
            _, second_cached = run_stage4(
                stage0_path,
                stage2_path,
                stage3_path,
                output_path,
                client,
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
            )

            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertEqual(client.calls, calls_after_first)

    def test_run_stage4_persists_raw_response_without_request_data(self) -> None:
        document = stage0_document()
        entities = stage2_document()
        process = stage3_document()
        client = FakeClient()
        client.last_raw_response = LLMRawResponse(
            provider="test",
            model="fake-actual",
            finish_reason="stop",
            content='{"properties": []}',
            usage=LLMTokenUsage(input_tokens=12, output_tokens=34),
            cost=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage2_path = root / "stage2_entities.json"
            stage3_path = root / "stage3_process.json"
            output_path = root / "stage4_properties.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage2_path.write_text(
                json.dumps(entities.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage3_path.write_text(
                json.dumps(process.model_dump(mode="json")),
                encoding="utf-8",
            )

            run_stage4(
                stage0_path,
                stage2_path,
                stage3_path,
                output_path,
                client,
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
                force=True,
                preview_relaxed=True,
            )

            artifact_path = root / "stage4_llm_response.json"
            self.assertTrue(artifact_path.is_file())
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["status"], "received")
            self.assertEqual(
                artifact["raw_response"]["content"],
                '{"properties": []}',
            )
            serialized = json.dumps(artifact).casefold()
            self.assertNotIn("api_key", serialized)
            self.assertNotIn("authorization", serialized)
            self.assertNotIn("system_prompt", serialized)
            self.assertNotIn("user_message", serialized)

    def test_stage4_172_cache_is_not_reused_after_locator_tightening(
        self,
    ) -> None:
        """1.7.8 收紧占位单元格锚点后，1.7.2 缓存必须重算。

        旧版本可能已把条件绑定到纯标点的占位格，其 table_locator
        与新版本不一致，直接复用会让错误锚点静默存活。
        """
        document = stage0_document()
        entities = stage2_document()
        process = stage3_document()
        client = FakeClient()
        prompt = rendered_prompt()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage2_path = root / "stage2_entities.json"
            stage3_path = root / "stage3_process.json"
            output_path = root / "stage4_properties.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage2_path.write_text(
                json.dumps(entities.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage3_path.write_text(
                json.dumps(process.model_dump(mode="json")),
                encoding="utf-8",
            )
            run_stage4(
                stage0_path,
                stage2_path,
                stage3_path,
                output_path,
                client,
                prompt,
                self.vocabulary,
                self.vocabulary_hash,
            )
            _, _, compatible_cache_key = _cache_components(
                document,
                entities,
                process,
                prompt,
                self.vocabulary_hash,
                client,
                implementation_version="1.7.2",
            )
            cached_payload = json.loads(output_path.read_text(encoding="utf-8"))
            cached_payload["provenance"]["implementation_version"] = "1.7.2"
            cached_payload["provenance"]["cache_key"] = compatible_cache_key
            output_path.write_text(
                json.dumps(cached_payload),
                encoding="utf-8",
            )
            calls_before = client.calls

            _, cached = run_stage4(
                stage0_path,
                stage2_path,
                stage3_path,
                output_path,
                client,
                prompt,
                self.vocabulary,
                self.vocabulary_hash,
            )

            self.assertFalse(cached)
            self.assertEqual(client.calls, calls_before + 1)

    def test_failure_response_can_be_replayed_without_network(self) -> None:
        document = stage0_document()
        entities = stage2_document()
        process = stage3_document()
        response = FakeClient().call_json("", "")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage2_path = root / "stage2_entities.json"
            stage3_path = root / "stage3_process.json"
            output_path = root / "stage4_properties.json"
            failure_path = root / "stage4_failure.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage2_path.write_text(
                json.dumps(entities.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage3_path.write_text(
                json.dumps(process.model_dump(mode="json")),
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
                            "input_per_million": "13.51",
                            "output_per_million": "66.5",
                            "input_cost": "0.001351",
                            "output_cost": "0.003325",
                            "total_cost": "0.004676",
                        },
                    }
                }),
                encoding="utf-8",
            )
            client = _failure_replay_client(
                failure_path,
                load_pipeline_config(DEFAULT_CONFIG_PATH),
            )

            run_stage4(
                stage0_path,
                stage2_path,
                stage3_path,
                output_path,
                client,
                rendered_prompt(),
                self.vocabulary,
                self.vocabulary_hash,
                force=True,
                max_validation_retries=0,
            )

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(client.calls, 1)
            self.assertEqual(result["provenance"]["model"], "saved-model")
            self.assertEqual(result["provenance"]["usage"]["input_tokens"], 100)
            self.assertEqual(result["provenance"]["cost"]["total_cost"], "0.004676")
            self.assertTrue(any(
                item["code"] == "failure_response_replayed"
                for item in result["warnings"]
            ))

    def test_failure_replay_requires_saved_raw_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            failure_path = Path(temp_dir) / "stage4_failure.json"
            failure_path.write_text(
                json.dumps({"raw_response": None}),
                encoding="utf-8",
            )

            with self.assertRaises(Stage4Error):
                _failure_replay_client(
                    failure_path,
                    load_pipeline_config(DEFAULT_CONFIG_PATH),
                )

    def test_failure_replay_accepts_markdown_fenced_content(self) -> None:
        """带 markdown 围栏的 raw response 必须可回放。

        回归保护：曾经 _failure_replay_client 直接 json.loads(content)，
        围栏响应会报 "不是完整 JSON"——与真实截断的报错完全相同，
        导致失败归因把"围栏"误判成"截断"。
        """
        payload = {"properties": [], "property_series": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            failure_path = Path(temp_dir) / "stage4_failure.json"
            failure_path.write_text(
                json.dumps({
                    "raw_response": {
                        "provider": "saved-provider",
                        "model": "saved-model",
                        "content": (
                            "```json\n"
                            + json.dumps(payload)
                            + "\n```"
                        ),
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                    }
                }),
                encoding="utf-8",
            )

            client = _failure_replay_client(
                failure_path,
                load_pipeline_config(DEFAULT_CONFIG_PATH),
            )
            self.assertEqual(client.response.data, payload)

    def test_failure_replay_rejects_truncated_content(self) -> None:
        """真实截断仍必须失败，不能被围栏兜底逻辑吞掉。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            failure_path = Path(temp_dir) / "stage4_failure.json"
            failure_path.write_text(
                json.dumps({
                    "raw_response": {
                        "content": '{"properties": [{"property_ty',
                    }
                }),
                encoding="utf-8",
            )

            with self.assertRaises(Stage4Error):
                _failure_replay_client(
                    failure_path,
                    load_pipeline_config(DEFAULT_CONFIG_PATH),
                )

    def test_stage4_legacy_implementation_cache_is_not_reused(self) -> None:
        document = stage0_document()
        entities = stage2_document()
        process = stage3_document()
        client = FakeClient()
        prompt = rendered_prompt()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage2_path = root / "stage2_entities.json"
            stage3_path = root / "stage3_process.json"
            output_path = root / "stage4_properties.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage2_path.write_text(
                json.dumps(entities.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage3_path.write_text(
                json.dumps(process.model_dump(mode="json")),
                encoding="utf-8",
            )
            run_stage4(
                stage0_path,
                stage2_path,
                stage3_path,
                output_path,
                client,
                prompt,
                self.vocabulary,
                self.vocabulary_hash,
            )
            _, _, old_cache_key = _cache_components(
                document,
                entities,
                process,
                prompt,
                self.vocabulary_hash,
                client,
                implementation_version="1.6.6",
            )
            cached_payload = json.loads(output_path.read_text(encoding="utf-8"))
            cached_payload["provenance"]["implementation_version"] = "1.6.6"
            cached_payload["provenance"]["cache_key"] = old_cache_key
            output_path.write_text(
                json.dumps(cached_payload),
                encoding="utf-8",
            )
            calls_before = client.calls

            _, cached = run_stage4(
                stage0_path,
                stage2_path,
                stage3_path,
                output_path,
                client,
                prompt,
                self.vocabulary,
                self.vocabulary_hash,
            )

            self.assertFalse(cached)
            self.assertEqual(client.calls, calls_before + 1)


    def test_preview_context_fallback_includes_all_non_reference_blocks_and_tables(self) -> None:
        document = stage0_document().model_copy(deep=True)
        for element in document.elements:
            element.section = "Abstract"
        document.elements[-1].section = "Conclusion"
        document.elements.append(document.elements[0].model_copy(update={
            "block_id": "P_REF_1",
            "section": "References",
            "text": "Reference entry that must not enter fallback context.",
            "source_block_index": 99,
        }))

        blocks, warnings, _ = select_context_blocks(
            document,
            stage2_document(),
            stage3_document(),
            max_input_chars=20000,
        )

        block_ids = {item.block_id for item in blocks}
        self.assertIn("T_2_0", block_ids)
        self.assertNotIn("P_REF_1", block_ids)
        self.assertTrue(any(item["code"] == "section_fallback" for item in warnings))

    def test_preview_point_inherits_unique_series_subject_before_semantic_bypass(self) -> None:
        payload = SeriesClient().call_json("", "").data
        series = payload["property_series"][0]
        series.update({
            "sample_id": "s001",
            "entity_id": "pe001",
            "sample_resolution_status": "resolved",
        })
        series["points"][0].update({
            "sample_id": None,
            "entity_id": None,
            "sample_resolution_status": None,
        })

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=True,
        )

        point = repaired["property_series"][0]["points"][0]
        self.assertEqual(point["sample_id"], "s001")
        self.assertEqual(point["entity_id"], "pe001")
        self.assertEqual(point["sample_resolution_status"], "resolved")
        self.assertEqual(repairs["point_subject_inherited_from_series"], 1)

    def test_preview_normalizes_source_text_alias_and_degrades_blank_locator(self) -> None:
        payload = {
            "properties": [],
            "unresolved_properties": [],
            "measurement_conditions": [],
            "property_series": [{
                "series_id": "series001",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "points": [{
                    "point_id": "pt001",
                    "sample_id": None,
                    "entity_id": None,
                    "sample_resolution_status": None,
                    "coordinates": [],
                    "evidence": [{
                        "block_id": "T_2_0",
                        "source_text": "8.5 to 8.6 (cal/ml)^1/2",
                        "table_locator": {
                            "table_id": "T_2_0",
                            "row_label": "dried PB film",
                            "column_label": "",
                        },
                    }],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=True,
        )

        evidence = repaired["property_series"][0]["points"][0]["evidence"][0]
        self.assertEqual(evidence["source_sentence"], "8.5 to 8.6 (cal/ml)^1/2")
        self.assertNotIn("source_text", evidence)
        self.assertIsNone(evidence["table_locator"])
        self.assertEqual(repairs["source_text_aliases_normalized"], 1)
        self.assertEqual(repairs["preview_blank_table_locators_degraded"], 1)

    def test_preview_salvage_keeps_multi_subject_series_points(self) -> None:
        payload = SeriesClient().call_json("", "").data
        series = payload["property_series"][0]
        first_point = series["points"][0]
        first_point.update({
            "sample_id": "s001",
            "entity_id": "pe001",
            "sample_resolution_status": "resolved",
        })
        second_point = json.loads(json.dumps(first_point))
        second_point.update({
            "point_id": "pt011",
            "sample_id": "s002",
            "entity_id": "pe002",
        })
        series.update({
            "sample_id": None,
            "entity_id": None,
            "sample_resolution_status": "unresolved",
            "points": [first_point, second_point],
            "coverage": None,
        })
        parsed = PropertyStageResponse.model_validate(payload)

        salvaged, materialized, report = _preview_salvage_materialization(
            parsed,
            list(stage0_document().elements),
        )

        self.assertEqual(len(salvaged.property_series), 1)
        self.assertEqual(len(salvaged.property_series[0].points), 2)
        self.assertEqual(len(materialized[3]), 1)
        self.assertEqual(report["dropped_points"], [])
        self.assertEqual(report["dropped_series"], [])

    def test_preview_only_removes_unsupported_series_field(self) -> None:
        preview_payload = SeriesClient().call_json("", "").data
        preview_payload["property_series"][0]["molecular_weight_type"] = "Mw"

        preview_repaired, preview_repairs = _repair_candidate_response_payload(
            preview_payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=True,
        )

        self.assertNotIn(
            "molecular_weight_type",
            preview_repaired["property_series"][0],
        )
        self.assertEqual(
            preview_repairs["preview_series_unsupported_fields_removed"],
            1,
        )

        strict_payload = SeriesClient().call_json("", "").data
        strict_payload["property_series"][0]["molecular_weight_type"] = "Mw"
        strict_repaired, strict_repairs = _repair_candidate_response_payload(
            strict_payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=False,
        )

        self.assertEqual(
            strict_repaired["property_series"][0]["molecular_weight_type"],
            "Mw",
        )
        self.assertEqual(
            strict_repairs["preview_series_unsupported_fields_removed"],
            0,
        )

    def test_preview_fills_blank_source_sentence_from_known_block(self) -> None:
        payload = {
            "properties": [],
            "unresolved_properties": [],
            "measurement_conditions": [],
            "property_series": [{
                "series_id": "series001",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "points": [{
                    "point_id": "pt001",
                    "sample_id": None,
                    "entity_id": None,
                    "sample_resolution_status": None,
                    "coordinates": [],
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": "",
                        "table_locator": None,
                    }],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=True,
        )

        evidence = repaired["property_series"][0]["points"][0]["evidence"][0]
        self.assertEqual(evidence["source_sentence"], RESULT_SENTENCE)
        self.assertEqual(repairs["preview_missing_source_sentences_filled"], 1)

    def test_preview_compacts_condition_quantity_range_fields(self) -> None:
        payload = {
            "properties": [],
            "unresolved_properties": [],
            "property_series": [],
            "measurement_conditions": [{
                "condition_id": "mc001",
                "condition_status": "reported",
                "temperature": {
                    "raw": "25 °C",
                    "value": None,
                    "value_min": 25,
                    "value_max": 25,
                    "unit": "°C",
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": RESULT_SENTENCE,
                        "table_locator": None,
                    }],
                },
                "other_conditions": {},
                "other_condition_evidence": {},
                "evidence": {
                    "block_id": "P_2_0",
                    "source_sentence": RESULT_SENTENCE,
                    "table_locator": None,
                },
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=True,
        )

        temperature = repaired["measurement_conditions"][0]["temperature"]
        self.assertEqual(temperature["value"], 25)
        self.assertNotIn("value_min", temperature)
        self.assertNotIn("value_max", temperature)
        self.assertEqual(
            repairs["preview_condition_quantity_ranges_compacted"],
            1,
        )

    def test_strict_does_not_fill_evidence_or_compact_condition_quantity(self) -> None:
        payload = {
            "properties": [],
            "unresolved_properties": [],
            "property_series": [],
            "measurement_conditions": [{
                "condition_id": "mc001",
                "condition_status": "reported",
                "temperature": {
                    "raw": "25 °C",
                    "value": None,
                    "value_min": 25,
                    "value_max": 25,
                    "unit": "°C",
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": "",
                        "table_locator": None,
                    }],
                },
                "other_conditions": {},
                "other_condition_evidence": {},
                "evidence": {
                    "block_id": "P_2_0",
                    "source_sentence": "",
                    "table_locator": None,
                },
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=False,
        )

        condition = repaired["measurement_conditions"][0]
        self.assertEqual(condition["evidence"]["source_sentence"], "")
        self.assertEqual(
            condition["temperature"]["evidence"][0]["source_sentence"],
            "",
        )
        self.assertIn("value_min", condition["temperature"])
        self.assertIn("value_max", condition["temperature"])
        self.assertEqual(repairs["preview_missing_source_sentences_filled"], 0)
        self.assertEqual(
            repairs["preview_condition_quantity_ranges_compacted"],
            0,
        )

    def test_preview_unwraps_singleton_coordinate_evidence(self) -> None:
        payload = {
            "properties": [],
            "unresolved_properties": [],
            "measurement_conditions": [],
            "property_series": [{
                "series_id": "series001",
                "points": [{
                    "point_id": "pt001",
                    "coordinates": [{
                        "name_raw": "solvent",
                        "value_raw": "NMP",
                        "evidence": [{
                            "block_id": "P_2_0",
                            "source_sentence": RESULT_SENTENCE,
                            "table_locator": None,
                        }],
                    }],
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": RESULT_SENTENCE,
                        "table_locator": None,
                    }],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=True,
        )

        evidence = repaired["property_series"][0]["points"][0][
            "coordinates"
        ][0]["evidence"]
        self.assertIsInstance(evidence, dict)
        self.assertEqual(evidence["block_id"], "P_2_0")
        self.assertEqual(
            repairs["preview_singleton_coordinate_evidence_unwrapped"],
            1,
        )

    def test_strict_keeps_singleton_coordinate_evidence_list(self) -> None:
        payload = {
            "properties": [],
            "unresolved_properties": [],
            "measurement_conditions": [],
            "property_series": [{
                "series_id": "series001",
                "points": [{
                    "point_id": "pt001",
                    "coordinates": [{
                        "name_raw": "solvent",
                        "value_raw": "NMP",
                        "evidence": [{
                            "block_id": "P_2_0",
                            "source_sentence": RESULT_SENTENCE,
                            "table_locator": None,
                        }],
                    }],
                    "evidence": [{
                        "block_id": "P_2_0",
                        "source_sentence": RESULT_SENTENCE,
                        "table_locator": None,
                    }],
                }],
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=False,
        )

        evidence = repaired["property_series"][0]["points"][0][
            "coordinates"
        ][0]["evidence"]
        self.assertIsInstance(evidence, list)
        self.assertEqual(
            repairs["preview_singleton_coordinate_evidence_unwrapped"],
            0,
        )

    def test_raw_response_artifact_serializes_decimal_costs(self) -> None:
        client = FakeClient()
        cost = LLMCallCost(
            currency="CNY",
            input_per_million=Decimal("1.1"),
            output_per_million=Decimal("2.2"),
            input_cost=Decimal("0.01"),
            output_cost=Decimal("0.02"),
            total_cost=Decimal("0.03"),
        )
        client.last_raw_response = LLMRawResponse(
            provider="test",
            model="fake-actual",
            finish_reason="stop",
            content='{"properties": []}',
            usage=LLMTokenUsage(input_tokens=12, output_tokens=34),
            cost=cost,
        )

        artifact = _stage4_raw_response_artifact(
            client,
            document_id="doc-test",
            history_start=0,
        )

        self.assertIsNotNone(artifact)
        json.dumps(artifact)
        self.assertEqual(
            artifact["raw_response"]["cost"]["total_cost"],
            "0.03",
        )

    def test_preview_rechecks_empty_reported_context_after_anchor_cleanup(self) -> None:
        payload = {
            "properties": [],
            "unresolved_properties": [],
            "property_series": [],
            "measurement_conditions": [{
                "condition_id": "cond001",
                "condition_status": "reported",
                "temperature": {
                    "raw": "999 K",
                    "value": 999,
                    "unit": "K",
                    "evidence": [{"block_id": "P_2_0"}],
                },
                "other_conditions": {},
                "other_condition_evidence": {},
                "evidence": {"block_id": "P_2_0"},
            }],
        }

        repaired, repairs = _repair_candidate_response_payload(
            payload,
            stage3_document(),
            stage0_document().elements,
            preview_relaxed=True,
        )

        condition = repaired["measurement_conditions"][0]
        self.assertIsNone(condition["temperature"])
        self.assertEqual(condition["condition_status"], "not_reported")
        self.assertGreaterEqual(repairs["empty_reported_contexts_downgraded"], 1)


if __name__ == "__main__":
    unittest.main()
