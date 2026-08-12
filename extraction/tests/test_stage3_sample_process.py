import json
import tempfile
import unittest
from pathlib import Path


from llm_client import (
    DEFAULT_CONFIG_PATH,
    LLMJSONResponse,
    ResolvedLLMConfig,
    load_pipeline_config,
)
from prompt_loader import PromptLoader
from schema.polymer_schema import (
    SampleProcessResponse,
    Stage0Document,
    Stage2Document,
)
from tests.helpers import add_model_confidence
from stages.stage3_sample_process import (
    IMPLEMENTATION_VERSION,
    Stage3Error,
    _cache_components,
    _remove_process_input_output_overlap,
    _resolve_surface_text,
    _split_misbound_hot_pressing_drying_outputs,
    _split_consecutive_extraction_drying_outputs,
    _split_preview_in_place_postprocess_outputs,
    _failure_replay_client,
    extract_samples_processes,
    run_stage3,
    select_context_blocks,
)


METHOD_SENTENCE = (
    "Buna CB was dried at 60 °C for 6 h to obtain dried PB film."
)
FRACTIONATION_SENTENCE = (
    "Buna CB was polymerized to obtain polymerized material; "
    "the hexane-soluble fraction was then isolated."
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
                "text": "Polybutadiene was studied.",
                "page": 0,
                "bbox": [1, 2, 3, 4],
                "source_block_index": 0,
            },
            {
                "block_id": "P_1_0",
                "type": "text",
                "section": method_section,
                "text": METHOD_SENTENCE,
                "page": 1,
                "bbox": [5, 6, 7, 8],
                "source_block_index": 1,
            },
        ],
        "warnings": [],
    })



def fractionation_stage0_document() -> Stage0Document:
    data = stage0_document().model_dump(mode="json")
    data["elements"][1]["text"] = FRACTIONATION_SENTENCE
    return Stage0Document.model_validate(data)


def stage2_document() -> Stage2Document:
    digest = "b" * 64
    return Stage2Document.model_validate({
        "schema_version": "1.0",
        "document_id": "reference_no_0000001",
        "polymer_entities": [
            {
                "entity_id": "pe001",
                "polymer_name": "cis-polybutadiene",
                "polymer_type": None,
                "variant_of": None,
                "representation_status": "expert_review_required",
                "structural_features": [],
                "source_names": ["cis-polybutadiene", "Buna CB"],
                "resolved_from_mentions": ["m001"],
                "evidence": {
                    "block_id": "P_1_0",
                    "page": 1,
                    "bbox": [5, 6, 7, 8],
                    "source_type": "text",
                    "source_sentence": METHOD_SENTENCE,
                },
                "source_image_refs": [],
            }
        ],
        "unresolved_mention_ids": [],
        "provenance": {
            "provider": "test",
            "model": "fake",
            "models": ["fake"],
            "prompt_id": "polymer.stage2.polymer_entity",
            "prompt_version": "1.0.0",
            "prompt_sha256": digest,
            "input_hash": digest,
            "model_config_hash": digest,
            "cache_key": digest,
            "output_schema_version": "polymer_entity_schema.v1",
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
                "samples": [
                    {
                        "sample_id": "s010",
                        "sample_kind": "commercial_batch",
                        "refers_to_entity": "pe001",
                        "sample_label_raw": "Buna CB",
                        "state_description": None,
                        "intended_use": [],
                        "evidence": {
                            "block_id": "P_1_0",
                            "source_sentence": METHOD_SENTENCE,
                        },
                    },
                    {
                        "sample_id": "s020",
                        "sample_kind": "processed_material",
                        "refers_to_entity": "pe001",
                        "sample_label_raw": "dried PB film",
                        "state_description": None,
                        "intended_use": [],
                        "evidence": {
                            "block_id": "P_1_0",
                            "source_sentence": METHOD_SENTENCE,
                        },
                    },
                ],
                "process_steps": [
                    {
                        "step_id": "ps010",
                        "process_type": "drying",
                        "input_sample_ids": ["s010"],
                        "output_sample_ids": ["s020"],
                        "parameters": {
                            "temperature": "60 °C",
                            "time": "6 h",
                        },
                        "evidence": {
                            "block_id": "P_1_0",
                            "source_sentence": METHOD_SENTENCE,
                        },
                    }
                ],
                "unresolved_entity_ids": [],
            }),
            provider="test",
            model="fake-actual",
        )


class SampleTypeClient(FakeClient):
    def __init__(
        self,
        *,
        polymer_type: str = "random_copolymer",
        material_type: str = "neat_resin",
    ) -> None:
        super().__init__()
        self.polymer_type = polymer_type
        self.material_type = material_type

    def call_json(self, *args, **kwargs) -> LLMJSONResponse:
        response = super().call_json(*args, **kwargs)
        for sample in response.data["samples"]:
            sample["polymer_type"] = self.polymer_type
            sample["material_type"] = self.material_type
        return response


class MissingEntityCoverageClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        return LLMJSONResponse(
            data={
                "samples": [],
                "process_steps": [],
                "unresolved_entity_ids": [],
            },
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
                    "samples": [],
                    "process_steps": [],
                    "unresolved_entity_ids": [],
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


class UnknownConfidenceFieldClient(FakeClient):
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
        response.data["samples"][0]["confidence"]["input_relation"] = "s001"
        return response


class MisplacedTableCellConfidenceClient(FakeClient):
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
        response.data["samples"][0]["confidence"][
            "uncertainty_codes"
        ] = ["table_cell"]
        return response


class MisplacedCrossSentenceConfidenceClient(FakeClient):
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
        response.data["samples"][0]["confidence"][
            "uncertainty_codes"
        ] = ["cross_sentence_link"]
        return response


class UnknownUncertaintyCodeClient(FakeClient):
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
        response.data["samples"][0]["confidence"][
            "uncertainty_codes"
        ] = ["not_a_schema_code"]
        return response


class MisplacedSampleEvidenceClient(FakeClient):
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
        response.data["samples"][1]["evidence"] = {
            "block_id": "P_1_1",
            "source_sentence": "An unrelated methods sentence.",
        }
        return response


class AmbiguousSampleEvidenceClient(MisplacedSampleEvidenceClient):
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
        response.data["samples"][1]["evidence"] = {
            "block_id": "P_1_2",
            "source_sentence": "Another unrelated methods sentence.",
        }
        return response


class UnsupportedLabelWithStateClient(FakeClient):
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
        response.data["samples"][1]["sample_label_raw"] = "PB films"
        response.data["samples"][1]["state_description"] = "dried PB film"
        return response


class UnsupportedLabelAndStateClient(UnsupportedLabelWithStateClient):
    def call_json(self, *args, **kwargs) -> LLMJSONResponse:
        response = super().call_json(*args, **kwargs)
        response.data["samples"][1]["state_description"] = "invented state"
        return response


class UnsupportedSampleLabelClient(FakeClient):
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
        sample = response.data["samples"][1]
        sample["refers_to_entity"] = None
        sample["sample_label_raw"] = "invented sample label"
        sample["state_description"] = None
        return response


class UnsupportedIntermediateLabelClient(FakeClient):
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
        sample = response.data["samples"][1]
        sample["sample_kind"] = "intermediate"
        sample["refers_to_entity"] = None
        sample["sample_label_raw"] = "unsupported-code"
        sample["state_description"] = None
        return response


class CompleteInputOutputOverlapClient(FakeClient):
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
        response.data["process_steps"][0]["input_sample_ids"] = ["s020"]
        return response


class PartialInputOutputOverlapClient(FakeClient):
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
        response.data["process_steps"][0]["input_sample_ids"] = [
            "s010",
            "s020",
        ]
        return response


class InvalidStateWithLabelClient(FakeClient):
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
        response.data["samples"][1]["state_description"] = (
            "crosslinked samples"
        )
        return response


class UnsupportedIntendedUseClient(FakeClient):
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
        response.data["samples"][1]["intended_use"] = [
            "to obtain",
            "invented downstream use",
        ]
        return response


class UnsupportedIntendedUseWithoutAnchorClient(FakeClient):
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
        response.data["samples"][1]["sample_label_raw"] = None
        response.data["samples"][1]["state_description"] = "invented state"
        response.data["samples"][1]["intended_use"] = ["invented use"]
        return response


class CycleClient(FakeClient):
    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        sample = {
            "sample_kind": "processed_material",
            "refers_to_entity": "pe001",
            "sample_label_raw": "Buna CB",
            "state_description": None,
            "intended_use": [],
            "evidence": {
                "block_id": "P_1_0",
                "source_sentence": METHOD_SENTENCE,
            },
        }
        step = {
            "process_type": "drying",
            "parameters": {},
            "evidence": {
                "block_id": "P_1_0",
                "source_sentence": METHOD_SENTENCE,
            },
        }
        return LLMJSONResponse(
            data=add_model_confidence({
                "samples": [
                    {**sample, "sample_id": "s001"},
                    {**sample, "sample_id": "s002"},
                ],
                "process_steps": [
                    {
                        **step,
                        "step_id": "ps001",
                        "input_sample_ids": ["s001"],
                        "output_sample_ids": ["s002"],
                    },
                    {
                        **step,
                        "step_id": "ps002",
                        "input_sample_ids": ["s002"],
                        "output_sample_ids": ["s001"],
                    },
                ],
                "unresolved_entity_ids": [],
            }),
            provider="test",
            model="fake-actual",
        )


class CycleAfterOverlapRepairClient(CycleClient):
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
        response.data["process_steps"][0]["input_sample_ids"].append("s002")
        return response


class DuplicateOutputAfterOverlapRepairClient(FakeClient):
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
        duplicate_step = dict(response.data["process_steps"][0])
        duplicate_step.update({
            "step_id": "ps020",
            "input_sample_ids": ["s020"],
            "output_sample_ids": ["s020"],
        })
        response.data["process_steps"].append(duplicate_step)
        return response



class DuplicateFractionProducerClient(FakeClient):
    evidence_sentence = FRACTIONATION_SENTENCE
    fraction_label = "hexane-soluble fraction"
    intermediate_label = "polymerized material"

    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        self.calls += 1
        evidence = {
            "block_id": "P_1_0",
            "source_sentence": self.evidence_sentence,
        }
        return LLMJSONResponse(
            data=add_model_confidence({
                "samples": [
                    {
                        "sample_id": "s010",
                        "sample_kind": "commercial_batch",
                        "refers_to_entity": "pe001",
                        "sample_label_raw": "Buna CB",
                        "state_description": None,
                        "intended_use": [],
                        "evidence": evidence,
                    },
                    {
                        "sample_id": "s020",
                        "sample_kind": "processed_material",
                        "refers_to_entity": "pe001",
                        "sample_label_raw": self.fraction_label,
                        "state_description": None,
                        "intended_use": [],
                        "evidence": evidence,
                    },
                    {
                        "sample_id": "s030",
                        "sample_kind": "intermediate",
                        "refers_to_entity": "pe001",
                        "sample_label_raw": self.intermediate_label,
                        "state_description": None,
                        "intended_use": [],
                        "evidence": evidence,
                    },
                ],
                "process_steps": [
                    {
                        "step_id": "ps010",
                        "process_type": "polymerization",
                        "input_sample_ids": ["s010"],
                        "output_sample_ids": ["s020", "s030"],
                        "parameters": {},
                        "evidence": evidence,
                    },
                    {
                        "step_id": "ps020",
                        "process_type": "fractionation",
                        "input_sample_ids": ["s030"],
                        "output_sample_ids": ["s020"],
                        "parameters": {},
                        "evidence": evidence,
                    },
                ],
                "unresolved_entity_ids": [],
            }),
            provider="test",
            model="fake-actual",
        )


class AmbiguousDuplicateFractionProducerClient(DuplicateFractionProducerClient):
    evidence_sentence = METHOD_SENTENCE
    fraction_label = "dried PB film"
    intermediate_label = "Buna CB"


class InvalidParameterClient(FakeClient):
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
        response.data["process_steps"][0]["parameters"]["temperature"] = "333 K"
        return response


class InventedSampleStateClient(FakeClient):
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
        response.data["samples"][1]["sample_label_raw"] = None
        response.data["samples"][1]["state_description"] = (
            "containing 2.0 phr DICUP"
        )
        return response


class ExpandedProcessTypeClient(FakeClient):
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
        response.data["process_steps"][0]["process_type"] = (
            "specimen_preparation"
        )
        return response


class DerivedSampleNameClient(FakeClient):
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
        response.data["samples"][1]["sample_label_raw"] = None
        response.data["samples"][1]["state_description"] = (
            "dried at 60 °C for 6 h"
        )
        return response


class CaseChangedSampleNameClient(FakeClient):
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
        response.data["samples"][1]["sample_label_raw"] = "DRIED PB FILM"
        return response


class HtmlEntitySampleNameClient(FakeClient):
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
        response.data["samples"][0]["sample_label_raw"] = "Buna'CB"
        return response


class FormattingChangedEvidenceClient(FakeClient):
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
        changed = (
            "BUNA CB  WAS DRIED AT 60 °C FOR 6 H TO OBTAIN DRIED PB FILM."
        )
        response.data["samples"][0]["evidence"]["source_sentence"] = changed
        response.data["process_steps"][0]["parameters"]["time"] = "6 H"
        return response


class ParaphrasedEvidenceClient(FakeClient):
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
        response.data["samples"][0]["evidence"]["source_sentence"] = (
            "The commercial rubber was dried."
        )
        response.data["process_steps"][0]["evidence"]["source_sentence"] = (
            "The material underwent a drying treatment."
        )
        return response


class PronounEvidenceClient(FakeClient):
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
        response.data["samples"][1]["sample_label_raw"] = None
        response.data["samples"][1]["state_description"] = (
            "dried under vacuum"
        )
        response.data["samples"][1]["evidence"] = {
            "block_id": "P_1_1",
            "source_sentence": "The sample was vacuum dried.",
        }
        return response


def rendered_prompt():
    return PromptLoader().render_stage_prompt(
        "polymer.stage3.sample_process",
        SampleProcessResponse,
        expected_stage="stage3_sample_process",
        expected_output_schema="sample_process_schema.v3",
    )


class Stage3Tests(unittest.TestCase):
    def test_misbound_hot_pressing_drying_outputs_are_split(self) -> None:
        sentence = "The blend was hot pressed and then dried."
        payload = {
            "samples": [
                {
                    "sample_id": "s001",
                    "refers_to_entity": "pe001",
                    "evidence": {"block_id": "P_1"},
                },
                {
                    "sample_id": "s002",
                    "refers_to_entity": "pe002",
                    "evidence": {"block_id": "P_2"},
                },
            ],
            "process_steps": [
                {
                    "step_id": "ps001",
                    "process_type": "hot_pressing",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": {
                        "block_id": "P_1",
                        "source_sentence": sentence,
                    },
                    "confidence": {"score": 0.8},
                },
                {
                    "step_id": "ps002",
                    "process_type": "drying",
                    "input_sample_ids": ["s002"],
                    "output_sample_ids": ["s002"],
                    "evidence": {
                        "block_id": "P_1",
                        "source_sentence": sentence,
                    },
                    "confidence": {"score": 0.8},
                },
            ],
        }

        repaired, repairs = _split_misbound_hot_pressing_drying_outputs(
            payload
        )

        self.assertEqual(
            repaired["process_steps"][0]["output_sample_ids"], ["s003"]
        )
        self.assertEqual(
            repaired["process_steps"][1]["output_sample_ids"], ["s004"]
        )
        self.assertEqual(repaired["samples"][-1]["refers_to_entity"], "pe001")
        self.assertEqual(
            repairs[0]["pattern"], "misbound_hot_pressing_then_drying"
        )
    def test_latex_group_resolves_compact_sample_label(self) -> None:
        source = r"The monomer ia, $\mathbf { i b , }$and ic was used."

        self.assertEqual(
            _resolve_surface_text(source, "ib"),
            r"\mathbf { i b , }",
        )

    def test_duplicate_process_sample_references_are_deduplicated(self) -> None:
        repaired, repairs = _remove_process_input_output_overlap({
            "process_steps": [{
                "step_id": "ps001",
                "input_sample_ids": [],
                "output_sample_ids": ["s001", "s001"],
            }],
        })

        self.assertEqual(
            repaired["process_steps"][0]["output_sample_ids"],
            ["s001"],
        )
        self.assertEqual(
            repairs[0]["duplicate_output_sample_ids"],
            ["s001"],
        )

    def test_consecutive_casting_and_pressing_get_intermediate_sample(
        self,
    ) -> None:
        payload = {
            "samples": [
                {"sample_id": "s001", "refers_to_entity": "pe001"},
                {"sample_id": "s002", "refers_to_entity": "pe001"},
            ],
            "process_steps": [
                {
                    "step_id": "ps001",
                    "process_type": "casting",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": {
                        "block_id": "P_1",
                        "source_sentence": "The blend was cast as a film.",
                    },
                    "confidence": {"score": 0.9},
                },
                {
                    "step_id": "ps002",
                    "process_type": "pressing",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": {
                        "block_id": "P_2",
                        "source_sentence": "The film was pressed.",
                    },
                    "confidence": {"score": 0.8},
                },
            ],
        }

        repaired, repairs = _split_consecutive_extraction_drying_outputs(
            payload
        )

        self.assertEqual(
            repaired["process_steps"][0]["output_sample_ids"], ["s003"]
        )
        self.assertEqual(
            repaired["process_steps"][1]["input_sample_ids"], ["s003"]
        )
        self.assertEqual(
            repaired["process_steps"][1]["output_sample_ids"], ["s002"]
        )
        self.assertEqual(repairs[0]["pattern"], "casting_then_pressing")

    def test_missing_entity_coverage_is_completed_as_unresolved(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            MissingEntityCoverageClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.unresolved_entity_ids, ["pe001"])
        warning = next(
            item
            for item in result.warnings
            if item["code"] == "unresolved_entities_completed"
        )
        self.assertEqual(warning["entity_ids"], ["pe001"])

    def test_html_entity_surface_returns_original_source_fragment(self) -> None:
        source = "<td>BTDA/4,4&#x27;-BABBP</td>"

        self.assertEqual(
            _resolve_surface_text(
                source,
                "BTDA/4,4'-BABBP",
                allow_html_entities=True,
            ),
            "BTDA/4,4&#x27;-BABBP",
        )
        self.assertIsNone(
            _resolve_surface_text(
                source,
                "BTDA/4,4'-BABDE",
                allow_html_entities=True,
            )
        )

    def test_previous_implementation_cache_is_not_reused(self) -> None:
        document = stage0_document()
        entities = stage2_document()
        client = FakeClient()
        prompt = rendered_prompt()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage2_path = root / "stage2_entities.json"
            output_path = root / "stage3_process.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage2_path.write_text(
                json.dumps(entities.model_dump(mode="json")),
                encoding="utf-8",
            )
            run_stage3(
                stage0_path,
                stage2_path,
                output_path,
                client,
                prompt,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            _, _, old_cache_key = _cache_components(
                document,
                entities,
                prompt,
                client,
                implementation_version="1.3.2",
            )
            payload["provenance"]["implementation_version"] = "1.3.2"
            payload["provenance"]["cache_key"] = old_cache_key
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            calls_after_first = client.calls

            _, cached = run_stage3(
                stage0_path,
                stage2_path,
                output_path,
                client,
                prompt,
            )

            self.assertEqual(IMPLEMENTATION_VERSION, "1.4.0")
            self.assertFalse(cached)
            self.assertEqual(client.calls, calls_after_first + 1)
    def test_sample_label_html_entity_is_recovered_with_warning(self) -> None:
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"][1]["text"] = (
            "Buna&#x27;CB was dried at 60 °C for 6 h to obtain dried PB film."
        )
        document = Stage0Document.model_validate(document_data)
        entities_data = stage2_document().model_dump(mode="json")
        entities_data["polymer_entities"][0]["evidence"]["source_sentence"] = (
            document_data["elements"][1]["text"]
        )
        entities = Stage2Document.model_validate(entities_data)

        result = extract_samples_processes(
            document,
            entities,
            HtmlEntitySampleNameClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.samples[0].sample_label_raw, "Buna&#x27;CB")
        warning = next(
            item
            for item in result.warnings
            if item["code"] == "sample_label_html_entity_surface_recovered"
        )
        self.assertEqual(warning["items"][0]["sample_id"], "s001")
        self.assertEqual(warning["items"][0]["to"], "Buna&#x27;CB")

    def test_consecutive_extraction_and_drying_get_intermediate_samples(
        self,
    ) -> None:
        samples = [
            {
                "sample_id": sample_id,
                "sample_kind": "test_specimen",
                "refers_to_entity": entity_id,
                "sample_label_raw": label,
                "state_description": None,
                "intended_use": [],
                "evidence": {
                    "block_id": "P_0_1",
                    "source_sentence": "PB film was dried.",
                },
                "confidence": {"score": 0.9},
            }
            for sample_id, entity_id, label in (
                ("s001", "pe001", "A"),
                ("s002", "pe002", "B"),
                ("s003", "pe001", "A"),
                ("s004", "pe002", "B"),
            )
        ]
        extraction_sentence = "Samples were extracted with hexane."
        payload = {
            "samples": samples,
            "process_steps": [
                {
                    "step_id": "ps001",
                    "process_type": "solvent_extraction",
                    "input_sample_ids": ["s001", "s002"],
                    "output_sample_ids": ["s003", "s004"],
                    "parameters": {},
                    "evidence": {
                        "block_id": "P_0_1",
                        "source_sentence": extraction_sentence,
                    },
                    "confidence": {"score": 0.9},
                },
                {
                    "step_id": "ps002",
                    "process_type": "drying",
                    "input_sample_ids": ["s003", "s004"],
                    "output_sample_ids": ["s003", "s004"],
                    "parameters": {},
                    "evidence": {
                        "block_id": "P_0_1",
                        "source_sentence": "PB film was dried.",
                    },
                    "confidence": {"score": 0.8},
                },
            ],
            "unresolved_entity_ids": [],
        }

        repaired, repairs = _split_consecutive_extraction_drying_outputs(
            payload
        )

        self.assertEqual(
            repaired["process_steps"][0]["output_sample_ids"],
            ["s005", "s006"],
        )
        self.assertEqual(
            repaired["process_steps"][1]["input_sample_ids"],
            ["s005", "s006"],
        )
        self.assertEqual(
            repaired["process_steps"][1]["output_sample_ids"],
            ["s003", "s004"],
        )
        self.assertEqual(
            [item["refers_to_entity"] for item in repaired["samples"][-2:]],
            ["pe001", "pe002"],
        )
        self.assertTrue(all(
            item["sample_kind"] == "intermediate"
            and item["state_description"] == extraction_sentence
            for item in repaired["samples"][-2:]
        ))
        self.assertEqual(len(repairs), 1)

    def test_consecutive_process_repair_requires_matching_entities(self) -> None:
        payload = {
            "samples": [
                {"sample_id": "s001", "refers_to_entity": "pe001"},
                {"sample_id": "s002", "refers_to_entity": "pe002"},
            ],
            "process_steps": [
                {
                    "step_id": "ps001",
                    "process_type": "solvent_extraction",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": {"source_sentence": "extracted"},
                    "confidence": {"score": 0.9},
                },
                {
                    "step_id": "ps002",
                    "process_type": "drying",
                    "input_sample_ids": ["s002"],
                    "output_sample_ids": ["s002"],
                },
            ],
        }

        repaired, repairs = _split_consecutive_extraction_drying_outputs(
            payload
        )

        self.assertEqual(repairs, [])
        self.assertEqual(len(repaired["samples"]), 2)

    def test_consecutive_process_repair_requires_explicit_entities(self) -> None:
        payload = {
            "samples": [
                {"sample_id": "s001", "refers_to_entity": None},
                {"sample_id": "s002", "refers_to_entity": None},
            ],
            "process_steps": [
                {
                    "step_id": "ps001",
                    "process_type": "solvent_extraction",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": {"source_sentence": "extracted"},
                    "confidence": {"score": 0.9},
                },
                {
                    "step_id": "ps002",
                    "process_type": "drying",
                    "input_sample_ids": ["s002"],
                    "output_sample_ids": ["s002"],
                },
            ],
        }

        repaired, repairs = _split_consecutive_extraction_drying_outputs(
            payload
        )

        self.assertEqual(repairs, [])
        self.assertEqual(len(repaired["samples"]), 2)

    def test_consecutive_casting_and_drying_get_intermediate_sample(
        self,
    ) -> None:
        sentence = "The solution was cast onto glass and dried to a film."
        evidence = {"block_id": "P_3_45", "source_sentence": sentence}
        payload = {
            "samples": [
                {"sample_id": "s001", "refers_to_entity": "pe001"},
                {"sample_id": "s002", "refers_to_entity": "pe001"},
            ],
            "process_steps": [
                {
                    "step_id": "ps001",
                    "process_type": "casting",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": evidence,
                    "confidence": {"score": 0.9},
                },
                {
                    "step_id": "ps002",
                    "process_type": "drying",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": evidence,
                    "confidence": {"score": 0.8},
                },
            ],
        }

        repaired, repairs = _split_consecutive_extraction_drying_outputs(
            payload
        )

        self.assertEqual(
            repaired["process_steps"][0]["output_sample_ids"], ["s003"]
        )
        self.assertEqual(
            repaired["process_steps"][1]["input_sample_ids"], ["s003"]
        )
        self.assertEqual(
            repaired["process_steps"][1]["output_sample_ids"], ["s002"]
        )
        self.assertEqual(repaired["samples"][-1]["refers_to_entity"], "pe001")
        self.assertEqual(repaired["samples"][-1]["state_description"], sentence)
        self.assertEqual(repairs[0]["pattern"], "casting_then_drying")

    def test_casting_drying_repair_requires_identical_evidence(self) -> None:
        payload = {
            "samples": [
                {"sample_id": "s001", "refers_to_entity": "pe001"},
                {"sample_id": "s002", "refers_to_entity": "pe001"},
            ],
            "process_steps": [
                {
                    "step_id": "ps001",
                    "process_type": "casting",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": {
                        "block_id": "P_1",
                        "source_sentence": "The solution was cast.",
                    },
                    "confidence": {"score": 0.9},
                },
                {
                    "step_id": "ps002",
                    "process_type": "drying",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": {
                        "block_id": "P_2",
                        "source_sentence": "The film was dried.",
                    },
                    "confidence": {"score": 0.8},
                },
            ],
        }

        repaired, repairs = _split_consecutive_extraction_drying_outputs(
            payload
        )

        self.assertEqual(repairs, [])
        self.assertEqual(repaired, payload)

    def test_non_casting_drying_shared_output_is_not_repaired(self) -> None:
        sentence = "The sample was prepared and dried."
        evidence = {"block_id": "P_1", "source_sentence": sentence}
        payload = {
            "samples": [
                {"sample_id": "s001", "refers_to_entity": "pe001"},
                {"sample_id": "s002", "refers_to_entity": "pe001"},
            ],
            "process_steps": [
                {
                    "step_id": "ps001",
                    "process_type": "annealing",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": evidence,
                    "confidence": {"score": 0.9},
                },
                {
                    "step_id": "ps002",
                    "process_type": "drying",
                    "input_sample_ids": ["s001"],
                    "output_sample_ids": ["s002"],
                    "evidence": evidence,
                    "confidence": {"score": 0.8},
                },
            ],
        }

        repaired, repairs = _split_consecutive_extraction_drying_outputs(
            payload
        )

        self.assertEqual(repairs, [])
        self.assertEqual(repaired, payload)

    def test_prompt_requires_verbatim_raw_fields(self) -> None:
        prompt = rendered_prompt()

        self.assertEqual(prompt.version, "1.3.0")
        self.assertIn("不得翻译、概括、添加括号解释", prompt.text)
        self.assertIn("无法从 evidence 逐字复制时必须设为 `null`", prompt.text)
        self.assertIn("`intended_use` 只能放入", prompt.text)
        self.assertIn("共混物 Sample 和纤维 Sample", prompt.text)
        self.assertIn("所有 `*_raw` 字段必须", prompt.text)

    def test_extracts_samples_process_and_remaps_ids(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            FakeClient(),
            rendered_prompt(),
        )

        self.assertEqual(
            [sample.sample_id for sample in result.samples],
            ["s001", "s002"],
        )
        self.assertEqual(result.process_steps[0].step_id, "ps001")
        self.assertEqual(
            result.process_steps[0].input_sample_ids,
            ["s001"],
        )
        self.assertEqual(
            result.process_steps[0].output_sample_ids,
            ["s002"],
        )
        self.assertEqual(
            result.process_steps[0].parameters["temperature"],
            "60 °C",
        )
        self.assertEqual(result.provenance.model, "fake-actual")
        self.assertEqual(result.warnings, [])

    def test_missing_entity_coverage_is_completed_without_retry(self) -> None:
        client = RetryClient()

        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            client,
            rendered_prompt(),
            max_validation_retries=1,
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(result.samples, [])
        self.assertEqual(result.unresolved_entity_ids, ["pe001"])

    def test_unknown_confidence_field_is_dropped_with_warning(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            UnknownConfidenceFieldClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        warning = next(
            item
            for item in result.warnings
            if item["code"] == "confidence_fields_compacted"
        )
        self.assertEqual(
            warning["fields"],
            ["samples[0].confidence.input_relation"],
        )

    def test_table_cell_confidence_detail_is_compacted(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            MisplacedTableCellConfidenceClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.samples[0].confidence.model_dump(), {"score": 0.9})
        warning = next(
            item
            for item in result.warnings
            if item["code"] == "confidence_fields_compacted"
        )
        self.assertEqual(
            warning["fields"],
            ["samples[0].confidence.uncertainty_codes"],
        )

    def test_cross_sentence_confidence_detail_is_compacted(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            MisplacedCrossSentenceConfidenceClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.samples[0].confidence.model_dump(), {"score": 0.9})
        self.assertTrue(any(
            item["code"] == "confidence_fields_compacted"
            for item in result.warnings
        ))

    def test_other_unknown_uncertainty_code_is_compacted(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            UnknownUncertaintyCodeClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )
        self.assertEqual(result.samples[0].confidence.model_dump(), {"score": 0.9})

    def test_sample_evidence_is_relinked_when_label_block_is_unique(self) -> None:
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"].append({
            "block_id": "P_1_1",
            "type": "text",
            "section": "Methods",
            "text": "An unrelated methods sentence.",
            "page": 1,
            "bbox": [5, 9, 7, 10],
            "source_block_index": 2,
        })

        result = extract_samples_processes(
            Stage0Document.model_validate(document_data),
            stage2_document(),
            MisplacedSampleEvidenceClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.samples[1].evidence.block_id, "P_1_0")
        self.assertTrue(any(
            item["code"] == "sample_evidence_relinked"
            for item in result.warnings
        ))

    def test_sample_evidence_relink_rejects_ambiguous_label(self) -> None:
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"].extend([
            {
                "block_id": "P_1_1",
                "type": "text",
                "section": "Methods",
                "text": "A second dried PB film was retained.",
                "page": 1,
                "bbox": [5, 9, 7, 10],
                "source_block_index": 2,
            },
            {
                "block_id": "P_1_2",
                "type": "text",
                "section": "Methods",
                "text": "Another unrelated methods sentence.",
                "page": 1,
                "bbox": [5, 11, 7, 12],
                "source_block_index": 3,
            },
        ])

        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                Stage0Document.model_validate(document_data),
                stage2_document(),
                AmbiguousSampleEvidenceClient(),
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_unsupported_label_is_dropped_when_exact_state_exists(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            UnsupportedLabelWithStateClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertIsNone(result.samples[1].sample_label_raw)
        self.assertEqual(result.samples[1].state_description, "dried PB film")
        self.assertTrue(any(
            item["code"] == "unsupported_sample_labels_dropped"
            for item in result.warnings
        ))

    def test_dropped_label_allows_state_to_fall_back_to_evidence(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            UnsupportedLabelAndStateClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertIsNone(result.samples[1].sample_label_raw)
        self.assertEqual(result.samples[1].state_description, METHOD_SENTENCE)

    def test_unresolved_intermediate_uses_verified_evidence_as_state(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            UnsupportedIntermediateLabelClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertIsNone(result.samples[1].sample_label_raw)
        self.assertEqual(result.samples[1].state_description, METHOD_SENTENCE)
        self.assertTrue(any(
            item["code"] == "state_description_replaced_with_evidence"
            for item in result.warnings
        ))

    def test_process_cycle_is_rejected(self) -> None:
        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                stage0_document(),
                stage2_document(),
                CycleClient(),
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_complete_input_output_overlap_removes_only_input(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            CompleteInputOutputOverlapClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.process_steps[0].input_sample_ids, [])
        self.assertEqual(result.process_steps[0].output_sample_ids, ["s002"])
        self.assertEqual(len(result.samples), 2)
        warning = next(
            item for item in result.warnings
            if item["code"] == "process_input_unresolved"
        )
        self.assertEqual(
            warning["steps"],
            [{
                "step_id": "ps001",
                "removed_input_sample_ids": ["s002"],
            }],
        )

    def test_partial_input_output_overlap_preserves_other_inputs(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            PartialInputOutputOverlapClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.process_steps[0].input_sample_ids, ["s001"])
        self.assertEqual(result.process_steps[0].output_sample_ids, ["s002"])

    def test_cycle_remaining_after_overlap_repair_is_rejected(self) -> None:
        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                stage0_document(),
                stage2_document(),
                CycleAfterOverlapRepairClient(),
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_duplicate_output_after_overlap_repair_is_rejected(self) -> None:
        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                stage0_document(),
                stage2_document(),
                DuplicateOutputAfterOverlapRepairClient(),
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_duplicate_fraction_output_is_rejected_in_strict(self) -> None:
        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                fractionation_stage0_document(),
                stage2_document(),
                DuplicateFractionProducerClient(),
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_preview_removes_duplicate_upstream_fraction_output(self) -> None:
        result = extract_samples_processes(
            fractionation_stage0_document(),
            stage2_document(),
            DuplicateFractionProducerClient(),
            rendered_prompt(),
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(result.process_steps[0].output_sample_ids, ["s003"])
        self.assertEqual(result.process_steps[1].input_sample_ids, ["s003"])
        self.assertEqual(result.process_steps[1].output_sample_ids, ["s002"])
        producers = [
            step.step_id
            for step in result.process_steps
            if "s002" in step.output_sample_ids
        ]
        self.assertEqual(producers, ["ps002"])
        warning = next(
            item for item in result.warnings
            if item["code"]
            == "preview_duplicate_upstream_fraction_outputs_removed"
        )
        self.assertEqual(
            warning["repairs"],
            [{
                "polymerization_step_id": "ps001",
                "fractionation_step_id": "ps002",
                "sample_ids": ["s002"],
            }],
        )

    def test_preview_does_not_guess_duplicate_fraction_without_anchor(self) -> None:
        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                stage0_document(),
                stage2_document(),
                AmbiguousDuplicateFractionProducerClient(),
                rendered_prompt(),
                max_validation_retries=0,
                preview_relaxed=True,
            )

    def test_preview_splits_in_place_drying_outputs(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            DuplicateOutputAfterOverlapRepairClient(),
            rendered_prompt(),
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(len(result.samples), 3)
        self.assertEqual(result.process_steps[1].input_sample_ids, ["s002"])
        self.assertEqual(result.process_steps[1].output_sample_ids, ["s003"])
        produced = [
            sample_id
            for step in result.process_steps
            for sample_id in step.output_sample_ids
        ]
        self.assertEqual(len(produced), len(set(produced)))
        warning = next(
            item for item in result.warnings
            if item["code"] == "preview_in_place_postprocess_outputs_split"
        )
        self.assertEqual(warning["repairs"], [{
            "step_id": "ps002",
            "input_sample_ids": ["s002"],
            "final_sample_ids": ["s003"],
        }])

    def test_parameter_not_in_source_is_dropped_with_warning(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            InvalidParameterClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertNotIn(
            "temperature",
            result.process_steps[0].parameters,
        )
        self.assertEqual(
            result.warnings[0]["code"],
            "parameters_not_in_source",
        )
        self.assertEqual(
            result.warnings[0]["steps"][0],
            {
                "step_id": "ps001",
                "parameter_keys": ["temperature"],
            },
        )

    def test_invalid_state_without_label_uses_verified_evidence(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            InventedSampleStateClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.samples[1].state_description, METHOD_SENTENCE)
        self.assertTrue(any(
            item["code"] == "state_description_replaced_with_evidence"
            for item in result.warnings
        ))

    def test_invalid_state_with_valid_label_is_dropped(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            InvalidStateWithLabelClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.samples[1].sample_label_raw, "dried PB film")
        self.assertIsNone(result.samples[1].state_description)
        self.assertTrue(any(
            item["code"] == "unsupported_state_descriptions_dropped"
            for item in result.warnings
        ))

    def test_invalid_intended_use_is_dropped_when_sample_is_anchored(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            UnsupportedIntendedUseClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(result.samples[1].intended_use, ["to obtain"])
        warning = next(
            item for item in result.warnings
            if item["code"] == "unsupported_intended_uses_dropped"
        )
        self.assertEqual(
            warning["samples"],
            [{
                "sample_id": "s002",
                "values": ["invented downstream use"],
            }],
        )

    def test_invalid_sample_label_is_rejected_in_strict(self) -> None:
        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                stage0_document(),
                stage2_document(),
                UnsupportedSampleLabelClient(),
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_preview_keeps_schema_valid_semantic_mismatch(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            UnsupportedSampleLabelClient(),
            rendered_prompt(),
            max_validation_retries=0,
            preview_relaxed=True,
        )

        self.assertEqual(result.samples[1].sample_label_raw, "invented sample label")
        warning = next(
            item for item in result.warnings
            if item["code"] == "preview_semantic_validation_bypassed"
        )
        self.assertIn("sample_label_raw", warning["reason"])

    def test_preview_does_not_bypass_invalid_process_graph(self) -> None:
        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                stage0_document(),
                stage2_document(),
                CycleClient(),
                rendered_prompt(),
                max_validation_retries=0,
                preview_relaxed=True,
            )

    def test_invalid_intended_use_without_anchor_is_rejected(self) -> None:
        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                stage0_document(),
                stage2_document(),
                UnsupportedIntendedUseWithoutAnchorClient(),
                rendered_prompt(),
                max_validation_retries=0,
            )

    def test_expanded_process_type_is_preserved(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            ExpandedProcessTypeClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(
            result.process_steps[0].process_type,
            "specimen_preparation",
        )

    def test_polymer_name_is_derived_from_linked_entity(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            DerivedSampleNameClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(
            result.samples[1].polymer_name,
            "cis-polybutadiene",
        )
        self.assertEqual(
            result.samples[1].state_description,
            "dried at 60 °C for 6 h",
        )

    def test_sample_types_are_preserved_when_entity_type_is_unknown(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            SampleTypeClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertTrue(all(
            sample.polymer_type == "random_copolymer"
            for sample in result.samples
        ))
        self.assertTrue(all(
            sample.material_type == "neat_resin"
            for sample in result.samples
        ))

    def test_entity_polymer_type_overrides_conflicting_sample_type(self) -> None:
        entity_data = stage2_document().model_dump(mode="json")
        entity_data["polymer_entities"][0]["polymer_type"] = "homopolymer"
        entities = Stage2Document.model_validate(entity_data)

        result = extract_samples_processes(
            stage0_document(),
            entities,
            SampleTypeClient(polymer_type="random_copolymer"),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertTrue(all(
            sample.polymer_type == "homopolymer"
            for sample in result.samples
        ))
        warning = next(
            item for item in result.warnings
            if item["code"] == "sample_polymer_type_overridden"
        )
        self.assertEqual(len(warning["repairs"]), 2)
        self.assertEqual(warning["repairs"][0]["model_value"], "random_copolymer")
        self.assertEqual(warning["repairs"][0]["resolved_value"], "homopolymer")

    def test_case_changed_sample_name_is_mapped_to_source(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            CaseChangedSampleNameClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(
            result.samples[1].sample_label_raw,
            "dried PB film",
        )

    def test_evidence_and_parameters_are_mapped_to_source(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            FormattingChangedEvidenceClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(
            result.samples[0].evidence.source_sentence,
            METHOD_SENTENCE,
        )
        self.assertEqual(
            result.process_steps[0].parameters["time"],
            "6 h",
        )

    def test_paraphrased_evidence_is_replaced_by_anchored_source(self) -> None:
        result = extract_samples_processes(
            stage0_document(),
            stage2_document(),
            ParaphrasedEvidenceClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(
            result.samples[0].evidence.source_sentence,
            METHOD_SENTENCE,
        )
        self.assertEqual(
            result.process_steps[0].evidence.source_sentence,
            METHOD_SENTENCE,
        )

    def test_pronoun_evidence_falls_back_to_exact_block_text(self) -> None:
        document_data = stage0_document().model_dump(mode="json")
        document_data["elements"].append({
            "block_id": "P_1_1",
            "type": "text",
            "section": "Methods",
            "text": "It was then dried under vacuum.",
            "page": 1,
            "bbox": [5, 9, 7, 10],
            "source_block_index": 2,
        })
        document = Stage0Document.model_validate(document_data)

        result = extract_samples_processes(
            document,
            stage2_document(),
            PronounEvidenceClient(),
            rendered_prompt(),
            max_validation_retries=0,
        )

        self.assertEqual(
            result.samples[1].evidence.source_sentence,
            "It was then dried under vacuum.",
        )

    def test_context_falls_back_to_entity_evidence(self) -> None:
        blocks, warnings, _ = select_context_blocks(
            stage0_document(method_section="Introduction"),
            stage2_document(),
        )

        self.assertEqual(
            [block.block_id for block in blocks],
            ["P_1_0"],
        )
        self.assertEqual(warnings[0]["code"], "section_fallback")

    def test_preview_relaxed_uses_distinct_cache_key(self) -> None:
        document = stage0_document()
        entities = stage2_document()
        client = FakeClient()
        prompt = rendered_prompt()

        strict_key = _cache_components(
            document,
            entities,
            prompt,
            client,
        )[2]
        preview_key = _cache_components(
            document,
            entities,
            prompt,
            client,
            preview_relaxed=True,
        )[2]

        self.assertNotEqual(strict_key, preview_key)

    def test_compatible_output_cache_is_reused(self) -> None:
        document = stage0_document()
        entities = stage2_document()
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage2_path = root / "stage2_entities.json"
            output_path = root / "stage3_process.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage2_path.write_text(
                json.dumps(entities.model_dump(mode="json")),
                encoding="utf-8",
            )

            _, first_cached = run_stage3(
                stage0_path,
                stage2_path,
                output_path,
                client,
                rendered_prompt(),
            )
            calls_after_first = client.calls
            _, second_cached = run_stage3(
                stage0_path,
                stage2_path,
                output_path,
                client,
                rendered_prompt(),
            )

            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertEqual(client.calls, calls_after_first)

    def test_failure_response_can_be_replayed_without_network(self) -> None:
        document = stage0_document()
        entities = stage2_document()
        response = FakeClient().call_json("", "")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage0_path = root / "stage0_blocks.json"
            stage2_path = root / "stage2_entities.json"
            output_path = root / "stage3_process.json"
            failure_path = root / "stage3_failure.json"
            stage0_path.write_text(
                json.dumps(document.model_dump(mode="json")),
                encoding="utf-8",
            )
            stage2_path.write_text(
                json.dumps(entities.model_dump(mode="json")),
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
                            "output_tokens": 25,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                        "cost": {
                            "currency": "CNY",
                            "input_per_million": "13.51",
                            "output_per_million": "66.5",
                            "input_cost": "0.001351",
                            "output_cost": "0.0016625",
                            "total_cost": "0.0030135",
                        },
                    }
                }),
                encoding="utf-8",
            )
            client = _failure_replay_client(
                failure_path,
                load_pipeline_config(DEFAULT_CONFIG_PATH),
            )

            run_stage3(
                stage0_path,
                stage2_path,
                output_path,
                client,
                rendered_prompt(),
                force=True,
                max_validation_retries=0,
            )

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(client.calls, 1)
            self.assertEqual(result["provenance"]["model"], "saved-model")
            self.assertEqual(result["provenance"]["usage"]["input_tokens"], 100)
            self.assertEqual(result["provenance"]["cost"]["total_cost"], "0.0030135")
            self.assertTrue(any(
                item["code"] == "failure_response_replayed"
                for item in result["warnings"]
            ))

    def test_failure_replay_requires_saved_raw_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            failure_path = Path(temp_dir) / "stage3_failure.json"
            failure_path.write_text(
                json.dumps({"raw_response": None}),
                encoding="utf-8",
            )

            with self.assertRaises(Stage3Error):
                _failure_replay_client(
                    failure_path,
                    load_pipeline_config(DEFAULT_CONFIG_PATH),
                )

    def test_document_id_mismatch_fails(self) -> None:
        entities = stage2_document().model_copy(
            update={"document_id": "reference_no_other"}
        )

        with self.assertRaises(Stage3Error):
            extract_samples_processes(
                stage0_document(),
                entities,
                FakeClient(),
                rendered_prompt(),
            )


if __name__ == "__main__":
    unittest.main()
