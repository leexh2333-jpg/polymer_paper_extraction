import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


from schema.polymer_schema import (
    ConditionQuantity,
    Evidence,
    PropertySeries,
    ProcessStep,
    Stage0Element,
    Stage1Document,
    StageCost,
    TokenUsageSummary,
)
from stages.table_grid import parse_table_cells
from stages.stage5_characterization import (
    DEFAULT_VOCABULARY_PATH,
    extract_characterizations,
    load_characterization_vocabulary,
)
from stages.stage6_validate_merge import run_stage6, validate_and_merge
from tests.test_stage5_characterization import (
    FTIR_SENTENCE,
    FakeClient,
    UnresolvedStage4LinkClient,
    rendered_prompt,
    stage0_document,
    stage2_document,
    stage3_document,
    stage4_document,
    stage4_with_unresolved_method,
)


def stage1_document() -> Stage1Document:
    digest = "9" * 64
    return Stage1Document.model_validate({
        "schema_version": "1.0",
        "document_id": "reference_no_0000002",
        "material_mentions": [{
            "mention_id": "m001",
            "text": "PB",
            "mention_role": "abbreviation",
            "evidence": {
                "block_id": "P_2_0",
                "page": 2,
                "bbox": [5, 6, 7, 8],
                "source_type": "text",
                "source_sentence": FTIR_SENTENCE,
            },
        }],
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


def stage5_document():
    methods, vocabulary, vocabulary_hash = (
        load_characterization_vocabulary(DEFAULT_VOCABULARY_PATH)
    )
    return extract_characterizations(
        stage0_document(),
        stage2_document(),
        stage3_document(),
        stage4_document(),
        FakeClient(),
        rendered_prompt(),
        methods,
        vocabulary,
        vocabulary_hash,
    )


def all_stages():
    return (
        stage0_document(),
        stage1_document(),
        stage2_document(),
        stage3_document(),
        stage4_document(),
        stage5_document(),
    )


class Stage6Tests(unittest.TestCase):
    def test_merges_and_deduplicates_evidence(self) -> None:
        stages = list(all_stages())
        condition = stages[4].measurement_conditions[0]
        condition.temperature = ConditionQuantity.model_validate({
            "raw": "-85 °C",
            "value": -85,
            "unit": "°C",
            "evidence": [condition.evidence.model_dump(mode="python")],
        })
        condition.condition_status = "reported"

        final, validation = validate_and_merge(*stages)

        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(validation.error_count, 0)
        self.assertEqual(validation.status, "passed_with_warnings")
        self.assertEqual(
            [item.code for item in validation.warnings],
            ["paper_metadata_incomplete"],
        )
        self.assertLess(len(final.evidence), 9)
        self.assertEqual(
            len(final.property_observations),
            2,
        )
        self.assertTrue(final.material_mentions[0].evidence_ids)
        self.assertNotIn(
            "evidence",
            final.material_mentions[0].model_dump(),
        )
        self.assertEqual(len(final.provenance), 6)
        self.assertEqual(final.schema_version, "1.6")
        self.assertIsNotNone(final.characterizations[0].confidence)
        self.assertEqual(final.cost_summary.status, "partial")
        self.assertEqual(
            final.quality_metrics.stage4_methods_with_characterization.ratio,
            1.0,
        )
        self.assertIsNotNone(
            final.property_observations[0].measurement_context
        )
        temperature = final.measurement_conditions[0].temperature
        self.assertIsNotNone(temperature)
        assert temperature is not None
        self.assertEqual(temperature.evidence, [])
        self.assertTrue(temperature.evidence_ids)
        self.assertTrue(set(temperature.evidence_ids) <= {
            item.evidence_id for item in final.evidence
        })

    def test_property_series_points_are_merged_with_coverage(self) -> None:
        stages = list(all_stages())
        stage4 = stages[4].model_copy(deep=True)
        evidence = stage4.properties[0].evidence[0]
        confidence = {"score": 0.8}
        stage4.property_series = [PropertySeries.model_validate({
            "series_id": "series001",
            "sample_id": "s001",
            "entity_id": "pe001",
            "sample_resolution_status": "resolved",
            "property_name_raw": "stress",
            "measurement_context": {
                "condition_status": "not_reported",
            },
            "points": [{
                "point_id": "pt001",
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "coordinates": [{
                    "name_raw": "strain",
                    "value_raw": "10%",
                    "unit_raw": "%",
                    "evidence": evidence.model_dump(mode="python"),
                }],
                "value_raw": "3.2",
                "coverage_status": "covered",
                "measurement_context": {
                    "condition_status": "not_reported",
                },
                "evidence": [evidence.model_dump(mode="python")],
                "confidence": confidence,
            }],
            "coverage": {
                "expected": 1,
                "covered": 1,
                "missing": 0,
                "not_applicable": 0,
                "ratio": 1.0,
            },
            "evidence": [evidence.model_dump(mode="python")],
            "confidence": confidence,
        })]
        stages[4] = stage4

        final, validation = validate_and_merge(*stages)

        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(validation.error_count, 0)
        self.assertEqual(len(final.property_series), 1)
        self.assertEqual(final.property_series[0].points[0].point_id, "pt001")
        self.assertEqual(
            final.property_series[0].points[0].coordinates[0].value_raw,
            "10%",
        )
        self.assertTrue(
            final.property_series[0].points[0].coordinates[0].evidence_ids
        )
        self.assertEqual(
            final.quality_metrics.series_points_covered.ratio,
            1.0,
        )

    def test_aggregate_property_keeps_multiple_series_references(self) -> None:
        stages = list(all_stages())
        stage4 = stages[4].model_copy(deep=True)
        evidence = stage4.properties[0].evidence[0]
        confidence = {"score": 0.8}

        def series(series_id: str, point_id: str) -> PropertySeries:
            return PropertySeries.model_validate({
                "series_id": series_id,
                "sample_id": "s001",
                "entity_id": "pe001",
                "sample_resolution_status": "resolved",
                "property_name_raw": stage4.properties[0].property_name_raw,
                "observation_group_id": "pog001",
                "measurement_context": {
                    "condition_status": "not_reported",
                },
                "points": [{
                    "point_id": point_id,
                    "sample_id": "s001",
                    "entity_id": "pe001",
                    "sample_resolution_status": "resolved",
                    "coordinates": [{
                        "name_raw": "row",
                        "value_raw": point_id,
                        "evidence": evidence.model_dump(mode="python"),
                    }],
                    "value_raw": "1",
                    "coverage_status": "covered",
                    "measurement_context": {
                        "condition_status": "not_reported",
                    },
                    "evidence": [evidence.model_dump(mode="python")],
                    "confidence": confidence,
                }],
                "coverage": {
                    "expected": 1,
                    "covered": 1,
                    "missing": 0,
                    "not_applicable": 0,
                    "ratio": 1.0,
                },
                "evidence": [evidence.model_dump(mode="python")],
                "confidence": confidence,
            })

        stage4.property_series = [
            series("series001", "pt001"),
            series("series002", "pt002"),
        ]
        prop = stage4.properties[0]
        prop.observation_group_id = "pog001"
        prop.observation_role = "aggregate"
        prop.series_id = None
        prop.series_ids = ["series001", "series002"]
        stages[4] = stage4

        final, validation = validate_and_merge(*stages)

        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(validation.error_count, 0)
        self.assertEqual(
            final.property_observations[0].series_ids,
            ["series001", "series002"],
        )
        self.assertNotIn(
            "unknown_series_reference",
            [item.code for item in validation.errors],
        )

    def test_stage6_enriches_legacy_table_locator(self) -> None:
        stages = list(all_stages())
        body = (
            "<table><tr><td>Sample</td><td>Value</td></tr>"
            "<tr><td>S1</td><td>3.2</td></tr></table>"
        )
        stage0 = stages[0].model_copy(deep=True)
        stage0.elements.append(Stage0Element(
            block_id="T_2_9",
            type="table",
            page=2,
            bbox=(1, 2, 3, 4),
            source_block_index=9,
            table_body=body,
            table_cells=None,
        ))
        stage4 = stages[4].model_copy(deep=True)
        stage4.properties[0].evidence = [Evidence(
            block_id="T_2_9",
            page=2,
            bbox=(1, 2, 3, 4),
            source_type="table",
            source_sentence="3.2",
            table_locator={
                "table_id": "T_2_9",
                "row_label": "S1",
                "column_label": "Value",
                "cell_value": "3.2",
            },
        )]
        stages[0] = stage0
        stages[4] = stage4

        final, validation = validate_and_merge(*stages)

        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(validation.error_count, 0)
        property_item = next(
            item
            for item in final.property_observations
            if getattr(item, "property_id", None) == "prop001"
        )
        evidence_item = next(
            item
            for item in final.evidence
            if item.evidence_id in property_item.evidence_ids
        )
        self.assertEqual(
            evidence_item.table_locator["cell_id"],
            "T_2_9:r0001:c0001",
        )

    def test_stage6_accepts_stable_blank_table_cell_locator(self) -> None:
        stages = list(all_stages())
        body = (
            "<table><tr><td>Sample</td><td>Value</td></tr>"
            "<tr><td>S1</td><td></td></tr></table>"
        )
        stage0 = stages[0].model_copy(deep=True)
        stage0.elements.append(Stage0Element(
            block_id="T_2_9",
            type="table",
            page=2,
            bbox=(1, 2, 3, 4),
            source_block_index=9,
            table_body=body,
            table_cells=parse_table_cells(body, "T_2_9"),
        ))
        stage4 = stages[4].model_copy(deep=True)
        stage4.properties[0].evidence = [Evidence(
            block_id="T_2_9",
            page=2,
            bbox=(1, 2, 3, 4),
            source_type="table",
            source_sentence="S1",
            table_locator={
                "table_id": "T_2_9",
                "row_label": "S1",
                "column_label": "Value",
                "cell_value": None,
                "cell_id": "T_2_9:r0001:c0001",
                "row_index": 1,
                "column_index": 1,
            },
        )]
        stages[0] = stage0
        stages[4] = stage4

        final, validation = validate_and_merge(*stages)

        self.assertIsNotNone(final)
        self.assertNotIn(
            "invalid_table_locator",
            [item.code for item in validation.errors],
        )
        self.assertNotIn(
            "table_locator_stable_cell_unresolved",
            [item.code for item in validation.warnings],
        )

    def test_stage6_rejects_unstable_blank_table_cell_locator(self) -> None:
        stages = list(all_stages())
        body = (
            "<table><tr><td>Sample</td><td>Value</td></tr>"
            "<tr><td>S1</td><td></td></tr></table>"
        )
        stage0 = stages[0].model_copy(deep=True)
        stage0.elements.append(Stage0Element(
            block_id="T_2_9",
            type="table",
            page=2,
            bbox=(1, 2, 3, 4),
            source_block_index=9,
            table_body=body,
            table_cells=None,
        ))
        stage4 = stages[4].model_copy(deep=True)
        stage4.properties[0].evidence = [Evidence(
            block_id="T_2_9",
            page=2,
            bbox=(1, 2, 3, 4),
            source_type="table",
            source_sentence="S1",
            table_locator={
                "table_id": "T_2_9",
                "row_label": "S1",
                "column_label": "Value",
                "cell_value": None,
            },
        )]
        stages[0] = stage0
        stages[4] = stage4

        _, validation = validate_and_merge(*stages)

        self.assertIn(
            "invalid_table_locator",
            [item.code for item in validation.errors],
        )

    def test_calculated_stage_cost_is_included_in_total(self) -> None:
        stages = list(all_stages())
        stage1 = stages[1].model_copy(deep=True)
        stage1.provenance.usage = TokenUsageSummary.model_validate({
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "billable_input_tokens": 1000,
            "total_tokens": 1100,
        })
        stage1.provenance.cost = StageCost.model_validate({
            "status": "calculated",
            "currency": "CNY",
            "input_per_million": "2",
            "output_per_million": "10",
            "input_cost": "0.002",
            "output_cost": "0.001",
            "total_cost": "0.003",
        })
        stages[1] = stage1

        final, _ = validate_and_merge(*stages)

        assert final is not None
        self.assertEqual(final.cost_summary.total_cost, Decimal("0.003"))
        stage_cost = next(
            item
            for item in final.cost_summary.stages
            if item.stage == "stage1_material_mention"
        )
        self.assertEqual(stage_cost.cost.status, "calculated")

    def test_invalid_evidence_blocks_publication(self) -> None:
        stages = list(all_stages())
        bad_stage1 = stages[1].model_copy(deep=True)
        bad_stage1.material_mentions[0].evidence.source_sentence = (
            "not present in source"
        )
        stages[1] = bad_stage1

        final, validation = validate_and_merge(*stages)

        self.assertIsNone(final)
        self.assertIn(
            "evidence_not_in_source",
            [item.code for item in validation.errors],
        )

    def test_ocr_not_done_blocks_publication(self) -> None:
        stages = list(all_stages())
        bad_stage0 = stages[0].model_copy(deep=True)
        bad_stage0.ocr["status"] = "failed"
        stages[0] = bad_stage0

        final, validation = validate_and_merge(*stages)

        self.assertIsNone(final)
        self.assertIn(
            "ocr_not_done",
            [item.code for item in validation.errors],
        )

    def test_orphan_sample_is_warning(self) -> None:
        stages = list(all_stages())
        stage3 = stages[3].model_copy(deep=True)
        stage3.samples.append(
            stage3.samples[0].model_copy(update={"sample_id": "s002"})
        )
        stages[3] = stage3

        final, validation = validate_and_merge(*stages)

        self.assertIsNotNone(final)
        self.assertIn(
            "orphan_sample",
            [item.code for item in validation.warnings],
        )

    def test_polymer_name_contamination_is_warning(self) -> None:
        stages = list(all_stages())
        stage3 = stages[3].model_copy(deep=True)
        stage3.samples[0].polymer_name = "dried PB film"
        stages[3] = stage3

        final, validation = validate_and_merge(*stages)

        self.assertIsNotNone(final)
        self.assertIn(
            "polymer_name_contamination",
            [item.code for item in validation.warnings],
        )

    def test_generic_process_and_unsupported_parameter_are_warnings(self) -> None:
        stages = list(all_stages())
        stage3 = stages[3].model_copy(deep=True)
        stage3.process_steps.append(ProcessStep(
            step_id="ps001",
            process_type="other",
            input_sample_ids=[],
            output_sample_ids=["s001"],
            parameters={"temperature": "333 K"},
            evidence=stage3.samples[0].evidence,
        ))
        stages[3] = stage3

        final, validation = validate_and_merge(*stages)

        self.assertIsNotNone(final)
        assert final is not None
        codes = [item.code for item in validation.warnings]
        self.assertIn("generic_process_type", codes)
        self.assertIn("unsupported_specific_value", codes)
        self.assertEqual(
            final.quality_metrics.standard_process_steps.ratio,
            0.0,
        )

    def test_missing_characterization_link_is_warning_and_metric(self) -> None:
        stages = list(all_stages())
        stage5 = stages[5].model_copy(deep=True)
        stage5.characterizations[1].derived_property_ids = []
        stages[5] = stage5

        final, validation = validate_and_merge(*stages)

        self.assertIsNotNone(final)
        assert final is not None
        self.assertIn(
            "missing_characterization",
            [item.code for item in validation.warnings],
        )
        self.assertEqual(
            final.quality_metrics.stage4_methods_with_characterization.ratio,
            0.0,
        )

    def test_unresolved_property_can_be_characterization_derivation(
        self,
    ) -> None:
        methods, vocabulary, vocabulary_hash = (
            load_characterization_vocabulary(DEFAULT_VOCABULARY_PATH)
        )
        stage4 = stage4_with_unresolved_method()
        stage5 = extract_characterizations(
            stage0_document(),
            stage2_document(),
            stage3_document(),
            stage4,
            UnresolvedStage4LinkClient(),
            rendered_prompt(),
            methods,
            vocabulary,
            vocabulary_hash,
        )

        final, validation = validate_and_merge(
            stage0_document(),
            stage1_document(),
            stage2_document(),
            stage3_document(),
            stage4,
            stage5,
        )

        self.assertIsNotNone(final)
        assert final is not None
        self.assertNotIn(
            "unknown_property_reference",
            [item.code for item in validation.errors],
        )
        self.assertIn(
            "uprop001",
            final.characterizations[0].derived_property_ids,
        )

    def test_output_is_deterministic_and_ocr_is_whitelisted(self) -> None:
        stages = list(all_stages())
        stage0 = stages[0].model_copy(deep=True)
        stage0.ocr["api_key"] = "must-not-propagate"
        stage0.ocr["download_url"] = "https://example.test/private"
        stages[0] = stage0

        first, first_validation = validate_and_merge(*stages)
        second, second_validation = validate_and_merge(*stages)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(
            first.model_dump(mode="json"),
            second.model_dump(mode="json"),
        )
        self.assertEqual(
            first_validation.model_dump(mode="json"),
            second_validation.model_dump(mode="json"),
        )
        serialized = json.dumps(first.provenance)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("download_url", serialized)

    def test_invalid_input_removes_stale_final(self) -> None:
        stages = all_stages()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / f"stage{index}.json" for index in range(6)]
            for path, model in zip(paths, stages):
                path.write_text(
                    json.dumps(model.model_dump(mode="json")),
                    encoding="utf-8",
                )
            paths[1].write_text("{}", encoding="utf-8")
            validation_path = root / "stage6_validation.json"
            final_path = root / "final.json"
            final_path.write_text('{"stale": true}', encoding="utf-8")

            validation, published = run_stage6(
                "reference_no_0000002",
                *paths,
                validation_path,
                final_path,
            )

            self.assertFalse(published)
            self.assertEqual(validation.status, "failed")
            self.assertTrue(validation_path.is_file())
            self.assertFalse(final_path.exists())

    def test_run_stage6_writes_null_doi_and_html_report(self) -> None:
        stages = list(all_stages())
        stage0 = stages[0].model_copy(deep=True)
        stage0.elements.append(Stage0Element(
            block_id="I_2_10",
            type="image",
            page=2,
            source_block_index=10,
            caption="Fig. 1. Test image",
            image_path="images/fig1.png",
            image_kind="chart",
        ))
        stages[0] = stage0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / f"stage{index}.json" for index in range(6)]
            for path, model in zip(paths, stages):
                path.write_text(
                    json.dumps(model.model_dump(mode="json")),
                    encoding="utf-8",
                )
            validation_path = root / "stage6_validation.json"
            final_path = root / "final.json"

            _, published = run_stage6(
                "reference_no_0000002",
                *paths,
                validation_path,
                final_path,
            )
            written = json.loads(final_path.read_text(encoding="utf-8"))
            report = (root / "report.html").read_text(encoding="utf-8")

        self.assertTrue(published)
        self.assertIn("doi", written["paper"])
        self.assertIsNone(written["paper"]["doi"])
        self.assertIn('"cost_summary"', json.dumps(written))
        self.assertEqual(
            len(written["property_observations"]),
            len(stages[4].properties) + len(stages[5].properties),
        )
        self.assertEqual(
            len(written["property_series"]),
            len(stages[4].property_series),
        )
        if written["property_series"]:
            self.assertEqual(
                len(written["property_series"][0]["points"]),
                len(stages[4].property_series[0].points),
            )
        self.assertIn("unresolved_property_observations", written)
        self.assertIn('id="graph"', report)
        self.assertIn('id="viewMode"', report)
        self.assertIn('id="confidenceFilter"', report)
        self.assertIn("Confidence 覆盖率", report)
        self.assertIn("appendHighlightedText", report)
        self.assertIn("reference_no_0000002", report)
        self.assertIn("fig001", report)
        self.assertIn("Fig. 1. Test image", report)


class Stage6PreviewTests(unittest.TestCase):
    """--preview-relaxed 只降级「表示层」问题，语义问题照旧判错。"""

    TABLE_BODY = (
        "<table>"
        "<tr><td>Sample</td><td>Td5 (°C)</td><td>Td50 (°C)</td></tr>"
        "<tr><td>PC-1</td><td>394</td><td>446</td></tr>"
        "</table>"
    )

    def _stages_with_table_evidence(self, locator, source_sentence):
        stages = list(all_stages())
        stage0 = stages[0].model_copy(deep=True)
        stage0.elements.append(Stage0Element(
            block_id="T_2_9",
            type="table",
            page=2,
            bbox=(1, 2, 3, 4),
            source_block_index=9,
            table_body=self.TABLE_BODY,
            table_cells=parse_table_cells(self.TABLE_BODY, "T_2_9"),
        ))
        stage4 = stages[4].model_copy(deep=True)
        stage4.properties[0].evidence = [Evidence(
            block_id="T_2_9",
            page=2,
            bbox=(1, 2, 3, 4),
            source_type="table",
            source_sentence=source_sentence,
            table_locator=locator,
        )]
        stages[0] = stage0
        stages[4] = stage4
        return stages

    def _codes(self, issues):
        return [issue.code for issue in issues]

    def test_pipe_rendered_row_rejected_by_strict_accepted_by_preview(self) -> None:
        """Stage 4R 写 "PC-1 | 394 | 446"，Stage 0 存的是 HTML。"""
        locator = {
            "table_id": "T_2_9",
            "row_label": "PC-1",
            "column_label": "Td50 (°C)",
            "cell_value": "446",
            "cell_id": "T_2_9:r0001:c0002",
            "row_index": 1,
            "column_index": 2,
        }
        stages = self._stages_with_table_evidence(locator, "PC-1 | 394 | 446")

        _, strict = validate_and_merge(*stages)
        self.assertIn("evidence_not_in_source", self._codes(strict.errors))

        final, preview = validate_and_merge(*stages, preview=True)
        self.assertIsNotNone(final)
        self.assertNotIn("evidence_not_in_source", self._codes(preview.errors))
        self.assertTrue(
            any(code.startswith("evidence_matched")
                for code in self._codes(preview.warnings))
        )

    def test_preview_degrades_missing_row_label_when_cell_id_locates(self) -> None:
        """表格首列为空时模型只能写 null，但 cell_id 仍能确定性定位。"""
        locator = {
            "table_id": "T_2_9",
            "row_label": None,
            "column_label": "Td50 (°C)",
            "cell_value": "446",
            "cell_id": "T_2_9:r0001:c0002",
            "row_index": 1,
            "column_index": 2,
        }
        stages = self._stages_with_table_evidence(locator, "446")

        _, strict = validate_and_merge(*stages)
        self.assertIn("invalid_table_locator", self._codes(strict.errors))

        final, preview = validate_and_merge(*stages, preview=True)
        self.assertIsNotNone(final)
        self.assertNotIn("invalid_table_locator", self._codes(preview.errors))
        self.assertIn("table_locator_label_missing", self._codes(preview.warnings))

    def test_preview_still_rejects_locator_pointing_at_wrong_cell(self) -> None:
        """cell_id 指向的格子里是 394，locator 却声明 446 —— 这是真错。"""
        locator = {
            "table_id": "T_2_9",
            "row_label": None,
            "column_label": "Td50 (°C)",
            "cell_value": "446",
            "cell_id": "T_2_9:r0001:c0001",
            "row_index": 1,
            "column_index": 1,
        }
        stages = self._stages_with_table_evidence(locator, "446")

        _, preview = validate_and_merge(*stages, preview=True)
        self.assertIn("invalid_table_locator", self._codes(preview.errors))

    def test_preview_still_rejects_fabricated_sentence(self) -> None:
        """内容根本不在该 block 里，preview 也不能放行。"""
        stages = list(all_stages())
        stage4 = stages[4].model_copy(deep=True)
        stage4.properties[0].evidence = [Evidence(
            block_id="P_2_0",
            page=2,
            bbox=(1, 2, 3, 4),
            source_type="text",
            source_sentence=(
                "The tensile strength reached 133 MPa after annealing at 250 °C "
                "for 300 minutes in vacuum"
            ),
        )]
        stages[4] = stage4

        _, preview = validate_and_merge(*stages, preview=True)
        self.assertIn("evidence_not_in_source", self._codes(preview.errors))

    def test_preview_metadata_written_to_final_json(self) -> None:
        locator = {
            "table_id": "T_2_9",
            "row_label": "PC-1",
            "column_label": "Td50 (°C)",
            "cell_value": "446",
            "cell_id": "T_2_9:r0001:c0002",
            "row_index": 1,
            "column_index": 2,
        }
        stages = self._stages_with_table_evidence(locator, "PC-1 | 394 | 446")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / f"stage{index}.json" for index in range(6)]
            for path, model in zip(paths, stages):
                path.write_text(
                    json.dumps(model.model_dump(mode="json")), encoding="utf-8"
                )
            final_path = root / "final.json"

            _, published = run_stage6(
                "reference_no_0000002",
                *paths,
                root / "stage6_validation.json",
                final_path,
                preview=True,
            )
            written = json.loads(final_path.read_text(encoding="utf-8"))
            self.assertTrue((root / "report.html").is_file())

        self.assertTrue(published)
        self.assertEqual(written["validation_mode"], "preview")
        self.assertEqual(
            written["validation_summary"]["validation_status"], "degraded"
        )
        self.assertTrue(written["validation_summary"]["degraded_codes"])

    def test_strict_run_carries_no_preview_metadata(self) -> None:
        stages = list(all_stages())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / f"stage{index}.json" for index in range(6)]
            for path, model in zip(paths, stages):
                path.write_text(
                    json.dumps(model.model_dump(mode="json")), encoding="utf-8"
                )
            final_path = root / "final.json"
            _, published = run_stage6(
                "reference_no_0000002",
                *paths,
                root / "stage6_validation.json",
                final_path,
            )
            written = json.loads(final_path.read_text(encoding="utf-8"))

        self.assertTrue(published)
        self.assertNotIn("validation_mode", written)


if __name__ == "__main__":
    unittest.main()
