from __future__ import annotations

import json
from pathlib import Path

from reports.render_extraction_html import (
    MATHJAX_VERSION,
    _load_optional_stage0,
    render_extraction_html,
)


def _final_document() -> dict:
    return {
        "document_id": "ref",
        "paper": {
            "ref_no": "ref",
            "title": r"Property of $\delta_p$",
            "doi": None,
            "year": 2026,
            "journal": "Test",
        },
        "material_mentions": [],
        "polymer_entities": [],
        "samples": [{
            "sample_id": "s001",
            "sample_kind": "reported",
            "refers_to_entity": "pe001",
            "sample_label_raw": "Sample A",
            "evidence_ids": ["ev001"],
            "confidence": {
                "score": 0.7,
                "field_scores": {},
                "uncertain_fields": ["state_description"],
                "evidence_basis": ["explicit_text"],
                "uncertainty_codes": ["condition_missing"],
            },
        }],
        "process_steps": [],
        "property_observations": [{
            "property_id": "prop001",
            "sample_id": "s001",
            "property_name_raw": r"$\delta_p$",
            "value_raw": r"$35 \pm 0.01^{\circ}\mathrm{C}$",
            "evidence_ids": ["ev001"],
            "confidence": {"score": 0.6},
        }],
        "measurement_conditions": [],
        "characterizations": [],
        "evidence": [{
            "evidence_id": "ev001",
            "block_id": "T_1_1",
            "page": 1,
            "bbox": None,
            "source_type": "text",
            "source_sentence": r"The value is $35 \pm 0.01^{\circ}\mathrm{C}$.",
        }, {
            "evidence_id": "ev002",
            "block_id": "T_1_2",
            "page": 1,
            "bbox": None,
            "source_type": "table",
            "source_sentence": (
                "<table><tr><th>A</th><th>B</th></tr>"
                "<tr><td rowspan=\"2\">x</td><td>1</td></tr></table>"
            ),
            "table_locator": {
                "table_id": "T_1_2",
                "cell_id": "T_1_2:r0001:c0001",
                "row_index": 1,
                "column_index": 1,
                "cell_value": "1",
            },
        }],
        "warnings": [],
        "cost_summary": None,
        "quality_metrics": {},
    }


def _stage0_document(image_path: str) -> dict:
    return {
        "elements": [{
            "block_id": "I_1_2",
            "type": "image",
            "page": 1,
            "bbox": [1, 2, 3, 4],
            "source_block_index": 2,
            "caption": r"Fig. 1. Plot of $\delta_p$",
            "image_path": image_path,
            "image_kind": "chart",
        }]
    }


def test_render_report_uses_local_mathjax_and_readable_detail(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    image = project_root / "wenxian" / "ref" / "images" / "fig 1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    output = project_root / "output" / "ref" / "report.html"

    result = render_extraction_html(
        _final_document(),
        output,
        stage0_data=_stage0_document(
            "wenxian/ref/images/fig 1.jpg"
        ),
        project_root=project_root,
    )

    html = result.read_text(encoding="utf-8")
    mathjax = (
        project_root
        / "output"
        / "_assets"
        / f"mathjax-{MATHJAX_VERSION}"
        / "tex-svg.js"
    )
    assert result == output.resolve()
    assert mathjax.is_file()
    assert (
        f'../_assets/mathjax-{MATHJAX_VERSION}/tex-svg.js'
        in html
    )
    assert "https://" not in html
    assert "MathJax.typesetClear([container])" in html
    assert 'overview:"概览"' in html
    assert (
        "function sampleLabel(item){return item.polymer_name||item.sample_kind}"
        in html
    )
    assert (
        'if(["polymer_type","copolymer_type","material_type"].includes(field)&&!hasValue(value))value="not specified";'
        in html
    )
    assert 'function nodeTypeSummary(node)' in html
    assert 'node.raw.polymer_type||"not specified"' in html
    assert 'node.raw.material_type||"not specified"' in html
    assert 'subtype.setAttribute("class","node-subtype")' in html
    assert 'raw:"原始 JSON"' in html
    assert "图注文本，未分析图像内容" in html
    assert "fig001" in html
    assert "fig%201.jpg" in html
    assert 'create("pre","raw-json mathjax-ignore",raw)' in html
    assert 'new DOMParser().parseFromString(source,"text/html")' in html
    assert 'cell.textContent=sourceCell.textContent||""' in html
    assert 'cell.classList.add("locator-cell")' in html
    assert "const tableGroups=new Map(),plain=[]" in html
    assert 'group.length===1?evidence.evidence_id:`${group.length} 条 evidence`' in html
    assert "表格证据 · 高亮单元格为 locator 定位" in html
    assert 'section.append(create("h4","","模型置信度"))' in html
    assert 'confidence.field_scores' not in html
    assert 'confidence.uncertain_fields' not in html


def test_render_report_keeps_null_doi_in_embedded_json(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.html"

    render_extraction_html(_final_document(), output)

    html = output.read_text(encoding="utf-8")
    assert '"doi":null' in html
    assert 'paper.doi===null?"null":text(paper.doi)' in html
    assert r"$35 \\pm 0.01^{\\circ}\\mathrm{C}$" in html


def test_load_optional_stage0_discovers_sibling_file(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "final.json"
    stage0_path = tmp_path / "stage0_blocks.json"
    final_path.write_text("{}", encoding="utf-8")
    expected = {"document_id": "ref", "elements": []}
    stage0_path.write_text(
        json.dumps(expected, ensure_ascii=False),
        encoding="utf-8",
    )

    assert _load_optional_stage0(final_path, None) == expected


def test_report_projects_collapsed_groups_and_unresolved_properties(
    tmp_path: Path,
) -> None:
    document = _final_document()
    document["material_mentions"] = [{
        "mention_id": "m001",
        "text": "Polymer A",
        "mention_role": "polymer_name",
        "evidence_ids": ["ev001"],
        "confidence": {"score": 0.92},
    }]
    document["polymer_entities"] = [{
        "entity_id": "pe001",
        "polymer_name": "Polymer A",
        "resolved_from_mentions": ["m001"],
        "evidence_ids": ["ev001"],
        "confidence": {"score": 0.9},
    }]
    document["samples"] = [
        {
            "sample_id": f"s{index:03d}",
            "sample_kind": "reported",
            "refers_to_entity": "pe001",
            "sample_label_raw": f"S{index}",
            "evidence_ids": ["ev001"],
            "confidence": {"score": 0.8},
        }
        for index in range(1, 5)
    ]
    document["unresolved_property_observations"] = [{
        "unresolved_id": "uprop001",
        "entity_id": "pe001",
        "property_name_raw": r"\delta_p",
        "value_raw": "8.55",
        "unit_raw": r"(cal/ml)^{1/2}",
        "reason": "sample_ambiguous",
        "evidence_ids": ["ev001"],
        "confidence": {"score": 0.5},
    }]
    output = tmp_path / "report.html"

    render_extraction_html(document, output)

    html = output.read_text(encoding="utf-8")
    assert '"unresolved_id":"uprop001"' in html
    assert 'addNode(item.unresolved_id,"unresolved_property"' in html
    assert "`mention-group-${entity.entity_id}`" in html
    assert "`sample-group-${entity.entity_id}`" in html
    assert 'lanes:[["mention_group","mention"],["entity"]' in html
    assert '"subject-sample"' in html
    assert '"subject-entity"' in html
    assert "实线：具体 Sample 已解析" in html
    assert "虚线：仅关联 PolymerEntity" in html
    assert '<option value="science">科学链条（默认）</option>' in html
    assert 'addProjected(entityId,process.id,"projected-material-process",ids)' in html
    assert 'addProjected(process.id,measurement.id,"projected-process-result",[sampleId])' in html
    assert "由底层 Sample 关系计算，仅用于阅读，不代表因果关系" in html
    assert "PDF 第 ${page+1} 页 · index ${page}" in html


def test_report_keeps_series_points_in_detail_and_wraps_plain_tex(
    tmp_path: Path,
) -> None:
    document = _final_document()
    document["process_steps"] = [{
        "step_id": "ps001",
        "process_type": "mixing",
        "input_sample_ids": ["s001"],
        "output_sample_ids": [],
        "parameters": {
            "mixing temperature": r"35 \pm 5^{\circ}C",
        },
        "evidence_ids": ["ev001"],
        "confidence": {"score": 0.8},
    }]
    document["measurement_conditions"] = [{
        "condition_id": "mc001",
        "solvent": "toluene",
        "condition_status": "reported",
        "evidence_ids": ["ev001"],
        "confidence": {"score": 0.8},
    }]
    document["property_series"] = [{
        "series_id": "series001",
        "entity_id": "pe001",
        "property_name_raw": r"\chi",
        "measurement_condition_id": "mc001",
        "measurement_context": {
            "temperature": {
                "raw": r"35 \pm 0.01^{\circ}C",
                "evidence_ids": ["ev001"],
            },
            "condition_status": "reported",
        },
        "points": [{
            "point_id": "point001",
            "observation_role": "series_point",
            "value_raw": "0.38",
            "coverage_status": "covered",
            "measurement_context": {
                "temperature": {
                    "raw": r"40 \pm 0.01^{\circ}C",
                    "evidence_ids": ["ev001"],
                },
                "condition_status": "reported",
            },
            "table_locator": {
                "cell_id": "table3-r1-c2",
                "row_index": 1,
                "column_index": 2,
            },
            "evidence_ids": ["ev001"],
            "confidence": {"score": 0.75},
        }],
        "evidence_ids": ["ev001"],
        "confidence": {"score": 0.7},
    }]
    output = tmp_path / "report.html"

    render_extraction_html(document, output)

    html = output.read_text(encoding="utf-8")
    assert '"series_id":"series001"' in html
    assert '"observation_role":"series_point"' in html
    assert '"cell_id":"table3-r1-c2"' in html
    assert 'addNode(item.series_id,"series"' in html
    assert '${scope}（${points.length} 行数据）' in html
    assert "Series 数据点（${points.length}）" in html
    assert 'header.append(create("th","","有效测量条件"),create("th","","Confidence"),create("th","","证据"))' in html
    assert "points.flatMap(point=>point.evidence_ids||[])" in html
    assert r"String.raw`^{\circ}\mathrm{C}`" in html
    assert 'mergeConditionLayer(values,sources,otherEvidenceIds,legacy,"旧 MeasurementCondition")' in html
    assert 'mergeConditionLayer(values,sources,otherEvidenceIds,series.measurement_context,"Series 继承")' in html
    assert 'series?"point 自有":"对象自身"' in html
    assert 'value.evidence_ids' in html
    assert 'layer.other_condition_evidence_ids' in html
    assert 'owner&&owner.evidence_ids' not in html
    assert "未提供测量条件；不从其他对象推断或补写。" in html


def test_report_labels_multi_subject_series(tmp_path: Path) -> None:
    document = _final_document()
    document["property_series"] = [{
        "series_id": "series001",
        "sample_id": None,
        "entity_id": None,
        "property_name_raw": "Tg",
        "points": [
            {"point_id": "pt001", "sample_id": "s001"},
            {"point_id": "pt002", "sample_id": "s002"},
        ],
        "evidence_ids": ["ev001"],
    }]
    output = tmp_path / "report.html"

    render_extraction_html(document, output)

    html = output.read_text(encoding="utf-8")
    assert '"sample_id":"s002"' in html
    assert '?"跨主体序列":"序列"' in html


def test_report_links_aggregate_to_multiple_series(tmp_path: Path) -> None:
    document = _final_document()
    document["property_observations"][0].update({
        "observation_role": "aggregate",
        "series_ids": ["series001", "series002"],
    })
    document["property_series"] = [
        {
            "series_id": series_id,
            "sample_id": "s001",
            "property_name_raw": r"\chi",
            "points": [],
            "evidence_ids": ["ev001"],
        }
        for series_id in ("series001", "series002")
    ]
    output = tmp_path / "report.html"

    render_extraction_html(document, output)

    html = output.read_text(encoding="utf-8")
    assert '"series_ids":["series001","series002"]' in html
    assert 'series_ids:"Series IDs"' in html
    assert 'edge(item.property_id,id,"aggregate of")' in html
    assert '"aggregate of":"汇总自 Series"' in html


def test_report_links_characterization_to_multiple_series(tmp_path: Path) -> None:
    document = _final_document()
    document["characterizations"] = [{
        "characterization_id": "char001",
        "method_raw": "DSC",
        "method_normalized": "DSC",
        "sample_id": "s001",
        "sample_resolution_status": "resolved",
        "series_ids": ["series001", "series002"],
        "derived_property_ids": [],
        "evidence_ids": ["ev001"],
    }]
    output = tmp_path / "report.html"

    render_extraction_html(document, output)

    html = output.read_text(encoding="utf-8")
    assert '"series_ids":["series001","series002"]' in html
    assert 'edge(item.characterization_id,id,"characterizes")' in html
    assert 'characterization:["method_raw","method_normalized","sample_id","entity_id","sample_ids","entity_ids","sample_resolution_status","series_id","series_ids"' in html


def test_report_uses_full_stage0_table_for_fragment_evidence(
    tmp_path: Path,
) -> None:
    document = _final_document()
    document["evidence"][1]["source_sentence"] = "<td>1</td></tr>"
    document["property_observations"][0]["evidence_ids"] = ["ev002"]
    output = tmp_path / "report.html"
    table_body = (
        "<table><tr><th>A</th><th>B</th></tr>"
        "<tr><td>x</td><td>1</td></tr></table>"
    )

    render_extraction_html(
        document,
        output,
        stage0_data={"elements": [{
            "block_id": "T_1_2",
            "type": "table",
            "table_body": table_body,
        }]},
    )

    html = output.read_text(encoding="utf-8")
    assert '"table_sources":{"T_1_2":"\\u003ctable\\u003e' in html
    assert "fullTable||evidence.source_sentence" in html
    assert 'const key=fullTable?item.block_id' in html
    assert 'point_id:"内部点 ID"' in html
    assert "论文中的对应项请看坐标、行/列索引和单元格 ID" in html


def test_report_labels_not_reported_condition_as_placeholder(
    tmp_path: Path,
) -> None:
    document = _final_document()
    document["measurement_conditions"] = [{
        "condition_id": "mc001",
        "condition_status": "not_reported",
        "evidence_ids": [],
        "confidence": {"score": 0.5},
    }]
    output = tmp_path / "report.html"

    render_extraction_html(document, output)

    html = output.read_text(encoding="utf-8")
    assert '"条件占位（原文未报告）"' in html


def test_candidate_report_displays_unvalidated_banner(tmp_path: Path) -> None:
    document = _final_document()
    document["publication"] = {
        "kind": "candidate",
        "status": "unvalidated",
        "message": "未经完整科学语义校验，仅用于预览模型抽取结果。",
    }
    output = tmp_path / "report_candidate.html"

    render_extraction_html(document, output)

    html = output.read_text(encoding="utf-8")
    assert "候选结果 · 未经完整科学语义校验" in html
    assert 'class="publication-banner"' in html
