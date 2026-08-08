from __future__ import annotations

from pathlib import Path

from reports.report_view_model import build_figure_groups, build_report_context


def _image(
    block_id: str,
    source_block_index: int,
    *,
    page: int = 4,
    caption: str | None = None,
    image_path: str | None = None,
) -> dict:
    return {
        "block_id": block_id,
        "type": "image",
        "page": page,
        "bbox": [10, 20, 100, 120],
        "source_block_index": source_block_index,
        "caption": caption,
        "image_path": image_path,
        "image_kind": "chart",
    }


def test_build_figure_groups_groups_panels_until_caption(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    report_dir = project_root / "code" / "extraction" / "output" / "ref"
    image_dir = project_root / "wenxian" / "ref" / "images"
    image_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    for name in ("panel a.jpg", "panel-b.jpg", "panel-c.jpg"):
        (image_dir / name).write_bytes(b"image")

    stage0 = {
        "elements": [
            _image(
                "I_4_1",
                1,
                image_path="wenxian/ref/images/panel a.jpg",
            ),
            _image(
                "I_4_2",
                2,
                image_path="wenxian/ref/images/panel-b.jpg",
            ),
            _image(
                "I_4_3",
                3,
                caption=r"Fig. 1. Value of $\delta_p$",
                image_path="wenxian/ref/images/panel-c.jpg",
            ),
            {"block_id": "T_4_4", "type": "text", "source_block_index": 4},
            _image(
                "I_4_5",
                5,
                image_path="wenxian/ref/images/missing.jpg",
            ),
        ]
    }

    figures, warnings = build_figure_groups(
        stage0,
        project_root=project_root,
        report_dir=report_dir,
    )

    assert [item["image_block_ids"] for item in figures] == [
        ["I_4_1", "I_4_2", "I_4_3"],
        ["I_4_5"],
    ]
    assert figures[0]["label"] == "Fig. 1"
    assert figures[0]["association_status"] == "grouped_by_adjacency"
    assert figures[0]["images"][0]["image_url"].endswith("panel%20a.jpg")
    assert figures[0]["images"][0]["image_exists"] is True
    assert figures[1]["association_status"] == "caption_missing"
    assert figures[1]["images"][0]["image_exists"] is False
    assert [item["object_id"] for item in warnings] == ["fig002"]


def test_build_report_context_only_links_explicit_block_references(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    report_dir = project_root / "output" / "ref"
    report_dir.mkdir(parents=True)
    stage0 = {
        "elements": [
            _image(
                "I_1_1",
                1,
                page=1,
                caption="Fig. 2. Polymer structure",
            )
        ]
    }
    final = {
        "polymer_entities": [{
            "entity_id": "pe001",
            "evidence_ids": ["ev001"],
            "source_image_refs": [{"block_id": "I_1_1"}],
        }],
        "property_observations": [{
            "property_id": "prop001",
            "evidence_ids": ["ev001"],
        }],
        "evidence": [{
            "evidence_id": "ev001",
            "block_id": "I_1_1",
            "source_type": "image",
        }],
    }

    context = build_report_context(
        final,
        stage0,
        project_root=project_root,
        report_dir=report_dir,
    )

    figure = context["figures"][0]
    assert context["evidence_to_figure"] == {"ev001": "fig001"}
    assert figure["linked_evidence_ids"] == ["ev001"]
    assert figure["linked_object_ids"] == ["pe001", "prop001"]
    assert figure["link_basis"] == ["image_evidence", "source_image_ref"]


def test_build_report_context_includes_stage0_table_sources(
    tmp_path: Path,
) -> None:
    table_body = "<table><tr><td>A</td></tr></table>"
    context = build_report_context(
        {"evidence": []},
        {"elements": [{
            "block_id": "T_1_1",
            "type": "table",
            "table_body": table_body,
        }]},
        project_root=tmp_path,
        report_dir=tmp_path,
    )

    assert context["table_sources"] == {"T_1_1": table_body}


def test_build_figure_groups_rejects_paths_outside_project(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    report_dir = project_root / "output"
    report_dir.mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"image")
    stage0 = {
        "elements": [
            _image(
                "I_1_1",
                1,
                page=1,
                caption="Fig. 1. Outside",
                image_path=str(outside),
            )
        ]
    }

    figures, _ = build_figure_groups(
        stage0,
        project_root=project_root,
        report_dir=report_dir,
    )

    assert figures[0]["images"][0]["image_url"] is None
    assert figures[0]["images"][0]["image_exists"] is False
