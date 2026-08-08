"""高分子抽取 Pipeline 的 Pydantic 数据模型。"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    field_validator,
    model_validator,
)


MetadataStatus = Literal["complete", "partial", "failed"]
MentionRole = Literal[
    "polymer_name",
    "abbreviation",
    "sample_label",
    "commercial_name",
]
PolymerType = Literal[
    "homopolymer",
    "random_copolymer",
    "block_copolymer",
    "graft_copolymer",
    "crosslinked_network",
    "blend",
]
StructuralFeatureTag = Literal[
    "sulfonic_acid_group",
    "aryl_ether_ketone_backbone",
    "naphthalene_moiety",
]
SampleKind = Literal[
    "synthesis_batch",
    "commercial_batch",
    "intermediate",
    "processed_material",
    "conditioned_state",
    "test_specimen",
    "post_test_state",
]
ProcessType = Literal[
    "polymerization",
    "copolymerization",
    "blending",
    "compounding",
    "mixing",
    "casting",
    "film_formation",
    "extrusion",
    "molding",
    "pressing",
    "ion_exchange",
    "annealing",
    "hydration",
    "drying",
    "fractionation",
    "purification",
    "reprecipitation",
    "solvent_extraction",
    "washing",
    "sulfonation",
    "crosslinking",
    "hot_pressing",
    "electrospinning",
    "specimen_preparation",
    "cutting",
    "punching",
    "coating",
    "surface_modification",
    "plasma_treatment",
    "other",
]
PropertyCategory = Literal[
    "physical_property",
    "optical_property",
    "thermal_property",
    "electrical_property",
    "physicochemical_property",
    "dilute_solution_property",
    "rheological_property",
    "tensile_property",
    "shear_property",
    "flexural_property",
    "compression_characteristic",
    "creep_characteristic",
    "heat_characteristic",
    "impact_strength",
    "hardness",
    "heat_resistance_and_combustion",
    "other_property",
]
Stage5PropertyCategory = Literal[
    "composition_structure",
    "morphology",
]
SampleResolutionStatus = Literal["resolved", "unresolved"]
Stage5SubjectResolutionStatus = Literal[
    "resolved", "unresolved", "multi_resolved"
]
ObservationRole = Literal["single", "series_point", "aggregate"]
SeriesReferenceID = Annotated[str, Field(pattern=r"^series\d{3,}$")]
CoverageStatus = Literal["covered", "missing", "not_applicable"]
Stage0ElementType = Literal[
    "text",
    "title",
    "table",
    "image",
    "equation",
    "footnote",
]
ConfidenceEvidenceBasis = Literal[
    "explicit_text",
    "exact_evidence_span",
    "table_cell",
    "caption",
    "controlled_vocabulary",
    "alias_resolution",
    "cross_sentence_link",
    "graph_relation",
]
ConfidenceUncertaintyCode = Literal[
    "ambiguous_surface",
    "nested_mention",
    "ambiguous_entity",
    "ambiguous_sample",
    "indirect_relation",
    "incomplete_context",
    "generic_vocabulary",
    "range_only",
    "method_ambiguous",
    "condition_missing",
    "tolerant_evidence_match",
]


def _validate_aggregate_series_reference(
    observation_role: str,
    series_id: str | None,
    series_ids: list[str] | None,
    label: str,
) -> None:
    has_single = series_id is not None
    has_multiple = series_ids is not None
    if observation_role == "aggregate":
        # 两种违规形态的处置完全不同，报错必须区分，否则会像
        # "缺 table_locator" 那样把模型引向错误的修法：
        # - 都填了：删掉一个即可（本地可判）。
        # - 都没填：说明覆盖范围未能核实，正确做法是改用
        #   unresolved_properties，而不是补一个猜的 series_id。
        if has_single and has_multiple:
            raise ValueError(
                f"aggregate {label} 不得同时填写 series_id 与 series_ids"
            )
        if not has_single and not has_multiple:
            raise ValueError(
                f"aggregate {label} 必须填写 series_id 或 series_ids；"
                "覆盖范围无法从原文核实时应改为 unresolved property，"
                "不得输出无绑定的 aggregate"
            )
    elif has_single or has_multiple:
        raise ValueError(f"非 aggregate {label} 不得引用 Series")
    if series_ids is not None and len(series_ids) != len(set(series_ids)):
        raise ValueError(f"{label}.series_ids 不得重复")


def _series_references(item: Any) -> set[str]:
    references = set(item.series_ids or [])
    if item.series_id is not None:
        references.add(item.series_id)
    return references


def _has_multiple_explicit_series_subjects(points: list[Any]) -> bool:
    if not points or any(
        point.sample_id is None and point.entity_id is None
        for point in points
    ):
        return False
    subjects = {
        (point.sample_id, point.entity_id)
        for point in points
    }
    return len(subjects) >= 2


def _has_auditable_unresolved_series_points(points: list[Any]) -> bool:
    return bool(points) and all(
        point.sample_resolution_status == "unresolved"
        and point.sample_id is None
        and point.entity_id is None
        and bool(point.coordinates)
        and bool(point.evidence)
        for point in points
    )


def _validate_optional_series_reference(
    series_id: str | None,
    series_ids: list[str] | None,
    label: str,
) -> None:
    if series_id is not None and series_ids is not None:
        raise ValueError(f"{label} 不得同时填写 series_id 与 series_ids")
    if series_ids is not None:
        if len(series_ids) < 2:
            raise ValueError(f"{label}.series_ids 至少包含两个 Series")
        if len(series_ids) != len(set(series_ids)):
            raise ValueError(f"{label}.series_ids 不得重复")


def _validate_stage5_subject_scope(
    *,
    sample_id: str | None,
    entity_id: str | None,
    sample_ids: list[str] | None,
    entity_ids: list[str] | None,
    status: str,
    label: str,
) -> None:
    for field_name, values in (
        ("sample_ids", sample_ids),
        ("entity_ids", entity_ids),
    ):
        if values is not None:
            if len(values) < 2:
                raise ValueError(f"{label}.{field_name} 至少包含两个 ID")
            if len(values) != len(set(values)):
                raise ValueError(f"{label}.{field_name} 不得重复")
    if status == "resolved":
        if sample_id is None:
            raise ValueError(f"resolved {label} 必须关联 sample_id")
        if sample_ids is not None or entity_ids is not None:
            raise ValueError(f"resolved {label} 不得填写多主体字段")
    elif status == "unresolved":
        if sample_id is not None:
            raise ValueError(f"unresolved {label} 不得填写 sample_id")
        if entity_id is None:
            raise ValueError(f"unresolved {label} 必须关联 entity_id")
        if sample_ids is not None or entity_ids is not None:
            raise ValueError(f"unresolved {label} 不得填写多主体字段")
    elif status == "multi_resolved":
        if sample_id is not None or entity_id is not None:
            raise ValueError(f"multi_resolved {label} 不得填写单主体字段")
        if sample_ids is None and entity_ids is None:
            raise ValueError(
                f"multi_resolved {label} 必须填写 sample_ids 或 entity_ids"
            )


class TokenUsageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cache_creation_input_tokens: NonNegativeInt = 0
    cache_read_input_tokens: NonNegativeInt = 0
    billable_input_tokens: NonNegativeInt
    total_tokens: NonNegativeInt

    @model_validator(mode="after")
    def validate_totals(self) -> "TokenUsageSummary":
        expected_input = (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )
        if self.billable_input_tokens != expected_input:
            raise ValueError("billable_input_tokens 与输入 token 明细不一致")
        if self.total_tokens != expected_input + self.output_tokens:
            raise ValueError("total_tokens 与 token 明细不一致")
        return self


class StageCost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["calculated", "unavailable", "not_applicable"]
    currency: str | None = None
    input_per_million: Decimal | None = Field(default=None, ge=0)
    output_per_million: Decimal | None = Field(default=None, ge=0)
    input_cost: Decimal | None = Field(default=None, ge=0)
    output_cost: Decimal | None = Field(default=None, ge=0)
    total_cost: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_amounts(self) -> "StageCost":
        amounts = (self.input_cost, self.output_cost, self.total_cost)
        if self.status == "calculated" and any(
            value is None for value in amounts
        ):
            raise ValueError("calculated 费用必须包含完整金额")
        if self.status == "unavailable" and any(
            value is not None for value in amounts
        ):
            raise ValueError("unavailable 费用金额必须为 null")
        if self.status == "not_applicable" and any(
            value != 0 for value in amounts
        ):
            raise ValueError("not_applicable 费用金额必须为 0")
        return self


class MetadataConfidence(BaseModel):
    """文献元数据抽取的字段级置信信息。"""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    field_scores: dict[str, float] = Field(default_factory=dict)
    uncertain_fields: list[str] = Field(default_factory=list)
    evidence_basis: list[ConfidenceEvidenceBasis] = Field(min_length=1)
    uncertainty_codes: list[ConfidenceUncertaintyCode] = Field(
        default_factory=list
    )

    @field_validator("uncertain_fields", "evidence_basis", "uncertainty_codes")
    @classmethod
    def validate_unique_list(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("metadata confidence 列表元素不得重复")
        return value

    @field_validator("field_scores")
    @classmethod
    def validate_field_scores(cls, value: dict[str, float]) -> dict[str, float]:
        if any(
            not key.strip() or score < 0 or score > 1
            for key, score in value.items()
        ):
            raise ValueError("field_scores 键不得为空且分数必须在 0-1")
        return value


class ModelConfidence(BaseModel):
    """模型自评置信度；未经人工标注集校准，不等同于正确概率。"""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)


def compact_confidence_payload(value: Any) -> tuple[Any, list[str]]:
    """递归将模型响应中的 confidence 收敛为仅含 score 的对象。"""
    dropped: list[str] = []

    def visit(current: Any, path: str) -> Any:
        if isinstance(current, list):
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(current)]
        if not isinstance(current, dict):
            return current
        cleaned: dict[str, Any] = {}
        for key, item in current.items():
            child_path = f"{path}.{key}" if path else key
            if key == "confidence" and isinstance(item, dict):
                cleaned_confidence: dict[str, Any] = {}
                if "score" in item:
                    cleaned_confidence["score"] = item["score"]
                dropped.extend(
                    f"{child_path}.{field}"
                    for field in item
                    if field != "score"
                )
                cleaned[key] = cleaned_confidence
            else:
                cleaned[key] = visit(item, child_path)
        return cleaned

    return visit(value, ""), dropped


class ConfidenceCandidateModel(BaseModel):
    """要求新 LLM 候选同步输出单值 confidence。"""

    model_config = ConfigDict(extra="forbid")

    confidence: ModelConfidence


class Paper(BaseModel):
    model_config = ConfigDict(extra="allow")

    ref_no: str = Field(min_length=1)
    pdf_filename: str = Field(min_length=1)
    source_pdf_path: str = Field(min_length=1)
    organized_pdf_path: str = Field(min_length=1)
    doi: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    journal: str | None = None
    year: int | None = None
    metadata_status: MetadataStatus
    metadata_extraction: dict[str, Any]

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (
            not value or any(not author.strip() for author in value)
        ):
            raise ValueError("authors 必须为非空字符串数组或 null")
        return value


class SourceElement(BaseModel):
    model_config = ConfigDict(extra="allow")

    block_id: str = Field(min_length=1)
    page_id: NonNegativeInt
    block_index: NonNegativeInt
    element_type: str = Field(min_length=1)
    bbox: tuple[float, float, float, float] | None = None
    alignment_status: str | None = None


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    paper: Paper
    source_files: dict[str, Any]
    ocr: dict[str, Any]
    elements: list[SourceElement]
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "SourceDocument":
        if self.paper.ref_no != self.document_id:
            raise ValueError("paper.ref_no 必须等于 document_id")
        block_ids = [element.block_id for element in self.elements]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("elements.block_id 不得重复")
        return self


class Stage0TableCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str = Field(min_length=1)
    row_index: NonNegativeInt
    column_index: NonNegativeInt
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    text: str
    is_header: bool = False


class Stage0Element(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1)
    type: Stage0ElementType
    section: str | None = None
    text: str | None = None
    page: NonNegativeInt
    bbox: tuple[float, float, float, float] | None = None
    source_block_index: NonNegativeInt
    alignment_status: str | None = None
    title_level: int | None = None
    caption: str | None = None
    table_body: str | None = None
    table_cells: list[Stage0TableCell] | None = None
    image_path: str | None = None
    image_kind: str | None = None
    equation_kind: Literal["display", "unresolved"] | None = None
    merged_source_block_ids: list[str] | None = None
    content: Any = None

    @model_validator(mode="after")
    def validate_type_payload(self) -> "Stage0Element":
        if self.type in {"text", "title", "equation", "footnote"} and self.text is None:
            raise ValueError(f"{self.type} element 必须包含 text")
        if self.type == "table" and self.table_body is None:
            raise ValueError("table element 必须包含 table_body")
        if self.type != "table" and self.table_cells is not None:
            raise ValueError("非 table element 不得包含 table_cells")
        if self.table_cells is not None:
            cell_ids = [cell.cell_id for cell in self.table_cells]
            if len(cell_ids) != len(set(cell_ids)):
                raise ValueError("table_cells.cell_id 不得重复")
        if self.type == "equation" and self.equation_kind is None:
            raise ValueError("equation element 必须包含 equation_kind")
        return self


class Stage0Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1"] = "1.1"
    source_document_schema_version: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    paper: Paper
    source_files: dict[str, Any]
    ocr: dict[str, Any]
    elements: list[Stage0Element]
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "Stage0Document":
        if self.paper.ref_no != self.document_id:
            raise ValueError("paper.ref_no 必须等于 document_id")
        block_ids = [element.block_id for element in self.elements]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("Stage 0 block_id 不得重复")
        return self


class MentionCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    mention_role: MentionRole

    @field_validator("block_id", "text")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("字段不得为空")
        return value


class MentionChunkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mentions: list[MentionCandidate] = Field(default_factory=list)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1)
    page: NonNegativeInt
    bbox: tuple[float, float, float, float] | None = None
    source_type: Literal["text", "title", "table", "image", "equation", "footnote"]
    source_sentence: str = Field(min_length=1)
    table_locator: dict[str, Any] | None = None


class MaterialMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention_id: str = Field(pattern=r"^m\d{3,}$")
    text: str = Field(min_length=1)
    mention_role: MentionRole
    evidence: Evidence
    confidence: ModelConfidence | None = None


class Stage1Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["stage1_material_mention"] = "stage1_material_mention"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_version: Literal[
        "material_mention_schema.v1",
        "material_mention_schema.v2",
    ]
    implementation_version: Literal[
        "1.0.0", "1.1.0", "1.1.1", "1.1.2", "1.2.0", "1.2.1",
        "1.2.2", "1.2.3", "1.2.4", "1.2.5", "1.2.6",
    ]
    chunk_count: int = Field(ge=1)
    usage: TokenUsageSummary | None = None
    cost: StageCost | None = None
    status: Literal["success"] = "success"


class Stage1Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(min_length=1)
    material_mentions: list[MaterialMention] = Field(default_factory=list)
    provenance: Stage1Provenance
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mentions(self) -> "Stage1Document":
        mention_ids = [mention.mention_id for mention in self.material_mentions]
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError("mention_id 不得重复")
        return self


class EntityEvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1)
    source_sentence: str = Field(min_length=1)


class PolymerEntityCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(pattern=r"^pe\d{3,}$")
    polymer_name: str = Field(min_length=1)
    polymer_type: PolymerType | None = None
    variant_of: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    structural_features: list[StructuralFeatureTag] = Field(default_factory=list)
    resolved_from_mentions: list[str] = Field(min_length=1)
    evidence: EntityEvidenceCandidate
    source_image_block_ids: list[str] = Field(default_factory=list)

    @field_validator(
        "structural_features",
        "resolved_from_mentions",
        "source_image_block_ids",
    )
    @classmethod
    def validate_unique_list(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("列表元素不得重复")
        return value


def _validate_variant_graph(
    entity_ids: set[str],
    variants: dict[str, str | None],
) -> None:
    for entity_id, parent_id in variants.items():
        if parent_id is None:
            continue
        if parent_id not in entity_ids:
            raise ValueError(f"variant_of 引用了未知实体：{parent_id}")
        if parent_id == entity_id:
            raise ValueError("variant_of 不得指向自身")

    for start in entity_ids:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise ValueError("variant_of 不得形成环")
            seen.add(current)
            current = variants.get(current)


class PolymerEntityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[PolymerEntityCandidate] = Field(default_factory=list)
    unresolved_mention_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "PolymerEntityResponse":
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity_id 不得重复")
        if len(self.unresolved_mention_ids) != len(set(self.unresolved_mention_ids)):
            raise ValueError("unresolved_mention_ids 不得重复")
        _validate_variant_graph(
            set(entity_ids),
            {entity.entity_id: entity.variant_of for entity in self.entities},
        )
        return self


class SourceImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1)
    page: NonNegativeInt
    bbox: tuple[float, float, float, float] | None = None
    image_path: str | None = None
    caption: str | None = None


class PolymerEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(pattern=r"^pe\d{3,}$")
    polymer_name: str = Field(min_length=1)
    polymer_type: PolymerType | None = None
    variant_of: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    representation_status: Literal["expert_review_required"] = (
        "expert_review_required"
    )
    structural_features: list[StructuralFeatureTag] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    resolved_from_mentions: list[str] = Field(min_length=1)
    evidence: Evidence
    source_image_refs: list[SourceImageReference] = Field(default_factory=list)
    confidence: ModelConfidence | None = None

    @field_validator(
        "structural_features",
        "source_names",
        "resolved_from_mentions",
    )
    @classmethod
    def validate_unique_list(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("列表元素不得重复")
        return value


class Stage2Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["stage2_polymer_entity"] = "stage2_polymer_entity"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_version: Literal[
        "polymer_entity_schema.v1",
        "polymer_entity_schema.v2",
    ]
    implementation_version: Literal[
        "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.3.1", "1.3.2", "1.3.3"
    ]
    context_block_count: NonNegativeInt
    context_chars: NonNegativeInt
    call_count: NonNegativeInt
    usage: TokenUsageSummary | None = None
    cost: StageCost | None = None
    status: Literal["success"] = "success"


class Stage2Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(min_length=1)
    polymer_entities: list[PolymerEntity] = Field(default_factory=list)
    unresolved_mention_ids: list[str] = Field(default_factory=list)
    provenance: Stage2Provenance
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "Stage2Document":
        entity_ids = [entity.entity_id for entity in self.polymer_entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity_id 不得重复")
        if len(self.unresolved_mention_ids) != len(set(self.unresolved_mention_ids)):
            raise ValueError("unresolved_mention_ids 不得重复")
        _validate_variant_graph(
            set(entity_ids),
            {
                entity.entity_id: entity.variant_of
                for entity in self.polymer_entities
            },
        )
        resolved_ids = [
            mention_id
            for entity in self.polymer_entities
            for mention_id in entity.resolved_from_mentions
        ]
        if len(resolved_ids) != len(set(resolved_ids)):
            raise ValueError("同一 mention 不得解析到多个实体")
        if set(resolved_ids) & set(self.unresolved_mention_ids):
            raise ValueError("resolved 与 unresolved mention 不得重叠")
        return self


class SampleCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=r"^s\d{3,}$")
    sample_kind: SampleKind
    refers_to_entity: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_label_raw: str | None = Field(default=None, min_length=1)
    state_description: str | None = Field(default=None, min_length=1)
    intended_use: list[str] = Field(default_factory=list)
    evidence: EntityEvidenceCandidate

    @field_validator("intended_use")
    @classmethod
    def validate_intended_use(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if (
            any(not item for item in cleaned)
            or len(cleaned) != len(set(cleaned))
        ):
            raise ValueError("intended_use 必须为不重复的非空原文字符串")
        return cleaned

    @model_validator(mode="after")
    def validate_sample_description(self) -> "SampleCandidate":
        if self.sample_label_raw is None and self.state_description is None:
            raise ValueError(
                "Sample 至少需要 sample_label_raw 或 state_description"
            )
        return self


class ProcessStepCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^ps\d{3,}$")
    process_type: ProcessType
    input_sample_ids: list[str] = Field(default_factory=list)
    output_sample_ids: list[str] = Field(min_length=1)
    parameters: dict[str, str] = Field(default_factory=dict)
    evidence: EntityEvidenceCandidate

    @field_validator("input_sample_ids", "output_sample_ids")
    @classmethod
    def validate_unique_list(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("sample 引用不得重复")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, raw_value in value.items():
            clean_key = key.strip()
            clean_value = raw_value.strip()
            if not clean_key or not clean_value:
                raise ValueError("parameters 的键和值不得为空")
            cleaned[clean_key] = clean_value
        return cleaned


def _validate_process_graph(
    samples: list[Any],
    process_steps: list[Any],
) -> None:
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id 不得重复")
    step_ids = [step.step_id for step in process_steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("step_id 不得重复")

    known_samples = set(sample_ids)
    produced: set[str] = set()
    adjacency: dict[str, set[str]] = {
        sample_id: set() for sample_id in sample_ids
    }
    for step in process_steps:
        referenced = set(step.input_sample_ids) | set(step.output_sample_ids)
        unknown = sorted(referenced - known_samples)
        if unknown:
            raise ValueError(f"ProcessStep 引用了未知 sample：{unknown}")
        overlap = set(step.input_sample_ids) & set(step.output_sample_ids)
        if overlap:
            raise ValueError(f"ProcessStep 输入输出不得重叠：{sorted(overlap)}")
        duplicate_outputs = produced & set(step.output_sample_ids)
        if duplicate_outputs:
            raise ValueError(
                f"同一 sample 不得由多个 ProcessStep 生成："
                f"{sorted(duplicate_outputs)}"
            )
        produced.update(step.output_sample_ids)
        for input_id in step.input_sample_ids:
            adjacency[input_id].update(step.output_sample_ids)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(sample_id: str) -> None:
        if sample_id in visiting:
            raise ValueError("ProcessStep sample DAG 不得形成环")
        if sample_id in visited:
            return
        visiting.add(sample_id)
        for output_id in adjacency[sample_id]:
            visit(output_id)
        visiting.remove(sample_id)
        visited.add(sample_id)

    for sample_id in sample_ids:
        visit(sample_id)


class SampleProcessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[SampleCandidate] = Field(default_factory=list)
    process_steps: list[ProcessStepCandidate] = Field(default_factory=list)
    unresolved_entity_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "SampleProcessResponse":
        if len(self.unresolved_entity_ids) != len(set(self.unresolved_entity_ids)):
            raise ValueError("unresolved_entity_ids 不得重复")
        _validate_process_graph(self.samples, self.process_steps)
        return self


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=r"^s\d{3,}$")
    sample_kind: SampleKind
    refers_to_entity: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    polymer_name: str = Field(min_length=1)
    sample_label_raw: str | None = None
    state_description: str | None = None
    intended_use: list[str] = Field(default_factory=list)
    evidence: Evidence
    confidence: ModelConfidence | None = None


class ProcessStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^ps\d{3,}$")
    process_type: ProcessType
    input_sample_ids: list[str] = Field(default_factory=list)
    output_sample_ids: list[str] = Field(min_length=1)
    parameters: dict[str, str] = Field(default_factory=dict)
    evidence: Evidence
    confidence: ModelConfidence | None = None

    @field_validator("input_sample_ids", "output_sample_ids")
    @classmethod
    def validate_unique_list(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("sample 引用不得重复")
        return value


class Stage3Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["stage3_sample_process"] = "stage3_sample_process"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_version: Literal[
        "sample_process_schema.v1",
        "sample_process_schema.v2",
    ]
    implementation_version: Literal[
        "1.0.0", "1.1.0", "1.1.1", "1.1.2", "1.2.0", "1.3.0", "1.3.1",
        "1.3.2", "1.3.3", "1.3.4", "1.3.5", "1.3.6", "1.3.7",
    ]
    context_block_count: NonNegativeInt
    context_chars: NonNegativeInt
    call_count: NonNegativeInt
    usage: TokenUsageSummary | None = None
    cost: StageCost | None = None
    status: Literal["success"] = "success"


class Stage3Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(min_length=1)
    samples: list[Sample] = Field(default_factory=list)
    process_steps: list[ProcessStep] = Field(default_factory=list)
    unresolved_entity_ids: list[str] = Field(default_factory=list)
    provenance: Stage3Provenance
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "Stage3Document":
        if len(self.unresolved_entity_ids) != len(set(self.unresolved_entity_ids)):
            raise ValueError("unresolved_entity_ids 不得重复")
        _validate_process_graph(self.samples, self.process_steps)
        return self


class ConditionQuantity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str = Field(min_length=1)
    value: float | None = None
    unit: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class TableLocatorCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str = Field(min_length=1)
    row_label: str = Field(min_length=1)
    column_label: str = Field(min_length=1)
    cell_value: str | None = Field(default=None, min_length=1)
    cell_id: str | None = Field(default=None, min_length=1)
    row_index: NonNegativeInt | None = None
    column_index: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_stable_coordinates(self) -> "TableLocatorCandidate":
        stable = (self.cell_id, self.row_index, self.column_index)
        if any(value is not None for value in stable) and not all(
            value is not None for value in stable
        ):
            raise ValueError(
                "cell_id、row_index、column_index 必须同时填写或同时为空"
            )
        return self


class PropertyEvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1)
    source_sentence: str = Field(min_length=1)
    table_locator: TableLocatorCandidate | None = None


class ConditionQuantityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str = Field(min_length=1)
    value: float | None = None
    unit: str | None = None
    evidence: list[PropertyEvidenceCandidate] = Field(default_factory=list)


class MeasurementContextCandidate(BaseModel):
    """模型输出的测量条件快照，保留字段级局部 evidence。"""

    model_config = ConfigDict(extra="forbid")

    temperature: ConditionQuantityCandidate | None = None
    frequency: ConditionQuantityCandidate | None = None
    humidity: ConditionQuantityCandidate | None = None
    pressure: ConditionQuantityCandidate | None = None
    wavelength: ConditionQuantityCandidate | None = None
    other_conditions: dict[str, str] = Field(default_factory=dict)
    other_condition_evidence: dict[
        str, list[PropertyEvidenceCandidate]
    ] = Field(default_factory=dict)
    condition_status: Literal["reported", "not_reported"]

    @model_validator(mode="after")
    def validate_status(self) -> "MeasurementContextCandidate":
        quantities = (
            self.temperature,
            self.frequency,
            self.humidity,
            self.pressure,
            self.wavelength,
        )
        has_condition = any(item is not None for item in quantities) or bool(
            self.other_conditions
        )
        if self.condition_status == "not_reported" and has_condition:
            raise ValueError("not_reported context 不得包含条件值")
        if self.condition_status == "reported" and not has_condition:
            raise ValueError("reported context 必须包含至少一个条件值")
        unknown = sorted(
            set(self.other_condition_evidence) - set(self.other_conditions)
        )
        if unknown:
            raise ValueError(
                f"other_condition_evidence 包含未知条件键：{unknown}"
            )
        return self


class MeasurementConditionCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str = Field(pattern=r"^mc\d{3,}$")
    temperature: ConditionQuantityCandidate | None = None
    frequency: ConditionQuantityCandidate | None = None
    humidity: ConditionQuantityCandidate | None = None
    pressure: ConditionQuantityCandidate | None = None
    wavelength: ConditionQuantityCandidate | None = None
    other_conditions: dict[str, str] = Field(default_factory=dict)
    other_condition_evidence: dict[
        str, list[PropertyEvidenceCandidate]
    ] = Field(default_factory=dict)
    condition_status: Literal["reported", "not_reported"]
    evidence: PropertyEvidenceCandidate

    @model_validator(mode="after")
    def validate_status(self) -> "MeasurementConditionCandidate":
        quantities = (
            self.temperature,
            self.frequency,
            self.humidity,
            self.pressure,
            self.wavelength,
        )
        has_condition = any(item is not None for item in quantities) or bool(
            self.other_conditions
        )
        if self.condition_status == "not_reported" and has_condition:
            raise ValueError("not_reported condition 不得包含条件值")
        if self.condition_status == "reported" and not has_condition:
            raise ValueError("reported condition 必须包含至少一个条件值")
        return self


class MeasurementContext(BaseModel):
    """随一次性质或表征保存的测量条件快照。"""

    model_config = ConfigDict(extra="forbid")

    temperature: ConditionQuantity | None = None
    frequency: ConditionQuantity | None = None
    humidity: ConditionQuantity | None = None
    pressure: ConditionQuantity | None = None
    wavelength: ConditionQuantity | None = None
    other_conditions: dict[str, str] = Field(default_factory=dict)
    other_condition_evidence: dict[str, list[Evidence]] = Field(
        default_factory=dict
    )
    other_condition_evidence_ids: dict[str, list[str]] = Field(
        default_factory=dict
    )
    condition_status: Literal["reported", "not_reported"]

    @model_validator(mode="after")
    def validate_status(self) -> "MeasurementContext":
        quantities = (
            self.temperature,
            self.frequency,
            self.humidity,
            self.pressure,
            self.wavelength,
        )
        has_condition = any(item is not None for item in quantities) or bool(
            self.other_conditions
        )
        if self.condition_status == "not_reported" and has_condition:
            raise ValueError("not_reported context 不得包含条件值")
        if self.condition_status == "reported" and not has_condition:
            raise ValueError("reported context 必须包含至少一个条件值")
        known = set(self.other_conditions)
        unknown = sorted(
            (set(self.other_condition_evidence) | set(
                self.other_condition_evidence_ids
            )) - known
        )
        if unknown:
            raise ValueError(f"条件 evidence 包含未知键：{unknown}")
        return self


class PropertyObservationCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(pattern=r"^prop\d{3,}$")
    sample_id: str = Field(pattern=r"^s\d{3,}$")
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: str | None = None
    property_code: str | None = None
    property_category: PropertyCategory | None = None
    molecular_weight_type: Literal[
        "Mn", "Mw", "Mv", "Mz", "unspecified"
    ] | None = None
    determination_method_raw: str | None = Field(default=None, min_length=1)
    observation_group_id: str | None = Field(
        default=None,
        pattern=r"^pog\d{3,}$",
    )
    observation_role: Literal["single", "aggregate"] = "single"
    series_id: SeriesReferenceID | None = None
    series_ids: list[SeriesReferenceID] | None = Field(
        default=None,
        min_length=2,
    )
    value_raw: str = Field(min_length=1)
    value_min: float | None = None
    value_max: float | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_condition_id: str = Field(pattern=r"^mc\d{3,}$")
    measurement_context: MeasurementContextCandidate | None = None
    evidence: list[PropertyEvidenceCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_values(self) -> "PropertyObservationCandidate":
        if (
            self.property_name_raw.strip().casefold()
            == self.value_raw.strip().casefold()
        ):
            raise ValueError("property_name_raw 不得等于 value_raw")
        controlled = (
            self.property_name_normalized,
            self.property_code,
            self.property_category,
        )
        if any(item is not None for item in controlled) and not all(
            item is not None for item in controlled
        ):
            raise ValueError("受控性质名称、代码和类别必须同时填写或同时为空")
        if (
            self.value_min is not None
            and self.value_max is not None
            and self.value_min > self.value_max
        ):
            raise ValueError("value_min 不得大于 value_max")
        _validate_aggregate_series_reference(
            self.observation_role,
            self.series_id,
            self.series_ids,
            "PropertyObservation",
        )
        return self


class UnresolvedPropertyCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    unresolved_id: str = Field(pattern=r"^uprop\d{3,}$")
    entity_id: str = Field(pattern=r"^pe\d{3,}$")
    sample_id: None = None
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: None = None
    property_code: None = None
    property_category: None = None
    molecular_weight_type: None = None
    determination_method_raw: str | None = Field(default=None, min_length=1)
    observation_group_id: str | None = Field(
        default=None,
        pattern=r"^pog\d{3,}$",
    )
    observation_role: Literal["single", "aggregate"] = "single"
    series_id: SeriesReferenceID | None = None
    series_ids: list[SeriesReferenceID] | None = Field(
        default=None,
        min_length=2,
    )
    value_raw: str = Field(min_length=1)
    value_min: None = None
    value_max: None = None
    unit_raw: str | None = None
    unit_normalized: None = None
    measurement_condition_id: None = None
    measurement_context: MeasurementContextCandidate | None = None
    reason: Literal["sample_ambiguous", "sample_not_found"]
    evidence: list[PropertyEvidenceCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_property_name(self) -> "UnresolvedPropertyCandidate":
        if (
            self.property_name_raw.strip().casefold()
            == self.value_raw.strip().casefold()
        ):
            raise ValueError("property_name_raw 不得等于 value_raw")
        _validate_aggregate_series_reference(
            self.observation_role,
            self.series_id,
            self.series_ids,
            "UnresolvedProperty",
        )
        return self


class SeriesCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: NonNegativeInt
    covered: NonNegativeInt
    missing: NonNegativeInt
    not_applicable: NonNegativeInt = 0
    ratio: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self) -> "SeriesCoverage":
        if self.expected != self.covered + self.missing:
            raise ValueError("expected 必须等于 covered + missing")
        expected_ratio = self.covered / self.expected if self.expected else 1.0
        if abs(self.ratio - expected_ratio) > 1e-9:
            raise ValueError("ratio 必须等于 covered / expected")
        return self


class PropertySeriesCoordinateCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name_raw: str = Field(min_length=1)
    value_raw: str = Field(min_length=1)
    unit_raw: str | None = None
    evidence: PropertyEvidenceCandidate


class PropertySeriesPointCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(pattern=r"^pt\d{3,}$")
    observation_role: Literal["series_point"] = "series_point"
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_resolution_status: SampleResolutionStatus | None = None
    coordinates: list[PropertySeriesCoordinateCandidate] = Field(
        default_factory=list
    )
    value_raw: str | None = None
    value_min: float | None = None
    value_max: float | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_context: MeasurementContextCandidate | None = None
    coverage_status: CoverageStatus
    evidence: list[PropertyEvidenceCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_coverage(self) -> "PropertySeriesPointCandidate":
        if self.coverage_status == "covered" and not self.value_raw:
            raise ValueError("covered series point 必须包含 value_raw")
        if self.sample_resolution_status == "resolved" and self.sample_id is None:
            raise ValueError("resolved series point 必须关联 sample_id")
        if self.sample_resolution_status == "unresolved" and self.sample_id is not None:
            raise ValueError("unresolved series point 不得填写 sample_id")
        if (
            self.sample_resolution_status is not None
            and
            self.sample_id is None
            and self.entity_id is None
            and not (
                self.sample_resolution_status == "unresolved"
                and bool(self.coordinates)
                and bool(self.evidence)
            )
        ):
            raise ValueError(
                "series point 至少关联 Sample 或 PolymerEntity，"
                "或以坐标和证据明确标记 unresolved"
            )
        if self.value_min is not None and self.value_max is not None:
            if self.value_min > self.value_max:
                raise ValueError("value_min 不得大于 value_max")
        return self


class PropertySeriesCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(pattern=r"^series\d{3,}$")
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_resolution_status: SampleResolutionStatus
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: str | None = None
    property_code: str | None = None
    property_category: PropertyCategory | None = None
    determination_method_raw: str | None = Field(default=None, min_length=1)
    observation_group_id: str | None = Field(
        default=None,
        pattern=r"^pog\d{3,}$",
    )
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_context: MeasurementContextCandidate
    points: list[PropertySeriesPointCandidate] = Field(min_length=1)
    coverage: SeriesCoverage | None = None
    evidence: list[PropertyEvidenceCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_series(self) -> "PropertySeriesCandidate":
        multiple_subjects = _has_multiple_explicit_series_subjects(self.points)
        if self.sample_resolution_status == "resolved" and self.sample_id is None:
            raise ValueError("resolved PropertySeries 必须关联 sample_id")
        if self.sample_resolution_status == "unresolved" and self.sample_id is not None:
            raise ValueError("unresolved PropertySeries 不得填写 sample_id")
        if (
            self.sample_id is None
            and self.entity_id is None
            and not multiple_subjects
            and not _has_auditable_unresolved_series_points(self.points)
        ):
            raise ValueError("PropertySeries 至少关联 Sample 或 PolymerEntity")
        point_ids = [point.point_id for point in self.points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("PropertySeries.point_id 不得重复")
        counts = {
            status: sum(point.coverage_status == status for point in self.points)
            for status in ("covered", "missing", "not_applicable")
        }
        if self.coverage is not None and (
            counts["covered"] != self.coverage.covered
            or counts["missing"] != self.coverage.missing
            or counts["not_applicable"] != self.coverage.not_applicable
        ):
            raise ValueError("PropertySeries coverage 与 points 状态计数不一致")
        return self


class PropertyStageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_conditions: list[MeasurementConditionCandidate] = Field(
        default_factory=list
    )
    properties: list[PropertyObservationCandidate] = Field(default_factory=list)
    unresolved_properties: list[UnresolvedPropertyCandidate] = Field(
        default_factory=list
    )
    property_series: list[PropertySeriesCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "PropertyStageResponse":
        condition_ids = [
            condition.condition_id for condition in self.measurement_conditions
        ]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("condition_id 不得重复")
        property_ids = [item.property_id for item in self.properties]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError("property_id 不得重复")
        unresolved_ids = [
            item.unresolved_id for item in self.unresolved_properties
        ]
        if len(unresolved_ids) != len(set(unresolved_ids)):
            raise ValueError("unresolved_id 不得重复")
        series_ids = [item.series_id for item in self.property_series]
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("series_id 不得重复")

        known_conditions = set(condition_ids)
        referenced_conditions = {
            item.measurement_condition_id for item in self.properties
        }
        unknown = sorted(referenced_conditions - known_conditions)
        if unknown:
            raise ValueError(f"property 引用了未知 condition：{unknown}")
        known_series = set(series_ids)
        referenced_series = set().union(*(
            _series_references(item)
            for item in [*self.properties, *self.unresolved_properties]
        )) if (self.properties or self.unresolved_properties) else set()
        unknown_series = sorted(referenced_series - known_series)
        if unknown_series:
            raise ValueError(f"aggregate 引用了未知 series：{unknown_series}")
        return self


class MeasurementCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str = Field(pattern=r"^mc\d{3,}$")
    temperature: ConditionQuantity | None = None
    frequency: ConditionQuantity | None = None
    humidity: ConditionQuantity | None = None
    pressure: ConditionQuantity | None = None
    wavelength: ConditionQuantity | None = None
    other_conditions: dict[str, str] = Field(default_factory=dict)
    other_condition_evidence: dict[str, list[Evidence]] = Field(
        default_factory=dict
    )
    condition_status: Literal["reported", "not_reported"]
    evidence: Evidence
    confidence: ModelConfidence | None = None


class PropertyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(pattern=r"^prop\d{3,}$")
    sample_id: str = Field(pattern=r"^s\d{3,}$")
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: str | None = None
    property_code: str | None = None
    property_category: PropertyCategory | None = None
    molecular_weight_type: Literal[
        "Mn", "Mw", "Mv", "Mz", "unspecified"
    ] | None = None
    determination_method_raw: str | None = Field(default=None, min_length=1)
    observation_group_id: str | None = Field(
        default=None,
        pattern=r"^pog\d{3,}$",
    )
    observation_role: Literal["single", "aggregate"] = "single"
    series_id: SeriesReferenceID | None = None
    series_ids: list[SeriesReferenceID] | None = Field(
        default=None,
        min_length=2,
    )
    value_raw: str = Field(min_length=1)
    value_min: float | None = None
    value_max: float | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_condition_id: str = Field(pattern=r"^mc\d{3,}$")
    measurement_context: MeasurementContext | None = None
    source_type: Literal["text", "title", "table", "image", "equation", "footnote"]
    evidence: list[Evidence] = Field(min_length=1)
    confidence: ModelConfidence | None = None

    @model_validator(mode="after")
    def validate_series_reference(self) -> "PropertyObservation":
        _validate_aggregate_series_reference(
            self.observation_role,
            self.series_id,
            self.series_ids,
            "PropertyObservation",
        )
        return self


class UnresolvedPropertyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unresolved_id: str = Field(pattern=r"^uprop\d{3,}$")
    entity_id: str = Field(pattern=r"^pe\d{3,}$")
    sample_id: None = None
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: None = None
    property_code: None = None
    property_category: None = None
    molecular_weight_type: None = None
    determination_method_raw: str | None = Field(default=None, min_length=1)
    observation_group_id: str | None = Field(
        default=None,
        pattern=r"^pog\d{3,}$",
    )
    observation_role: Literal["single", "aggregate"] = "single"
    series_id: SeriesReferenceID | None = None
    series_ids: list[SeriesReferenceID] | None = Field(
        default=None,
        min_length=2,
    )
    value_raw: str = Field(min_length=1)
    value_min: None = None
    value_max: None = None
    unit_raw: str | None = None
    unit_normalized: None = None
    measurement_condition_id: None = None
    measurement_context: MeasurementContext | None = None
    reason: Literal["sample_ambiguous", "sample_not_found"]
    evidence: list[Evidence] = Field(min_length=1)
    confidence: ModelConfidence | None = None

    @model_validator(mode="after")
    def validate_series_reference(self) -> "UnresolvedPropertyObservation":
        _validate_aggregate_series_reference(
            self.observation_role,
            self.series_id,
            self.series_ids,
            "UnresolvedPropertyObservation",
        )
        return self


class PropertySeriesCoordinate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name_raw: str = Field(min_length=1)
    value_raw: str = Field(min_length=1)
    unit_raw: str | None = None
    evidence: Evidence


class PropertySeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(pattern=r"^pt\d{3,}$")
    observation_role: Literal["series_point"] = "series_point"
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_resolution_status: SampleResolutionStatus
    coordinates: list[PropertySeriesCoordinate] = Field(default_factory=list)
    value_raw: str | None = None
    value_min: float | None = None
    value_max: float | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_context: MeasurementContext
    coverage_status: CoverageStatus
    evidence: list[Evidence] = Field(min_length=1)
    confidence: ModelConfidence


class PropertySeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(pattern=r"^series\d{3,}$")
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_resolution_status: SampleResolutionStatus
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: str | None = None
    property_code: str | None = None
    property_category: PropertyCategory | None = None
    determination_method_raw: str | None = Field(default=None, min_length=1)
    observation_group_id: str | None = Field(
        default=None,
        pattern=r"^pog\d{3,}$",
    )
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_context: MeasurementContext
    points: list[PropertySeriesPoint] = Field(min_length=1)
    coverage: SeriesCoverage
    evidence: list[Evidence] = Field(min_length=1)
    confidence: ModelConfidence

    @model_validator(mode="after")
    def validate_series(self) -> "PropertySeries":
        multiple_subjects = _has_multiple_explicit_series_subjects(self.points)
        if self.sample_resolution_status == "resolved" and self.sample_id is None:
            raise ValueError("resolved PropertySeries 必须关联 sample_id")
        if self.sample_resolution_status == "unresolved" and self.sample_id is not None:
            raise ValueError("unresolved PropertySeries 不得填写 sample_id")
        if (
            self.sample_id is None
            and self.entity_id is None
            and not multiple_subjects
            and not _has_auditable_unresolved_series_points(self.points)
        ):
            raise ValueError("PropertySeries 至少关联 Sample 或 PolymerEntity")
        point_ids = [point.point_id for point in self.points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("PropertySeries.point_id 不得重复")
        counts = {
            status: sum(point.coverage_status == status for point in self.points)
            for status in ("covered", "missing", "not_applicable")
        }
        if (
            counts["covered"] != self.coverage.covered
            or counts["missing"] != self.coverage.missing
            or counts["not_applicable"] != self.coverage.not_applicable
        ):
            raise ValueError("PropertySeries coverage 与 points 状态计数不一致")
        return self


class Stage4Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["stage4_property"] = "stage4_property"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_version: Literal[
        "property_observation_schema.v1",
        "property_observation_schema.v2",
        "property_observation_schema.v3",
        "property_observation_schema.v4",
        "property_observation_schema.v5",
        "property_observation_schema.v6",
        "property_observation_schema.v7",
    ]
    implementation_version: Literal[
        "1.0.0", "1.1.0", "1.2.0", "1.2.1", "1.2.2", "1.2.3",
        "1.2.4", "1.2.5", "1.2.6", "1.2.7", "1.2.8", "1.2.9",
        "1.3.0", "1.3.1", "1.3.2", "1.3.3", "1.3.4", "1.4.0",
        "1.4.1",
        "1.5.0",
        "1.6.0",
        "1.6.1",
        "1.6.2",
        "1.6.3",
        "1.6.4",
        "1.6.5",
        "1.6.6",
        "1.6.7",
        "1.6.8",
        "1.6.9",
        "1.6.10",
        "1.6.11",
        "1.6.12",
        "1.6.13",
        "1.7.0",
        "1.7.1",
        "1.7.2",
        "1.7.3",
        "1.7.4", "1.7.5", "1.7.6", "1.7.7", "1.7.8", "1.7.9", "1.7.10",
    ]
    context_block_count: NonNegativeInt
    context_chars: NonNegativeInt
    call_count: NonNegativeInt
    usage: TokenUsageSummary | None = None
    cost: StageCost | None = None
    status: Literal["success"] = "success"


class Stage4Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(min_length=1)
    measurement_conditions: list[MeasurementCondition] = Field(
        default_factory=list
    )
    properties: list[PropertyObservation] = Field(default_factory=list)
    unresolved_properties: list[UnresolvedPropertyObservation] = Field(
        default_factory=list
    )
    property_series: list[PropertySeries] = Field(default_factory=list)
    provenance: Stage4Provenance
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "Stage4Document":
        condition_ids = [
            condition.condition_id for condition in self.measurement_conditions
        ]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("condition_id 不得重复")
        property_ids = [item.property_id for item in self.properties]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError("property_id 不得重复")
        unresolved_ids = [
            item.unresolved_id for item in self.unresolved_properties
        ]
        if len(unresolved_ids) != len(set(unresolved_ids)):
            raise ValueError("unresolved_id 不得重复")
        series_ids = [item.series_id for item in self.property_series]
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("series_id 不得重复")
        known_conditions = set(condition_ids)
        unknown = sorted({
            item.measurement_condition_id
            for item in self.properties
        } - known_conditions)
        if unknown:
            raise ValueError(f"property 引用了未知 condition：{unknown}")
        known_series = set(series_ids)
        referenced_series = set().union(*(
            _series_references(item)
            for item in [*self.properties, *self.unresolved_properties]
        )) if (self.properties or self.unresolved_properties) else set()
        unknown_series = sorted(referenced_series - known_series)
        if unknown_series:
            raise ValueError(f"aggregate 引用了未知 series：{unknown_series}")
        return self


class CharacterizationCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    characterization_id: str = Field(pattern=r"^char\d{3,}$")
    method_raw: str = Field(min_length=1)
    method_normalized: str = Field(min_length=1)
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_ids: list[Annotated[str, Field(pattern=r"^s\d{3,}$")]] | None = None
    entity_ids: list[Annotated[str, Field(pattern=r"^pe\d{3,}$")]] | None = None
    sample_resolution_status: Stage5SubjectResolutionStatus
    series_id: str | None = Field(default=None, pattern=r"^series\d{3,}$")
    series_ids: list[SeriesReferenceID] | None = None
    instrument: str | None = None
    measurement_context: MeasurementContextCandidate | None = None
    parameters: dict[str, str] = Field(default_factory=dict)
    result_summary: str | None = None
    derived_property_ids: list[str] = Field(default_factory=list)
    evidence: list[PropertyEvidenceCandidate] = Field(min_length=1)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, raw_value in value.items():
            clean_key = key.strip()
            clean_value = raw_value.strip()
            if not clean_key or not clean_value:
                raise ValueError("parameters 的键和值不得为空")
            cleaned[clean_key] = clean_value
        return cleaned

    @field_validator("derived_property_ids")
    @classmethod
    def validate_unique_property_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("derived_property_ids 不得重复")
        return value

    @model_validator(mode="after")
    def validate_resolution(self) -> "CharacterizationCandidate":
        _validate_stage5_subject_scope(
            sample_id=self.sample_id,
            entity_id=self.entity_id,
            sample_ids=self.sample_ids,
            entity_ids=self.entity_ids,
            status=self.sample_resolution_status,
            label="Characterization",
        )
        _validate_optional_series_reference(
            self.series_id,
            self.series_ids,
            "Characterization",
        )
        return self


class Stage5PropertyCandidate(ConfidenceCandidateModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(pattern=r"^prop_s5_\d{3,}$")
    characterization_id: str = Field(pattern=r"^char\d{3,}$")
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_ids: list[Annotated[str, Field(pattern=r"^s\d{3,}$")]] | None = None
    entity_ids: list[Annotated[str, Field(pattern=r"^pe\d{3,}$")]] | None = None
    sample_resolution_status: Stage5SubjectResolutionStatus
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: str = Field(min_length=1)
    property_category: Stage5PropertyCategory
    value_raw: str = Field(min_length=1)
    value_min: float | None = None
    value_max: float | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_context: MeasurementContextCandidate | None = None
    spectral_assignment: str | None = None
    solvent: str | None = None
    source_stage: Literal["stage5"] = "stage5"
    evidence: list[PropertyEvidenceCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_values(self) -> "Stage5PropertyCandidate":
        _validate_stage5_subject_scope(
            sample_id=self.sample_id,
            entity_id=self.entity_id,
            sample_ids=self.sample_ids,
            entity_ids=self.entity_ids,
            status=self.sample_resolution_status,
            label="Stage 5 property",
        )
        if (
            self.value_min is not None
            and self.value_max is not None
            and self.value_min > self.value_max
        ):
            raise ValueError("value_min 不得大于 value_max")
        return self


class CharacterizationStageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    characterizations: list[CharacterizationCandidate] = Field(
        default_factory=list
    )
    properties: list[Stage5PropertyCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "CharacterizationStageResponse":
        characterization_ids = [
            item.characterization_id for item in self.characterizations
        ]
        if len(characterization_ids) != len(set(characterization_ids)):
            raise ValueError("characterization_id 不得重复")
        property_ids = [item.property_id for item in self.properties]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError("Stage 5 property_id 不得重复")
        known_characterizations = set(characterization_ids)
        unknown = sorted({
            item.characterization_id for item in self.properties
        } - known_characterizations)
        if unknown:
            raise ValueError(f"Stage 5 property 引用了未知 characterization：{unknown}")
        owners = {
            item.characterization_id: set(item.derived_property_ids)
            for item in self.characterizations
        }
        for item in self.properties:
            if item.property_id not in owners[item.characterization_id]:
                raise ValueError(
                    f"{item.property_id} 未出现在所属 Characterization 的 "
                    "derived_property_ids"
                )
        referenced_stage5_ids = {
            property_id
            for item in self.characterizations
            for property_id in item.derived_property_ids
            if property_id.startswith("prop_s5_")
        }
        if referenced_stage5_ids != set(property_ids):
            raise ValueError(
                "derived_property_ids 中的 Stage 5 property 引用与 properties 不一致"
            )
        return self


class Characterization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    characterization_id: str = Field(pattern=r"^char\d{3,}$")
    method_raw: str = Field(min_length=1)
    method_normalized: str = Field(min_length=1)
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_ids: list[Annotated[str, Field(pattern=r"^s\d{3,}$")]] | None = None
    entity_ids: list[Annotated[str, Field(pattern=r"^pe\d{3,}$")]] | None = None
    sample_resolution_status: Stage5SubjectResolutionStatus
    series_id: str | None = Field(default=None, pattern=r"^series\d{3,}$")
    series_ids: list[SeriesReferenceID] | None = None
    instrument: str | None = None
    measurement_context: MeasurementContext | None = None
    parameters: dict[str, str] = Field(default_factory=dict)
    result_summary: str | None = None
    derived_property_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: ModelConfidence | None = None

    @field_validator("derived_property_ids")
    @classmethod
    def validate_unique_property_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("derived_property_ids 不得重复")
        return value

    @model_validator(mode="after")
    def validate_resolution(self) -> "Characterization":
        _validate_stage5_subject_scope(
            sample_id=self.sample_id,
            entity_id=self.entity_id,
            sample_ids=self.sample_ids,
            entity_ids=self.entity_ids,
            status=self.sample_resolution_status,
            label="Characterization",
        )
        _validate_optional_series_reference(
            self.series_id,
            self.series_ids,
            "Characterization",
        )
        return self


class Stage5PropertyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(pattern=r"^prop_s5_\d{3,}$")
    characterization_id: str = Field(pattern=r"^char\d{3,}$")
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_ids: list[Annotated[str, Field(pattern=r"^s\d{3,}$")]] | None = None
    entity_ids: list[Annotated[str, Field(pattern=r"^pe\d{3,}$")]] | None = None
    sample_resolution_status: Stage5SubjectResolutionStatus
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: str = Field(min_length=1)
    property_category: Stage5PropertyCategory
    value_raw: str = Field(min_length=1)
    value_min: float | None = None
    value_max: float | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_context: MeasurementContext | None = None
    spectral_assignment: str | None = None
    solvent: str | None = None
    source_stage: Literal["stage5"] = "stage5"
    source_type: Stage0ElementType
    evidence: list[Evidence] = Field(min_length=1)
    confidence: ModelConfidence | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "Stage5PropertyObservation":
        _validate_stage5_subject_scope(
            sample_id=self.sample_id,
            entity_id=self.entity_id,
            sample_ids=self.sample_ids,
            entity_ids=self.entity_ids,
            status=self.sample_resolution_status,
            label="Stage 5 property",
        )
        return self


class Stage5Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["stage5_characterization"] = "stage5_characterization"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_version: Literal[
        "characterization_schema.v1",
        "characterization_schema.v2",
        "characterization_schema.v3",
        "characterization_schema.v4",
    ]
    implementation_version: Literal[
        "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.5.1",
        "1.6.0",
        "1.6.1",
        "1.6.2",
        "1.6.3",
        "1.6.4",
        "1.6.5",
        "1.6.6",
        "1.7.0",
        "1.7.1",
    ]
    context_block_count: NonNegativeInt
    context_chars: NonNegativeInt
    call_count: NonNegativeInt
    usage: TokenUsageSummary | None = None
    cost: StageCost | None = None
    status: Literal["success"] = "success"


class Stage5Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(min_length=1)
    characterizations: list[Characterization] = Field(default_factory=list)
    properties: list[Stage5PropertyObservation] = Field(default_factory=list)
    provenance: Stage5Provenance
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "Stage5Document":
        characterization_ids = [
            item.characterization_id for item in self.characterizations
        ]
        if len(characterization_ids) != len(set(characterization_ids)):
            raise ValueError("characterization_id 不得重复")
        property_ids = [item.property_id for item in self.properties]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError("Stage 5 property_id 不得重复")
        known_characterizations = set(characterization_ids)
        unknown = sorted({
            item.characterization_id for item in self.properties
        } - known_characterizations)
        if unknown:
            raise ValueError(f"Stage 5 property 引用了未知 characterization：{unknown}")
        owners = {
            item.characterization_id: set(item.derived_property_ids)
            for item in self.characterizations
        }
        for item in self.properties:
            if item.property_id not in owners[item.characterization_id]:
                raise ValueError(
                    f"{item.property_id} 未出现在所属 Characterization 的 "
                    "derived_property_ids"
                )
        referenced_stage5_ids = {
            property_id
            for item in self.characterizations
            for property_id in item.derived_property_ids
            if property_id.startswith("prop_s5_")
        }
        if referenced_stage5_ids != set(property_ids):
            raise ValueError(
                "derived_property_ids 中的 Stage 5 property 引用与 properties 不一致"
            )
        return self


class FinalEvidence(Evidence):
    evidence_id: str = Field(pattern=r"^ev\d{3,}$")


class FinalMaterialMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention_id: str = Field(pattern=r"^m\d{3,}$")
    text: str = Field(min_length=1)
    mention_role: MentionRole
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence | None = None


class FinalPolymerEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(pattern=r"^pe\d{3,}$")
    polymer_name: str = Field(min_length=1)
    polymer_type: PolymerType | None = None
    variant_of: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    representation_status: Literal["expert_review_required"]
    structural_features: list[StructuralFeatureTag] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    resolved_from_mentions: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    source_image_refs: list[SourceImageReference] = Field(default_factory=list)
    confidence: ModelConfidence | None = None


class FinalSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=r"^s\d{3,}$")
    sample_kind: SampleKind
    refers_to_entity: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    polymer_name: str = Field(min_length=1)
    sample_label_raw: str | None = None
    state_description: str | None = None
    intended_use: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence | None = None


class FinalProcessStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^ps\d{3,}$")
    process_type: ProcessType
    input_sample_ids: list[str] = Field(default_factory=list)
    output_sample_ids: list[str] = Field(min_length=1)
    parameters: dict[str, str] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence | None = None


class FinalMeasurementCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str = Field(pattern=r"^mc\d{3,}$")
    temperature: ConditionQuantity | None = None
    frequency: ConditionQuantity | None = None
    humidity: ConditionQuantity | None = None
    pressure: ConditionQuantity | None = None
    wavelength: ConditionQuantity | None = None
    other_conditions: dict[str, str] = Field(default_factory=dict)
    other_condition_evidence_ids: dict[str, list[str]] = Field(
        default_factory=dict
    )
    condition_status: Literal["reported", "not_reported"]
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence | None = None


class FinalPropertyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(pattern=r"^prop\d{3,}$")
    sample_id: str = Field(pattern=r"^s\d{3,}$")
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: str | None = None
    property_code: str | None = None
    property_category: PropertyCategory | None = None
    molecular_weight_type: Literal[
        "Mn", "Mw", "Mv", "Mz", "unspecified"
    ] | None = None
    determination_method_raw: str | None = Field(default=None, min_length=1)
    observation_group_id: str | None = Field(
        default=None,
        pattern=r"^pog\d{3,}$",
    )
    observation_role: Literal["single", "aggregate"] = "single"
    series_id: SeriesReferenceID | None = None
    series_ids: list[SeriesReferenceID] | None = Field(
        default=None,
        min_length=2,
    )
    value_raw: str = Field(min_length=1)
    value_min: float | None = None
    value_max: float | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_condition_id: str = Field(pattern=r"^mc\d{3,}$")
    measurement_context: MeasurementContext | None = None
    source_type: Stage0ElementType
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence | None = None

    @model_validator(mode="after")
    def validate_series_reference(self) -> "FinalPropertyObservation":
        _validate_aggregate_series_reference(
            self.observation_role,
            self.series_id,
            self.series_ids,
            "FinalPropertyObservation",
        )
        return self


class FinalUnresolvedPropertyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unresolved_id: str = Field(pattern=r"^uprop\d{3,}$")
    entity_id: str = Field(pattern=r"^pe\d{3,}$")
    sample_id: None = None
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: None = None
    property_code: None = None
    property_category: None = None
    molecular_weight_type: None = None
    determination_method_raw: str | None = Field(default=None, min_length=1)
    observation_group_id: str | None = Field(
        default=None,
        pattern=r"^pog\d{3,}$",
    )
    observation_role: Literal["single", "aggregate"] = "single"
    series_id: SeriesReferenceID | None = None
    series_ids: list[SeriesReferenceID] | None = Field(
        default=None,
        min_length=2,
    )
    value_raw: str = Field(min_length=1)
    value_min: None = None
    value_max: None = None
    unit_raw: str | None = None
    unit_normalized: None = None
    measurement_condition_id: None = None
    measurement_context: MeasurementContext | None = None
    reason: Literal["sample_ambiguous", "sample_not_found"]
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence | None = None

    @model_validator(mode="after")
    def validate_series_reference(
        self,
    ) -> "FinalUnresolvedPropertyObservation":
        _validate_aggregate_series_reference(
            self.observation_role,
            self.series_id,
            self.series_ids,
            "FinalUnresolvedPropertyObservation",
        )
        return self


class FinalCharacterization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    characterization_id: str = Field(pattern=r"^char\d{3,}$")
    method_raw: str = Field(min_length=1)
    method_normalized: str = Field(min_length=1)
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_ids: list[Annotated[str, Field(pattern=r"^s\d{3,}$")]] | None = None
    entity_ids: list[Annotated[str, Field(pattern=r"^pe\d{3,}$")]] | None = None
    sample_resolution_status: Stage5SubjectResolutionStatus
    series_id: str | None = Field(default=None, pattern=r"^series\d{3,}$")
    series_ids: list[SeriesReferenceID] | None = None
    instrument: str | None = None
    measurement_context: MeasurementContext | None = None
    parameters: dict[str, str] = Field(default_factory=dict)
    result_summary: str | None = None
    derived_property_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence | None = None

    @model_validator(mode="after")
    def validate_series_references(self) -> "FinalCharacterization":
        _validate_stage5_subject_scope(
            sample_id=self.sample_id,
            entity_id=self.entity_id,
            sample_ids=self.sample_ids,
            entity_ids=self.entity_ids,
            status=self.sample_resolution_status,
            label="Characterization",
        )
        _validate_optional_series_reference(
            self.series_id,
            self.series_ids,
            "Characterization",
        )
        return self


class FinalStage5PropertyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str = Field(pattern=r"^prop_s5_\d{3,}$")
    characterization_id: str = Field(pattern=r"^char\d{3,}$")
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_ids: list[Annotated[str, Field(pattern=r"^s\d{3,}$")]] | None = None
    entity_ids: list[Annotated[str, Field(pattern=r"^pe\d{3,}$")]] | None = None
    sample_resolution_status: Stage5SubjectResolutionStatus
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: str = Field(min_length=1)
    property_category: Stage5PropertyCategory
    value_raw: str = Field(min_length=1)
    value_min: float | None = None
    value_max: float | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_context: MeasurementContext | None = None
    spectral_assignment: str | None = None
    solvent: str | None = None
    source_stage: Literal["stage5"] = "stage5"
    source_type: Stage0ElementType
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "FinalStage5PropertyObservation":
        _validate_stage5_subject_scope(
            sample_id=self.sample_id,
            entity_id=self.entity_id,
            sample_ids=self.sample_ids,
            entity_ids=self.entity_ids,
            status=self.sample_resolution_status,
            label="Stage 5 property",
        )
        return self


class FinalPropertySeriesCoordinate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name_raw: str = Field(min_length=1)
    value_raw: str = Field(min_length=1)
    unit_raw: str | None = None
    evidence_ids: list[str] = Field(min_length=1)


class FinalPropertySeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(pattern=r"^pt\d{3,}$")
    observation_role: Literal["series_point"] = "series_point"
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_resolution_status: SampleResolutionStatus
    coordinates: list[FinalPropertySeriesCoordinate] = Field(
        default_factory=list
    )
    value_raw: str | None = None
    value_min: float | None = None
    value_max: float | None = None
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_context: MeasurementContext
    coverage_status: CoverageStatus
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence


class FinalPropertySeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(pattern=r"^series\d{3,}$")
    sample_id: str | None = Field(default=None, pattern=r"^s\d{3,}$")
    entity_id: str | None = Field(default=None, pattern=r"^pe\d{3,}$")
    sample_resolution_status: SampleResolutionStatus
    property_name_raw: str = Field(min_length=1)
    property_name_normalized: str | None = None
    property_code: str | None = None
    property_category: PropertyCategory | None = None
    determination_method_raw: str | None = Field(default=None, min_length=1)
    observation_group_id: str | None = Field(
        default=None,
        pattern=r"^pog\d{3,}$",
    )
    unit_raw: str | None = None
    unit_normalized: str | None = None
    measurement_context: MeasurementContext
    points: list[FinalPropertySeriesPoint] = Field(min_length=1)
    coverage: SeriesCoverage
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ModelConfidence


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str = Field(min_length=1)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    object_id: str | None = None


class ValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "passed_with_warnings", "failed"]
    error_count: NonNegativeInt
    warning_count: NonNegativeInt


class CompletenessMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complete: NonNegativeInt
    total: NonNegativeInt
    ratio: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_ratio(self) -> "CompletenessMetric":
        if self.complete > self.total:
            raise ValueError("complete 不得大于 total")
        expected = self.complete / self.total if self.total else 1.0
        if abs(self.ratio - expected) > 1e-9:
            raise ValueError("ratio 必须等于 complete / total")
        return self


class QualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    properties_with_units: CompletenessMetric
    standard_process_steps: CompletenessMetric
    stage4_methods_with_characterization: CompletenessMetric
    objects_with_confidence: CompletenessMetric
    series_points_covered: CompletenessMetric = Field(
        default_factory=lambda: CompletenessMetric(
            complete=0,
            total=0,
            ratio=1.0,
        )
    )


class StageBilling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str = Field(min_length=1)
    provider: str | None = None
    model: str | None = None
    call_count: NonNegativeInt | None = None
    usage: TokenUsageSummary | None = None
    cost: StageCost


class CostSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: Literal["CNY"] = "CNY"
    status: Literal["complete", "partial"]
    stages: list[StageBilling] = Field(min_length=1)
    total_cost: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "CostSummary":
        expected = sum(
            (
                item.cost.total_cost
                for item in self.stages
                if item.cost.total_cost is not None
            ),
            start=Decimal(0),
        )
        if self.total_cost != expected:
            raise ValueError("total_cost 与各阶段费用之和不一致")
        expected_status = (
            "partial"
            if any(
                item.cost.status == "unavailable"
                for item in self.stages
            )
            else "complete"
        )
        if self.status != expected_status:
            raise ValueError("费用汇总状态与阶段状态不一致")
        return self


class Stage6Validation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(min_length=1)
    status: Literal["passed", "passed_with_warnings", "failed"]
    error_count: NonNegativeInt
    warning_count: NonNegativeInt
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    checked_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)
    quality_metrics: QualityMetrics | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> "Stage6Validation":
        if self.error_count != len(self.errors):
            raise ValueError("error_count 必须等于 errors 数量")
        if self.warning_count != len(self.warnings):
            raise ValueError("warning_count 必须等于 warnings 数量")
        expected_status = (
            "failed"
            if self.errors
            else "passed_with_warnings"
            if self.warnings
            else "passed"
        )
        if self.status != expected_status:
            raise ValueError("Stage 6 validation status 与问题数量不一致")
        return self


class FinalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.3", "1.4", "1.5", "1.6"] = "1.6"
    document_id: str = Field(min_length=1)
    paper: Paper
    material_mentions: list[FinalMaterialMention] = Field(default_factory=list)
    polymer_entities: list[FinalPolymerEntity] = Field(default_factory=list)
    unresolved_mention_ids: list[str] = Field(default_factory=list)
    samples: list[FinalSample] = Field(default_factory=list)
    process_steps: list[FinalProcessStep] = Field(default_factory=list)
    unresolved_entity_ids: list[str] = Field(default_factory=list)
    property_observations: list[
        FinalPropertyObservation | FinalStage5PropertyObservation
    ] = Field(default_factory=list)
    measurement_conditions: list[FinalMeasurementCondition] = Field(
        default_factory=list
    )
    unresolved_property_observations: list[
        FinalUnresolvedPropertyObservation
    ] = Field(default_factory=list)
    property_series: list[FinalPropertySeries] = Field(default_factory=list)
    characterizations: list[FinalCharacterization] = Field(default_factory=list)
    evidence: list[FinalEvidence] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    validation_summary: ValidationSummary
    cost_summary: CostSummary
    quality_metrics: QualityMetrics

    @model_validator(mode="after")
    def validate_references(self) -> "FinalDocument":
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id 不得重复")
        known_evidence = set(evidence_ids)
        evidence_consumers = [
            *self.material_mentions,
            *self.polymer_entities,
            *self.samples,
            *self.process_steps,
            *self.property_observations,
            *self.measurement_conditions,
            *self.unresolved_property_observations,
            *self.property_series,
            *self.characterizations,
        ]
        for item in evidence_consumers:
            unknown = sorted(set(item.evidence_ids) - known_evidence)
            if unknown:
                raise ValueError(f"对象引用了未知 evidence：{unknown}")

        mention_ids = {item.mention_id for item in self.material_mentions}
        entity_ids = {item.entity_id for item in self.polymer_entities}
        sample_ids = {item.sample_id for item in self.samples}
        condition_ids = {
            item.condition_id for item in self.measurement_conditions
        }
        property_ids = {
            item.property_id for item in self.property_observations
        } | {
            item.unresolved_id
            for item in self.unresolved_property_observations
        }
        series_ids = {item.series_id for item in self.property_series}
        characterization_ids = {
            item.characterization_id for item in self.characterizations
        }
        for item in self.polymer_entities:
            if set(item.resolved_from_mentions) - mention_ids:
                raise ValueError("PolymerEntity 引用了未知 mention")
            if item.variant_of and item.variant_of not in entity_ids:
                raise ValueError("PolymerEntity.variant_of 引用了未知 entity")
        for item in self.samples:
            if item.refers_to_entity and item.refers_to_entity not in entity_ids:
                raise ValueError("Sample 引用了未知 entity")
        for item in self.process_steps:
            if (
                set(item.input_sample_ids) | set(item.output_sample_ids)
            ) - sample_ids:
                raise ValueError("ProcessStep 引用了未知 sample")
        for item in self.property_observations:
            if isinstance(item, FinalPropertyObservation):
                if item.sample_id not in sample_ids:
                    raise ValueError("PropertyObservation 引用了未知 sample")
                if item.measurement_condition_id not in condition_ids:
                    raise ValueError(
                        "PropertyObservation 引用了未知 measurement condition"
                    )
                if _series_references(item) - series_ids:
                    raise ValueError("PropertyObservation 引用了未知 series")
            else:
                if item.sample_id and item.sample_id not in sample_ids:
                    raise ValueError("Stage 5 property 引用了未知 sample")
                if item.entity_id and item.entity_id not in entity_ids:
                    raise ValueError("Stage 5 property 引用了未知 entity")
                if set(item.sample_ids or []) - sample_ids:
                    raise ValueError("Stage 5 property 引用了未知 samples")
                if set(item.entity_ids or []) - entity_ids:
                    raise ValueError("Stage 5 property 引用了未知 entities")
                if item.characterization_id not in characterization_ids:
                    raise ValueError(
                        "Stage 5 property 引用了未知 characterization"
                    )
        for item in self.unresolved_property_observations:
            if _series_references(item) - series_ids:
                raise ValueError("Unresolved property 引用了未知 series")
        for item in self.property_series:
            if item.sample_id and item.sample_id not in sample_ids:
                raise ValueError("PropertySeries 引用了未知 sample")
            if item.entity_id and item.entity_id not in entity_ids:
                raise ValueError("PropertySeries 引用了未知 entity")
            for point in item.points:
                if point.sample_id and point.sample_id not in sample_ids:
                    raise ValueError("PropertySeries point 引用了未知 sample")
                if point.entity_id and point.entity_id not in entity_ids:
                    raise ValueError("PropertySeries point 引用了未知 entity")
                unknown = sorted(set(point.evidence_ids) - known_evidence)
                if unknown:
                    raise ValueError(
                        f"PropertySeries point 引用了未知 evidence：{unknown}"
                    )
                for coordinate in point.coordinates:
                    unknown = sorted(
                        set(coordinate.evidence_ids) - known_evidence
                    )
                    if unknown:
                        raise ValueError(
                            "PropertySeries coordinate 引用了未知 evidence："
                            f"{unknown}"
                        )
        for item in self.characterizations:
            if item.sample_id and item.sample_id not in sample_ids:
                raise ValueError("Characterization 引用了未知 sample")
            if item.entity_id and item.entity_id not in entity_ids:
                raise ValueError("Characterization 引用了未知 entity")
            if set(item.sample_ids or []) - sample_ids:
                raise ValueError("Characterization 引用了未知 samples")
            if set(item.entity_ids or []) - entity_ids:
                raise ValueError("Characterization 引用了未知 entities")
            if _series_references(item) - series_ids:
                raise ValueError("Characterization 引用了未知 series")
            if set(item.derived_property_ids) - property_ids:
                raise ValueError("Characterization 引用了未知 property")
        return self
