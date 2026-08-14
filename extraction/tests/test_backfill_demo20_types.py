import unittest

from schema.polymer_schema import PolymerEntity, Sample, Stage2Document, Stage2Provenance
from tools.backfill_demo20_types import align_sample_polymer_types


class TypeBackfillTests(unittest.TestCase):
    def test_sample_types_are_aligned_from_linked_entity(self) -> None:
        provenance = Stage2Provenance.model_validate({
            "provider": "test",
            "model": "test",
            "models": ["test"],
            "prompt_id": "test",
            "prompt_version": "1",
            "prompt_sha256": "a" * 64,
            "input_hash": "b" * 64,
            "model_config_hash": "c" * 64,
            "cache_key": "d" * 64,
            "output_schema_version": "polymer_entity_schema.v3",
            "implementation_version": "1.4.0",
            "context_block_count": 1,
            "context_chars": 1,
            "call_count": 1,
        })
        entity = PolymerEntity.model_validate({
            "entity_id": "pe001",
            "polymer_name": "random copolymer",
            "polymer_type": "copolymer",
            "copolymer_type": "ran",
            "representation_status": "expert_review_required",
            "source_names": ["random copolymer"],
            "resolved_from_mentions": ["m001"],
            "evidence": {
                "block_id": "P_0_0",
                "page": 0,
                "source_type": "text",
                "source_sentence": "random copolymer",
            },
        })
        stage2 = Stage2Document(
            document_id="reference_no_0000001",
            polymer_entities=[entity],
            provenance=provenance,
        )
        sample = Sample.model_validate({
            "sample_id": "s001",
            "sample_kind": "processed_material",
            "refers_to_entity": "pe001",
            "polymer_name": "random copolymer",
            "evidence": {
                "block_id": "P_0_0",
                "page": 0,
                "source_type": "text",
                "source_sentence": "random copolymer",
            },
        })

        samples, items = align_sample_polymer_types([sample], stage2)

        self.assertEqual(samples[0].polymer_type, "copolymer")
        self.assertEqual(samples[0].copolymer_type, "ran")
        self.assertEqual(items[0]["entity_id"], "pe001")


if __name__ == "__main__":
    unittest.main()
