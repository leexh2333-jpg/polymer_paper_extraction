import tempfile
import unittest
from pathlib import Path

import yaml


from llm_client import load_pipeline_config
from prompt_loader import PromptError, PromptLoader
from schema.polymer_schema import MentionChunkResponse


class PromptLoaderTests(unittest.TestCase):
    def test_pipeline_prompt_ids_exist_and_stage1_renders(self) -> None:
        loader = PromptLoader()
        config = load_pipeline_config()
        referenced = {
            stage["prompt_id"]
            for stage in (config.get("stages") or {}).values()
            if isinstance(stage, dict) and stage.get("prompt_id")
        }

        self.assertTrue(referenced.issubset(loader.prompt_ids))
        rendered = loader.render_stage_prompt(
            "polymer.stage1.material_mention",
            MentionChunkResponse,
            expected_stage="stage1_material_mention",
            expected_output_schema="material_mention_schema.v2",
        )
        self.assertEqual(len(rendered.sha256), 64)
        self.assertNotIn("{{output_schema}}", rendered.text)
        self.assertIn('"mention_role"', rendered.text)

    def test_duplicate_prompt_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = (
                "---\n"
                "prompt_id: duplicate\n"
                "version: 1.0.0\n"
                "stage: test\n"
                "output_schema: test.v1\n"
                "---\n"
                "Body\n"
            )
            (root / "a.md").write_text(content, encoding="utf-8")
            (root / "b.md").write_text(content, encoding="utf-8")

            with self.assertRaises(PromptError):
                PromptLoader(root)
