import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from stage_minus1_reorganize_mineru import find_images_dir, reorganize_paper


class ReorganizeMineruTests(unittest.TestCase):
    def test_reorganize_preserves_markdown_and_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "mineru_output"
            output_root = root / "wenxian"
            ref_no = "reference_no_0000001"
            paper_dir = input_root / ref_no
            images_dir = paper_dir / "images"
            images_dir.mkdir(parents=True)

            (paper_dir / "uuid_content_list.json").write_text(
                json.dumps(
                    [
                        {
                            "type": "image",
                            "page_idx": 0,
                            "bbox": [1, 2, 3, 4],
                            "img_path": "images/hash.jpg",
                            "image_caption": ["Figure 2. Demo"],
                        },
                        {
                            "type": "image",
                            "page_idx": 0,
                            "bbox": [5, 6, 7, 8],
                            "img_path": "images/other_hash.jpg",
                            "image_caption": ["Figure 2. Demo"],
                        },
                        {
                            "type": "table",
                            "page_idx": 1,
                            "bbox": [1, 2, 3, 4],
                            "img_path": "",
                            "table_caption": [],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (paper_dir / "uuid_content_list_v2.json").write_text(
                json.dumps([[{"type": "image", "bbox": [1, 2, 3, 4]}]]),
                encoding="utf-8",
            )
            (paper_dir / "full.md").write_text(
                "![](images/hash.jpg)\n![](images/other_hash.jpg)",
                encoding="utf-8",
            )
            (paper_dir / "uuid_origin.pdf").write_bytes(b"%PDF-1.4")
            (images_dir / "hash.jpg").write_bytes(b"image")
            (images_dir / "other_hash.jpg").write_bytes(b"image")

            stats = reorganize_paper(input_root, output_root, ref_no)

            self.assertEqual(stats["errors"], [])
            self.assertEqual(stats["warnings"], [])
            self.assertTrue(stats["markdown_copied"])
            self.assertTrue(stats["content_v2_copied"])
            organized = output_root / ref_no
            content = json.loads((organized / "content.json").read_text(encoding="utf-8"))
            self.assertEqual(content["blocks"][0]["img_path"], "images/hash.jpg")
            self.assertEqual(
                content["blocks"][1]["img_path"],
                "images/other_hash.jpg",
            )
            self.assertEqual(content["blocks"][2]["img_path"], "")
            self.assertEqual(
                (organized / f"{ref_no}.md").read_text(encoding="utf-8"),
                "![](images/hash.jpg)\n![](images/other_hash.jpg)",
            )
            self.assertTrue((organized / "content_v2.json").is_file())
            self.assertTrue((organized / "images" / "hash.jpg").is_file())
            self.assertTrue((organized / "images" / "other_hash.jpg").is_file())
            self.assertTrue((organized / "origin.pdf").is_file())

    def test_legacy_images_directory_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paper_dir = Path(temp_dir) / "reference_no_0000001"
            legacy_dir = paper_dir / "reference_no_0000001_images"
            legacy_dir.mkdir(parents=True)

            self.assertEqual(
                find_images_dir(paper_dir, "reference_no_0000001"),
                legacy_dir,
            )


if __name__ == "__main__":
    unittest.main()
