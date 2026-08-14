import json
import sqlite3
import subprocess
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

from batch_runner import (
    PREVIEW_STAGE,
    PREVIEW_STAGES,
    STAGE4R_PREVIEW_STAGE,
    STAGES,
    START_STAGE_CHOICES,
    VALIDATE_EXISTING_INPUTS,
    BatchStateStore,
    RunnerSettings,
    StageSpec,
    build_batch_acceptance,
    build_run_summary,
    build_stage_command,
    run_document,
    run_stage_process,
    select_document_paths,
    strict_stage_window,
)


class StageWindowTests(unittest.TestCase):
    def test_stage2_window_requires_only_stage0_and_stage1_outputs(self) -> None:
        stages, required = strict_stage_window("stage2_polymer_entity")

        self.assertEqual(stages, STAGES[2:])
        self.assertEqual(required, ("stage0_blocks.json", "stage1_mentions.json"))
        self.assertIn("stage2_polymer_entity", START_STAGE_CHOICES)

    def test_default_window_keeps_the_full_strict_pipeline(self) -> None:
        stages, required = strict_stage_window(None)

        self.assertEqual(stages, STAGES)
        self.assertEqual(required, ())


class BatchStateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = BatchStateStore(self.root / "batch.sqlite3")
        self.document_path = self.root / "reference_no_0000001_document.json"
        self.document_path.write_text("{}", encoding="utf-8")
        self.store.register_documents([self.document_path])
        self.store.prepare_run("run-1", ["reference_no_0000001"])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_claim_is_atomic_and_same_run_does_not_repeat_failure(self) -> None:
        claimed = self.store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["ref_no"], "reference_no_0000001")
        self.assertIsNone(
            self.store.claim_next(
                run_id="run-1",
                worker_id="worker-2",
                allowed_statuses=["pending", "failed"],
                lease_seconds=60,
            )
        )

        self.store.finish_document(
            "reference_no_0000001",
            worker_id="worker-1",
            status="failed",
            error_message="test failure",
        )
        self.assertIsNone(
            self.store.claim_next(
                run_id="run-1",
                worker_id="worker-2",
                allowed_statuses=["failed"],
                lease_seconds=60,
            )
        )
        self.store.prepare_run("run-2", ["reference_no_0000001"])
        retried = self.store.claim_next(
            run_id="run-2",
            worker_id="worker-2",
            allowed_statuses=["failed"],
            lease_seconds=60,
        )
        self.assertIsNotNone(retried)

    def test_stale_running_attempt_becomes_interrupted_uncertain(self) -> None:
        self.store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=1,
        )
        attempt_id = self.store.start_attempt(
            run_id="run-1",
            ref_no="reference_no_0000001",
            stage="stage4_property",
            worker_id="worker-1",
            lease_seconds=1,
            stdout_path=self.root / "stdout.log",
            stderr_path=self.root / "stderr.log",
        )

        recovered = self.store.recover_stale(now=10**12)

        self.assertEqual(recovered, 1)
        self.assertEqual(
            self.store.summary(),
            {"interrupted_uncertain": 1},
        )
        with closing(sqlite3.connect(self.store.path)) as connection:
            status = connection.execute(
                "SELECT status FROM stage_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()[0]
        self.assertEqual(status, "interrupted_uncertain")
        self.assertIsNone(
            self.store.claim_next(
                run_id="run-2",
                worker_id="worker-2",
                allowed_statuses=["pending"],
                lease_seconds=60,
            )
        )
        self.store.prepare_run("run-2", ["reference_no_0000001"])
        retried = self.store.claim_next(
            run_id="run-2",
            worker_id="worker-2",
            allowed_statuses=["interrupted_uncertain"],
            lease_seconds=60,
        )
        self.assertIsNotNone(retried)

    def test_run_can_only_claim_selected_documents(self) -> None:
        second_path = self.root / "reference_no_0000002_document.json"
        second_path.write_text("{}", encoding="utf-8")
        self.store.register_documents([second_path])
        self.store.prepare_run("run-2", ["reference_no_0000002"])

        claimed = self.store.claim_next(
            run_id="run-2",
            worker_id="worker-2",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["ref_no"], "reference_no_0000002")


class DocumentSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        for ref_no in ("reference_no_0000001", "reference_no_0000002"):
            (self.root / f"{ref_no}_document.json").write_text(
                "{}", encoding="utf-8"
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_repeatable_ref_no_preserves_requested_order_and_deduplicates(self) -> None:
        paths = select_document_paths(
            self.root,
            ref_nos=[
                "reference_no_0000002",
                "reference_no_0000001",
                "reference_no_0000002",
            ],
        )

        self.assertEqual(
            [path.name for path in paths],
            [
                "reference_no_0000002_document.json",
                "reference_no_0000001_document.json",
            ],
        )

    def test_ref_list_ignores_blank_lines_and_comments(self) -> None:
        ref_list = self.root / "refs.txt"
        ref_list.write_text(
            "# regression set\nreference_no_0000001\n\n"
            "reference_no_0000002_document.json\n",
            encoding="utf-8",
        )

        paths = select_document_paths(self.root, ref_list_path=ref_list)

        self.assertEqual(len(paths), 2)


class StageCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.settings = RunnerSettings(
            config_path=root / "pipeline.yaml",
            input_dir=root / "input",
            output_dir=root / "output",
            logs_dir=root / "logs",
            force=True,
            heartbeat_seconds=10,
            lease_seconds=90,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_stage0_uses_document_input_directory(self) -> None:
        command = build_stage_command(
            STAGES[0],
            ref_no="reference_no_0000001",
            settings=self.settings,
        )

        self.assertIn("--input-dir", command)
        self.assertIn("--output-dir", command)
        self.assertIn("--force", command)

    def test_stage6_does_not_receive_unsupported_force_argument(self) -> None:
        command = build_stage_command(
            STAGES[-1],
            ref_no="reference_no_0000001",
            settings=self.settings,
        )

        self.assertIn("--input-root", command)
        self.assertNotIn("--force", command)

    def test_stage4r_is_preview_only_and_runs_between_stage4_and_stage5(self) -> None:
        preview_ids = [spec.stage_id for spec in PREVIEW_STAGES]
        strict_ids = [spec.stage_id for spec in STAGES]

        self.assertNotIn(STAGE4R_PREVIEW_STAGE.stage_id, strict_ids)
        self.assertEqual(
            preview_ids[preview_ids.index("stage4_property") + 1],
            STAGE4R_PREVIEW_STAGE.stage_id,
        )
        self.assertEqual(
            preview_ids[preview_ids.index(STAGE4R_PREVIEW_STAGE.stage_id) + 1],
            "stage5_characterization",
        )

    def test_stage4r_command_applies_recovery_in_preview_output(self) -> None:
        command = build_stage_command(
            STAGE4R_PREVIEW_STAGE,
            ref_no="reference_no_0000001",
            settings=self.settings,
        )

        self.assertIn("--input-root", command)
        self.assertIn("--output-root", command)
        self.assertIn("--config", command)
        self.assertIn("--force", command)
        self.assertIn("--apply", command)
        self.assertIn("--allow-filled-up-sample-binding", command)

    def test_preview_stage_uses_external_candidate_publisher(self) -> None:
        command = build_stage_command(
            PREVIEW_STAGE,
            ref_no="reference_no_0000001",
            settings=self.settings,
        )

        self.assertTrue(Path(command[1]).is_absolute())
        self.assertEqual(Path(command[1]).name, "publish_candidate.py")
        self.assertEqual(
            PREVIEW_STAGE.output_names,
            ("candidate.json", "report_candidate.html"),
        )

    def test_preview_stage2_uses_relaxed_validation(self) -> None:
        settings = RunnerSettings(
            config_path=self.settings.config_path,
            input_dir=self.settings.input_dir,
            output_dir=self.settings.output_dir,
            logs_dir=self.settings.logs_dir,
            force=False,
            heartbeat_seconds=10,
            lease_seconds=90,
            preview=True,
        )

        command = build_stage_command(
            STAGES[2],
            ref_no="reference_no_0000001",
            settings=settings,
        )

        self.assertIn("--preview-relaxed", command)

    def test_preview_stage1_uses_relaxed_validation(self) -> None:
        settings = RunnerSettings(
            config_path=self.settings.config_path,
            input_dir=self.settings.input_dir,
            output_dir=self.settings.output_dir,
            logs_dir=self.settings.logs_dir,
            force=False,
            heartbeat_seconds=10,
            lease_seconds=90,
            preview=True,
        )

        command = build_stage_command(
            STAGES[1],
            ref_no="reference_no_0000001",
            settings=settings,
            extra_args=("--replay-failure", "--force"),
        )

        self.assertIn("--replay-failure", command)
        self.assertIn("--preview-relaxed", command)

    def test_preview_runs_stage6_before_publishing_candidate(self) -> None:
        """Preview 也要跑 Stage 6，只是带 --preview-relaxed。"""
        preview_ids = [spec.stage_id for spec in PREVIEW_STAGES]

        self.assertIn("stage6_validate_merge", preview_ids)
        self.assertEqual(
            preview_ids[preview_ids.index("stage6_validate_merge") + 1],
            PREVIEW_STAGE.stage_id,
        )

    def test_preview_stage6_uses_relaxed_validation(self) -> None:
        settings = RunnerSettings(
            config_path=self.settings.config_path,
            input_dir=self.settings.input_dir,
            output_dir=self.settings.output_dir,
            logs_dir=self.settings.logs_dir,
            force=False,
            heartbeat_seconds=10,
            lease_seconds=90,
            preview=True,
        )

        command = build_stage_command(
            STAGES[6],
            ref_no="reference_no_0000001",
            settings=settings,
        )

        self.assertIn("--preview-relaxed", command)

    def test_strict_stage6_never_receives_relaxed_flag(self) -> None:
        command = build_stage_command(
            STAGES[6],
            ref_no="reference_no_0000001",
            settings=self.settings,
        )

        self.assertNotIn("--preview-relaxed", command)

    def test_preview_stage3_uses_relaxed_validation(self) -> None:
        settings = RunnerSettings(
            config_path=self.settings.config_path,
            input_dir=self.settings.input_dir,
            output_dir=self.settings.output_dir,
            logs_dir=self.settings.logs_dir,
            force=False,
            heartbeat_seconds=10,
            lease_seconds=90,
            preview=True,
        )

        command = build_stage_command(
            STAGES[3],
            ref_no="reference_no_0000001",
            settings=settings,
        )

        self.assertIn("--preview-relaxed", command)

    def test_preview_failure_is_replayed_before_partial_fallback(self) -> None:
        root = Path(self.temporary_directory.name)
        store = BatchStateStore(root / "recovered.sqlite3")
        document_path = root / "reference_no_0000001_document.json"
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-recovered", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-recovered",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        failure_dir = self.settings.output_dir / "reference_no_0000001"
        failure_dir.mkdir(parents=True)
        (failure_dir / "stage2_failure.json").write_text(
            "{}",
            encoding="utf-8",
        )
        failed_stage = StageSpec(
            "stage2_polymer_entity",
            "stage2_polymer_entity.py",
            ("stage2_entities.json",),
            True,
        )

        replay_settings = RunnerSettings(
            config_path=self.settings.config_path,
            input_dir=self.settings.input_dir,
            output_dir=self.settings.output_dir,
            logs_dir=self.settings.logs_dir,
            force=False,
            heartbeat_seconds=10,
            lease_seconds=90,
            preview=True,
        )
        with patch(
            "batch_runner.run_stage_process",
            side_effect=[(True, None), (True, None)],
        ) as process_call:
            succeeded = run_document(
                store=store,
                run_id="run-recovered",
                worker_id="worker-1",
                ref_no="reference_no_0000001",
                settings=replay_settings,
                llm_semaphore=threading.Semaphore(1),
                stop_event=threading.Event(),
                stage_specs=(failed_stage, PREVIEW_STAGE),
            )

        self.assertTrue(succeeded)
        self.assertEqual(process_call.call_count, 2)
        self.assertEqual(
            process_call.call_args_list[0].kwargs["extra_args"],
            ("--replay-failure", "--preview-relaxed", "--force"),
        )
        with closing(sqlite3.connect(store.path)) as connection:
            status = connection.execute(
                "SELECT status FROM documents WHERE ref_no = ?",
                ("reference_no_0000001",),
            ).fetchone()[0]
        self.assertEqual(status, "candidate_complete")

    def test_stage3_preview_replay_uses_relaxed_flag(self) -> None:
        root = Path(self.temporary_directory.name)
        store = BatchStateStore(root / "stage3-recovered.sqlite3")
        document_path = root / "reference_no_0000002_document.json"
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-stage3", ["reference_no_0000002"])
        store.claim_next(
            run_id="run-stage3",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        failure_dir = self.settings.output_dir / "reference_no_0000002"
        failure_dir.mkdir(parents=True)
        (failure_dir / "stage3_failure.json").write_text(
            "{}", encoding="utf-8"
        )
        failed_stage = StageSpec(
            "stage3_sample_process",
            "stage3_sample_process.py",
            ("stage3_process.json",),
            True,
        )

        replay_settings = RunnerSettings(
            config_path=self.settings.config_path,
            input_dir=self.settings.input_dir,
            output_dir=self.settings.output_dir,
            logs_dir=self.settings.logs_dir,
            force=False,
            heartbeat_seconds=10,
            lease_seconds=90,
            preview=True,
        )
        with patch(
            "batch_runner.run_stage_process",
            side_effect=[(True, None), (True, None)],
        ) as process_call:
            succeeded = run_document(
                store=store,
                run_id="run-stage3",
                worker_id="worker-1",
                ref_no="reference_no_0000002",
                settings=replay_settings,
                llm_semaphore=threading.Semaphore(1),
                stop_event=threading.Event(),
                stage_specs=(failed_stage, PREVIEW_STAGE),
            )

        self.assertTrue(succeeded)
        self.assertEqual(
            process_call.call_args_list[0].kwargs["extra_args"],
            ("--replay-failure", "--preview-relaxed", "--force"),
        )

    def test_preview_failure_publishes_partial_candidate(self) -> None:
        root = Path(self.temporary_directory.name)
        store = BatchStateStore(root / "partial.sqlite3")
        document_path = root / "reference_no_0000001_document.json"
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-partial", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-partial",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        failed_stage = StageSpec(
            "stage2_polymer_entity",
            "stage2_polymer_entity.py",
            ("stage2_entities.json",),
            True,
        )

        with patch(
            "batch_runner.run_stage_process",
            side_effect=[(False, "schema mismatch"), (True, None)],
        ) as process_call:
            succeeded = run_document(
                store=store,
                run_id="run-partial",
                worker_id="worker-1",
                ref_no="reference_no_0000001",
                settings=self.settings,
                llm_semaphore=threading.Semaphore(1),
                stop_event=threading.Event(),
                stage_specs=(failed_stage, PREVIEW_STAGE),
            )

        self.assertFalse(succeeded)
        self.assertEqual(process_call.call_count, 2)
        self.assertEqual(
            process_call.call_args_list[1].kwargs["spec"],
            PREVIEW_STAGE,
        )
        with closing(sqlite3.connect(store.path)) as connection:
            status = connection.execute(
                "SELECT status FROM documents WHERE ref_no = ?",
                ("reference_no_0000001",),
            ).fetchone()[0]
        self.assertEqual(status, "candidate_partial")

    def test_stage_process_heartbeats_and_persists_success(self) -> None:
        store = BatchStateStore(Path(self.temporary_directory.name) / "batch.sqlite3")
        document_path = (
            Path(self.temporary_directory.name)
            / "reference_no_0000001_document.json"
        )
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-1", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        output_path = (
            self.settings.output_dir
            / "reference_no_0000001"
            / "stage0_blocks.json"
        )
        output_path.parent.mkdir(parents=True)
        output_path.write_text("{}", encoding="utf-8")
        process = Mock()
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="stage0", timeout=10),
            0,
        ]

        with patch("batch_runner.subprocess.Popen", return_value=process):
            succeeded, error = run_stage_process(
                store=store,
                run_id="run-1",
                worker_id="worker-1",
                ref_no="reference_no_0000001",
                spec=STAGES[0],
                settings=self.settings,
                stop_event=threading.Event(),
            )

        self.assertTrue(succeeded)
        self.assertIsNone(error)
        with closing(sqlite3.connect(store.path)) as connection:
            row = connection.execute(
                "SELECT status, exit_code FROM stage_attempts"
            ).fetchone()
        self.assertEqual(row, ("succeeded", 0))

    def test_stop_while_waiting_for_llm_slot_does_not_start_stage(self) -> None:
        store = BatchStateStore(Path(self.temporary_directory.name) / "batch.sqlite3")
        document_path = (
            Path(self.temporary_directory.name)
            / "reference_no_0000001_document.json"
        )
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-1", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        stop_event = threading.Event()

        class StopSemaphore:
            def acquire(self, timeout: float) -> bool:
                stop_event.set()
                return False

            def release(self) -> None:
                raise AssertionError("release must not be called")

        llm_stage = StageSpec(
            "stage1_material_mention",
            "stage1_material_mention.py",
            ("stage1_mentions.json",),
            True,
        )
        with patch("batch_runner.STAGES", (llm_stage,)), patch(
            "batch_runner.run_stage_process"
        ) as process_call:
            succeeded = run_document(
                store=store,
                run_id="run-1",
                worker_id="worker-1",
                ref_no="reference_no_0000001",
                settings=self.settings,
                llm_semaphore=StopSemaphore(),
                stop_event=stop_event,
            )

        self.assertFalse(succeeded)
        process_call.assert_not_called()
        self.assertEqual(store.summary(), {"interrupted_uncertain": 1})

    def test_waiting_for_llm_slot_renews_document_lease(self) -> None:
        store = BatchStateStore(Path(self.temporary_directory.name) / "batch.sqlite3")
        document_path = (
            Path(self.temporary_directory.name)
            / "reference_no_0000001_document.json"
        )
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-1", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )

        class DelayedSemaphore:
            def __init__(self) -> None:
                self.calls = 0
                self.released = False

            def acquire(self, timeout: float) -> bool:
                self.calls += 1
                return self.calls >= 2

            def release(self) -> None:
                self.released = True

        semaphore = DelayedSemaphore()
        llm_stage = StageSpec(
            "stage1_material_mention",
            "stage1_material_mention.py",
            ("stage1_mentions.json",),
            True,
        )
        with patch("batch_runner.STAGES", (llm_stage,)), patch(
            "batch_runner.run_stage_process", return_value=(True, None)
        ) as process_call, patch.object(
            store,
            "heartbeat_document",
            wraps=store.heartbeat_document,
        ) as heartbeat:
            succeeded = run_document(
                store=store,
                run_id="run-1",
                worker_id="worker-1",
                ref_no="reference_no_0000001",
                settings=self.settings,
                llm_semaphore=semaphore,
                stop_event=threading.Event(),
            )

        self.assertTrue(succeeded)
        self.assertGreaterEqual(heartbeat.call_count, 2)
        process_call.assert_called_once()
        self.assertTrue(semaphore.released)

    def test_validate_existing_runs_only_stage6_without_llm_slot(self) -> None:
        store = BatchStateStore(Path(self.temporary_directory.name) / "batch.sqlite3")
        document_path = (
            Path(self.temporary_directory.name)
            / "reference_no_0000001_document.json"
        )
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-1", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        output_base = self.settings.output_dir / "reference_no_0000001"
        output_base.mkdir(parents=True)
        for name in VALIDATE_EXISTING_INPUTS:
            (output_base / name).write_text("{}", encoding="utf-8")

        class NoLLMSemaphore:
            def acquire(self, timeout: float) -> bool:
                raise AssertionError("Stage 6 不应申请 LLM 并发槽")

            def release(self) -> None:
                raise AssertionError("Stage 6 不应释放 LLM 并发槽")

        with patch(
            "batch_runner.run_stage_process", return_value=(True, None)
        ) as process_call:
            succeeded = run_document(
                store=store,
                run_id="run-1",
                worker_id="worker-1",
                ref_no="reference_no_0000001",
                settings=self.settings,
                llm_semaphore=NoLLMSemaphore(),
                stop_event=threading.Event(),
                stage_specs=(STAGES[-1],),
                required_existing_outputs=VALIDATE_EXISTING_INPUTS,
            )

        self.assertTrue(succeeded)
        process_call.assert_called_once()
        self.assertEqual(
            process_call.call_args.kwargs["spec"].stage_id,
            "stage6_validate_merge",
        )
        self.assertEqual(store.summary(), {"succeeded": 1})

    def test_validate_existing_missing_input_does_not_start_stage(self) -> None:
        store = BatchStateStore(Path(self.temporary_directory.name) / "batch.sqlite3")
        document_path = (
            Path(self.temporary_directory.name)
            / "reference_no_0000001_document.json"
        )
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-1", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        output_base = self.settings.output_dir / "reference_no_0000001"
        output_base.mkdir(parents=True)
        for name in VALIDATE_EXISTING_INPUTS:
            if name != "stage5_characterizations.json":
                (output_base / name).write_text("{}", encoding="utf-8")

        with patch("batch_runner.run_stage_process") as process_call:
            succeeded = run_document(
                store=store,
                run_id="run-1",
                worker_id="worker-1",
                ref_no="reference_no_0000001",
                settings=self.settings,
                llm_semaphore=Mock(),
                stop_event=threading.Event(),
                stage_specs=(STAGES[-1],),
                required_existing_outputs=VALIDATE_EXISTING_INPUTS,
            )

        self.assertFalse(succeeded)
        process_call.assert_not_called()
        self.assertEqual(store.summary(), {"failed": 1})

    def test_stage6_only_summary_has_no_new_charge(self) -> None:
        root = Path(self.temporary_directory.name)
        store = BatchStateStore(root / "batch.sqlite3")
        document_path = root / "reference_no_0000001_document.json"
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-1", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        attempt_id = store.start_attempt(
            run_id="run-1",
            ref_no="reference_no_0000001",
            stage="stage6_validate_merge",
            worker_id="worker-1",
            lease_seconds=60,
            stdout_path=root / "stage6.stdout.log",
            stderr_path=root / "stage6.stderr.log",
        )
        store.finish_attempt(
            attempt_id,
            status="succeeded",
            exit_code=0,
            result_kind="executed",
        )
        store.finish_document(
            "reference_no_0000001",
            worker_id="worker-1",
            status="succeeded",
        )

        report = build_run_summary(
            store=store,
            run_id="run-1",
            output_dir=self.settings.output_dir,
            settings={"validate_existing": True},
        )

        self.assertEqual(report["billing"]["status"], "no_new_charge")
        self.assertEqual(report["billing"]["call_count"], 0)
        self.assertEqual(report["billing"]["cost"]["total_cost"], "0")

    def test_run_summary_aggregates_stage4_coordinate_column_warnings(self) -> None:
        root = Path(self.temporary_directory.name)
        store = BatchStateStore(root / "batch.sqlite3")
        document_path = root / "reference_no_0000001_document.json"
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-1", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        attempt_id = store.start_attempt(
            run_id="run-1",
            ref_no="reference_no_0000001",
            stage="stage4_property",
            worker_id="worker-1",
            lease_seconds=60,
            stdout_path=root / "stage4.stdout.log",
            stderr_path=root / "stage4.stderr.log",
        )
        store.finish_attempt(
            attempt_id,
            status="succeeded",
            exit_code=0,
            result_kind="cached",
        )
        output_base = self.settings.output_dir / "reference_no_0000001"
        output_base.mkdir(parents=True)
        (output_base / "stage4_properties.json").write_text(
            json.dumps({
                "warnings": [{
                    "code": "table_property_column_represented_as_coordinate",
                    "columns": [{
                        "table_id": "T_4_0",
                        "column_index": 2,
                        "column_label": "Mn",
                        "value_count": 2,
                        "coordinate_cell_count": 2,
                        "unrepresented_cell_count": 0,
                    }],
                }],
            }),
            encoding="utf-8",
        )

        report = build_run_summary(
            store=store,
            run_id="run-1",
            output_dir=self.settings.output_dir,
            settings={},
        )

        aggregate = report["artifact_warnings"][
            "table_property_column_represented_as_coordinate"
        ]
        self.assertEqual(aggregate["column_count"], 1)
        self.assertEqual(aggregate["columns"][0]["ref_no"], "reference_no_0000001")
        self.assertEqual(aggregate["columns"][0]["column_label"], "Mn")

    def test_zero_call_failure_is_audited_as_no_new_charge(self) -> None:
        root = Path(self.temporary_directory.name)
        store = BatchStateStore(root / "batch.sqlite3")
        document_path = root / "reference_no_0000001_document.json"
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-1", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        attempt_id = store.start_attempt(
            run_id="run-1",
            ref_no="reference_no_0000001",
            stage="stage1_material_mention",
            worker_id="worker-1",
            lease_seconds=60,
            stdout_path=root / "stage1.stdout.log",
            stderr_path=root / "stage1.stderr.log",
        )
        output_base = self.settings.output_dir / "reference_no_0000001"
        output_base.mkdir(parents=True)
        (output_base / "stage1_failure.json").write_text(
            json.dumps({
                "call_count": 0,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "billable_input_tokens": 0,
                    "total_tokens": 0,
                },
                "cost": {
                    "status": "not_applicable",
                    "currency": "CNY",
                    "input_per_million": "13.51",
                    "output_per_million": "66.5",
                    "input_cost": "0",
                    "output_cost": "0",
                    "total_cost": "0",
                },
                "raw_response": None,
            }),
            encoding="utf-8",
        )
        store.finish_attempt(
            attempt_id,
            status="failed",
            exit_code=1,
            result_kind="executed",
        )

        report = build_run_summary(
            store=store,
            run_id="run-1",
            output_dir=self.settings.output_dir,
            settings={},
        )

        self.assertEqual(report["billing"]["status"], "no_new_charge")
        self.assertEqual(report["billing"]["call_count"], 0)
        self.assertEqual(report["billing"]["unknown_attempts"], [])

    def test_run_summary_counts_executed_cost_but_not_cached_cost(self) -> None:
        root = Path(self.temporary_directory.name)
        store = BatchStateStore(root / "batch.sqlite3")
        document_path = root / "reference_no_0000001_document.json"
        document_path.write_text("{}", encoding="utf-8")
        store.register_documents([document_path])
        store.prepare_run("run-1", ["reference_no_0000001"])
        store.claim_next(
            run_id="run-1",
            worker_id="worker-1",
            allowed_statuses=["pending"],
            lease_seconds=60,
        )
        output_base = self.settings.output_dir / "reference_no_0000001"
        output_base.mkdir(parents=True)

        executed_id = store.start_attempt(
            run_id="run-1",
            ref_no="reference_no_0000001",
            stage="stage2_polymer_entity",
            worker_id="worker-1",
            lease_seconds=60,
            stdout_path=root / "stage2.stdout.log",
            stderr_path=root / "stage2.stderr.log",
        )
        (output_base / "stage2_entities.json").write_text(
            json.dumps(
                {
                    "provenance": {
                        "call_count": 1,
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "billable_input_tokens": 100,
                            "total_tokens": 120,
                        },
                        "cost": {
                            "status": "calculated",
                            "currency": "CNY",
                            "input_per_million": "13.51",
                            "output_per_million": "66.5",
                            "input_cost": "0.10",
                            "output_cost": "0.20",
                            "total_cost": "0.30",
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        store.finish_attempt(
            executed_id,
            status="succeeded",
            exit_code=0,
            result_kind="executed",
        )

        cached_id = store.start_attempt(
            run_id="run-1",
            ref_no="reference_no_0000001",
            stage="stage3_sample_process",
            worker_id="worker-1",
            lease_seconds=60,
            stdout_path=root / "stage3.stdout.log",
            stderr_path=root / "stage3.stderr.log",
        )
        store.finish_attempt(
            cached_id,
            status="succeeded",
            exit_code=0,
            result_kind="cached",
        )
        failed_id = store.start_attempt(
            run_id="run-1",
            ref_no="reference_no_0000001",
            stage="stage1_material_mention",
            worker_id="worker-1",
            lease_seconds=60,
            stdout_path=root / "stage1.stdout.log",
            stderr_path=root / "stage1.stderr.log",
        )
        (output_base / "stage1_failure.json").write_text(
            json.dumps(
                {
                    "usage": {
                        "input_tokens": 50,
                        "output_tokens": 10,
                        "billable_input_tokens": 50,
                        "total_tokens": 60,
                    },
                    "cost": {
                        "status": "calculated",
                        "currency": "CNY",
                        "input_per_million": "13.51",
                        "output_per_million": "66.5",
                        "input_cost": "0.10",
                        "output_cost": "0.30",
                        "total_cost": "0.40",
                    },
                }
            ),
            encoding="utf-8",
        )
        store.finish_attempt(
            failed_id,
            status="failed",
            exit_code=1,
            result_kind="executed",
        )
        store.finish_document(
            "reference_no_0000001",
            worker_id="worker-1",
            status="succeeded",
        )

        report = build_run_summary(
            store=store,
            run_id="run-1",
            output_dir=self.settings.output_dir,
            settings={},
        )

        self.assertEqual(report["billing"]["status"], "calculated")
        self.assertEqual(report["billing"]["charged_attempt_count"], 2)
        self.assertEqual(report["billing"]["cached_attempt_count"], 1)
        self.assertEqual(report["billing"]["call_count"], 2)
        self.assertEqual(report["billing"]["cost"]["total_cost"], "0.70")


if __name__ == "__main__":
    unittest.main()


class BatchAcceptanceTests(unittest.TestCase):
    def test_preview_validate_existing_accepts_succeeded(self) -> None:
        accepted = build_batch_acceptance(
            {"succeeded": 20},
            preview=True,
            validate_existing=True,
        )

        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["expected_status"], "succeeded")

    def test_preview_accepts_only_all_candidate_complete(self) -> None:
        accepted = build_batch_acceptance(
            {"candidate_complete": 20},
            preview=True,
        )

        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["accepted_count"], 20)
        self.assertEqual(accepted["blocking_statuses"], {})

    def test_preview_partial_is_blocking(self) -> None:
        accepted = build_batch_acceptance(
            {"candidate_complete": 19, "candidate_partial": 1},
            preview=True,
        )

        self.assertFalse(accepted["accepted"])
        self.assertEqual(
            accepted["blocking_statuses"],
            {"candidate_partial": 1},
        )

    def test_strict_accepts_only_all_succeeded(self) -> None:
        accepted = build_batch_acceptance(
            {"succeeded": 20},
            preview=False,
        )

        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["expected_status"], "succeeded")
