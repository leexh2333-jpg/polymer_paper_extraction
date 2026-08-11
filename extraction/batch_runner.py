"""可恢复的文献级并行抽取调度器。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm_client import DEFAULT_CONFIG_PATH, load_pipeline_config


EXTRACTION_ROOT = Path(__file__).resolve().parent
STAGES_ROOT = EXTRACTION_ROOT / "stages"
PREVIEW_SCRIPT = EXTRACTION_ROOT.parent / "preview" / "publish_candidate.py"
LLM_STAGE_OUTPUTS = {
    "stage1_material_mention": "stage1_mentions.json",
    "stage2_polymer_entity": "stage2_entities.json",
    "stage3_sample_process": "stage3_process.json",
    "stage4_property": "stage4_properties.json",
    "stage5_characterization": "stage5_characterizations.json",
}
LLM_STAGE_FAILURES = {
    "stage1_material_mention": "stage1_failure.json",
    "stage2_polymer_entity": "stage2_failure.json",
    "stage3_sample_process": "stage3_failure.json",
    "stage4_property": "stage4_failure.json",
    "stage5_characterization": "stage5_failure.json",
}


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    script_name: str
    output_names: tuple[str, ...]
    uses_llm: bool


STAGES = (
    StageSpec(
        "stage0_load_document",
        "stage0_load_document.py",
        ("stage0_blocks.json",),
        False,
    ),
    StageSpec(
        "stage1_material_mention",
        "stage1_material_mention.py",
        ("stage1_mentions.json",),
        True,
    ),
    StageSpec(
        "stage2_polymer_entity",
        "stage2_polymer_entity.py",
        ("stage2_entities.json",),
        True,
    ),
    StageSpec(
        "stage3_sample_process",
        "stage3_sample_process.py",
        ("stage3_process.json",),
        True,
    ),
    StageSpec(
        "stage4_property",
        "stage4_property.py",
        ("stage4_properties.json",),
        True,
    ),
    StageSpec(
        "stage5_characterization",
        "stage5_characterization.py",
        ("stage5_characterizations.json",),
        True,
    ),
    StageSpec(
        "stage6_validate_merge",
        "stage6_validate_merge.py",
        ("stage6_validation.json", "final.json", "report.html"),
        False,
    ),
)

STAGE4R_PREVIEW_STAGE = StageSpec(
    "stage4r_table_recovery",
    "stage4r_table_recovery.py",
    ("stage4r_recovery.json", "stage4_properties.recovery_preview.json"),
    False,
)
PREVIEW_STAGE = StageSpec(
    "candidate_publish",
    str(PREVIEW_SCRIPT),
    ("candidate.json", "report_candidate.html"),
    False,
)
PREVIEW_STAGES = (
    *STAGES[:5],
    STAGE4R_PREVIEW_STAGE,
    STAGES[5],
    # Stage 6 在 preview 下带 --preview-relaxed 跑：证据的表示层差异降级为
    # warning，仍产出 final.json / report.html。跑不过的篇目会走下面的
    # candidate_partial 分支，照样发布 candidate.json，不影响整批推进。
    STAGES[6],
    PREVIEW_STAGE,
)
PREVIEW_RECOVERABLE_FAILURES = {
    "stage1_material_mention": "stage1_failure.json",
    "stage2_polymer_entity": "stage2_failure.json",
    "stage3_sample_process": "stage3_failure.json",
    "stage4_property": "stage4_failure.json",
    "stage5_characterization": "stage5_failure.json",
}

VALIDATE_EXISTING_INPUTS = tuple(
    name for spec in STAGES[:-1] for name in spec.output_names
)


class BatchRunnerError(RuntimeError):
    """批处理配置或执行错误。"""


class BatchStateStore:
    """使用短事务维护跨进程可见的文献和阶段状态。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    ref_no TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT,
                    owner_id TEXT,
                    lease_until REAL,
                    last_run_id TEXT,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stage_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    ref_no TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    lease_until REAL NOT NULL,
                    ended_at REAL,
                    exit_code INTEGER,
                    result_kind TEXT,
                    stdout_path TEXT,
                    stderr_path TEXT,
                    error_message TEXT,
                    FOREIGN KEY(ref_no) REFERENCES documents(ref_no)
                );

                CREATE TABLE IF NOT EXISTS run_documents (
                    run_id TEXT NOT NULL,
                    ref_no TEXT NOT NULL,
                    PRIMARY KEY(run_id, ref_no),
                    FOREIGN KEY(ref_no) REFERENCES documents(ref_no)
                );

                CREATE INDEX IF NOT EXISTS idx_documents_status
                    ON documents(status, last_run_id, ref_no);
                CREATE INDEX IF NOT EXISTS idx_attempts_status
                    ON stage_attempts(status, lease_until);
                CREATE INDEX IF NOT EXISTS idx_run_documents_ref
                    ON run_documents(ref_no, run_id);
                """
            )
            attempt_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(stage_attempts)"
                ).fetchall()
            }
            if "result_kind" not in attempt_columns:
                connection.execute(
                    "ALTER TABLE stage_attempts ADD COLUMN result_kind TEXT"
                )

    def register_documents(self, paths: Iterable[Path]) -> int:
        now = time.time()
        rows = [
            (path.name.removesuffix("_document.json"), str(path.resolve()), now, now)
            for path in paths
        ]
        with closing(self._connect()) as connection, connection:
            connection.executemany(
                """
                INSERT INTO documents (
                    ref_no, source_path, status, created_at, updated_at
                ) VALUES (?, ?, 'pending', ?, ?)
                ON CONFLICT(ref_no) DO UPDATE SET
                    source_path = excluded.source_path,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def prepare_run(self, run_id: str, ref_nos: Sequence[str]) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executemany(
                "INSERT INTO run_documents (run_id, ref_no) VALUES (?, ?)",
                ((run_id, ref_no) for ref_no in ref_nos),
            )

    def recover_stale(self, *, now: float | None = None) -> int:
        cutoff = time.time() if now is None else now
        message = "worker heartbeat expired; remote request state may be unknown"
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            attempts = connection.execute(
                """
                UPDATE stage_attempts
                SET status = 'interrupted_uncertain', ended_at = ?,
                    error_message = ?
                WHERE status = 'running' AND lease_until < ?
                """,
                (cutoff, message, cutoff),
            ).rowcount
            connection.execute(
                """
                UPDATE documents
                SET status = 'interrupted_uncertain', owner_id = NULL,
                    lease_until = NULL, last_error = ?, updated_at = ?
                WHERE status = 'running' AND lease_until < ?
                """,
                (message, cutoff, cutoff),
            )
            connection.commit()
        return attempts

    def claim_next(
        self,
        *,
        run_id: str,
        worker_id: str,
        allowed_statuses: Sequence[str],
        lease_seconds: float,
    ) -> sqlite3.Row | None:
        placeholders = ", ".join("?" for _ in allowed_statuses)
        now = time.time()
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"""
                SELECT documents.ref_no, documents.source_path, documents.status
                FROM documents
                INNER JOIN run_documents
                    ON run_documents.ref_no = documents.ref_no
                   AND run_documents.run_id = ?
                WHERE documents.status IN ({placeholders})
                  AND (
                      documents.last_run_id IS NULL
                      OR documents.last_run_id <> ?
                  )
                ORDER BY documents.ref_no
                LIMIT 1
                """,
                (run_id, *allowed_statuses, run_id),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            updated = connection.execute(
                """
                UPDATE documents
                SET status = 'running', owner_id = ?, lease_until = ?,
                    last_run_id = ?, last_error = NULL, updated_at = ?
                WHERE ref_no = ? AND status = ?
                """,
                (
                    worker_id,
                    now + lease_seconds,
                    run_id,
                    now,
                    row["ref_no"],
                    row["status"],
                ),
            ).rowcount
            connection.commit()
            return row if updated == 1 else None

    def start_attempt(
        self,
        *,
        run_id: str,
        ref_no: str,
        stage: str,
        worker_id: str,
        lease_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
    ) -> str:
        now = time.time()
        attempt_id = uuid.uuid4().hex
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO stage_attempts (
                    attempt_id, run_id, ref_no, stage, status, worker_id,
                    started_at, heartbeat_at, lease_until,
                    stdout_path, stderr_path
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    run_id,
                    ref_no,
                    stage,
                    worker_id,
                    now,
                    now,
                    now + lease_seconds,
                    str(stdout_path),
                    str(stderr_path),
                ),
            )
            connection.execute(
                """
                UPDATE documents
                SET current_stage = ?, lease_until = ?, updated_at = ?
                WHERE ref_no = ? AND owner_id = ?
                """,
                (stage, now + lease_seconds, now, ref_no, worker_id),
            )
        return attempt_id

    def heartbeat(
        self,
        *,
        attempt_id: str,
        ref_no: str,
        worker_id: str,
        lease_seconds: float,
    ) -> None:
        now = time.time()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE stage_attempts
                SET heartbeat_at = ?, lease_until = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (now, now + lease_seconds, attempt_id),
            )
            connection.execute(
                """
                UPDATE documents
                SET lease_until = ?, updated_at = ?
                WHERE ref_no = ? AND owner_id = ? AND status = 'running'
                """,
                (now + lease_seconds, now, ref_no, worker_id),
            )

    def heartbeat_document(
        self,
        *,
        ref_no: str,
        worker_id: str,
        lease_seconds: float,
        current_stage: str,
    ) -> None:
        now = time.time()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE documents
                SET current_stage = ?, lease_until = ?, updated_at = ?
                WHERE ref_no = ? AND owner_id = ? AND status = 'running'
                """,
                (
                    current_stage,
                    now + lease_seconds,
                    now,
                    ref_no,
                    worker_id,
                ),
            )

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        exit_code: int | None,
        result_kind: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE stage_attempts
                SET status = ?, ended_at = ?, exit_code = ?, result_kind = ?,
                    error_message = ?
                WHERE attempt_id = ?
                """,
                (
                    status,
                    time.time(),
                    exit_code,
                    result_kind,
                    error_message,
                    attempt_id,
                ),
            )

    def finish_document(
        self,
        ref_no: str,
        *,
        worker_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, current_stage = NULL, owner_id = NULL,
                    lease_until = NULL, last_error = ?, updated_at = ?
                WHERE ref_no = ? AND owner_id = ?
                """,
                (status, error_message, time.time(), ref_no, worker_id),
            )

    def summary(self) -> dict[str, int]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM documents GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def run_snapshot(self, run_id: str) -> dict[str, list[dict[str, Any]]]:
        with closing(self._connect()) as connection, connection:
            documents = connection.execute(
                """
                SELECT documents.ref_no, documents.status,
                       documents.current_stage, documents.last_error,
                       documents.updated_at
                FROM documents
                INNER JOIN run_documents
                    ON run_documents.ref_no = documents.ref_no
                   AND run_documents.run_id = ?
                ORDER BY documents.ref_no
                """,
                (run_id,),
            ).fetchall()
            attempts = connection.execute(
                """
                SELECT attempt_id, ref_no, stage, status, worker_id,
                       started_at, heartbeat_at, ended_at, exit_code,
                       result_kind, stdout_path, stderr_path, error_message
                FROM stage_attempts
                WHERE run_id = ?
                ORDER BY started_at, ref_no, stage
                """,
                (run_id,),
            ).fetchall()
        return {
            "documents": [dict(row) for row in documents],
            "attempts": [dict(row) for row in attempts],
        }


@dataclass(frozen=True)
class RunnerSettings:
    config_path: Path
    input_dir: Path
    output_dir: Path
    logs_dir: Path
    force: bool
    heartbeat_seconds: float
    lease_seconds: float
    preview: bool = False


def build_stage_command(
    spec: StageSpec,
    *,
    ref_no: str,
    settings: RunnerSettings,
    extra_args: Sequence[str] = (),
) -> list[str]:
    script_path = Path(spec.script_name)
    if not script_path.is_absolute():
        script_path = STAGES_ROOT / script_path
    command = [
        sys.executable,
        str(script_path),
        "--ref-no",
        ref_no,
        "--config",
        str(settings.config_path),
    ]
    if spec.stage_id == "stage0_load_document":
        command.extend(
            ["--input-dir", str(settings.input_dir), "--output-dir", str(settings.output_dir)]
        )
    else:
        command.extend(
            ["--input-root", str(settings.output_dir), "--output-root", str(settings.output_dir)]
        )
    if settings.force and spec.stage_id != "stage6_validate_merge":
        command.append("--force")
    if spec.stage_id == STAGE4R_PREVIEW_STAGE.stage_id:
        command.append("--apply")
    if settings.preview and (
        spec.stage_id in PREVIEW_RECOVERABLE_FAILURES
        or spec.stage_id == "stage6_validate_merge"
    ):
        command.append("--preview-relaxed")
    for argument in extra_args:
        if argument not in command:
            command.append(argument)
    return command


def _outputs_exist(spec: StageSpec, output_dir: Path, ref_no: str) -> bool:
    base = output_dir / ref_no
    return all((base / name).is_file() for name in spec.output_names)


def run_stage_process(
    *,
    store: BatchStateStore,
    run_id: str,
    worker_id: str,
    ref_no: str,
    spec: StageSpec,
    settings: RunnerSettings,
    stop_event: threading.Event,
    extra_args: Sequence[str] = (),
) -> tuple[bool, str | None]:
    log_dir = settings.logs_dir / ref_no
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    stdout_path = log_dir / f"{stamp}_{spec.stage_id}_{suffix}.stdout.log"
    stderr_path = log_dir / f"{stamp}_{spec.stage_id}_{suffix}.stderr.log"
    attempt_id = store.start_attempt(
        run_id=run_id,
        ref_no=ref_no,
        stage=spec.stage_id,
        worker_id=worker_id,
        lease_seconds=settings.lease_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    command = build_stage_command(
        spec,
        ref_no=ref_no,
        settings=settings,
        extra_args=extra_args,
    )
    is_failure_replay = "--replay-failure" in extra_args
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=EXTRACTION_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
            while True:
                if stop_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    message = "batch supervisor interrupted; remote request state may be unknown"
                    store.finish_attempt(
                        attempt_id,
                        status="interrupted_uncertain",
                        exit_code=process.returncode,
                        result_kind="unknown",
                        error_message=message,
                    )
                    return False, message
                try:
                    exit_code = process.wait(timeout=settings.heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    store.heartbeat(
                        attempt_id=attempt_id,
                        ref_no=ref_no,
                        worker_id=worker_id,
                        lease_seconds=settings.lease_seconds,
                    )
    except OSError as exc:
        message = f"failed to start stage process: {exc}"
        store.finish_attempt(
            attempt_id,
            status="failed",
            exit_code=None,
            result_kind="not_started",
            error_message=message,
        )
        return False, message

    if exit_code != 0:
        message = f"{spec.stage_id} exited with code {exit_code}; see {stderr_path}"
        store.finish_attempt(
            attempt_id,
            status="failed",
            exit_code=exit_code,
            result_kind="replay_failed" if is_failure_replay else "executed",
            error_message=message,
        )
        return False, message
    if not _outputs_exist(spec, settings.output_dir, ref_no):
        message = f"{spec.stage_id} exited successfully but required output is missing"
        store.finish_attempt(
            attempt_id,
            status="failed",
            exit_code=exit_code,
            result_kind="replay_failed" if is_failure_replay else "executed",
            error_message=message,
        )
        return False, message

    try:
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        stdout_text = ""
    if is_failure_replay:
        result_kind = "replayed"
    else:
        result_kind = "cached" if "[cached]" in stdout_text else "executed"
    store.finish_attempt(
        attempt_id,
        status="succeeded",
        exit_code=exit_code,
        result_kind=result_kind,
    )
    return True, None


def run_document(
    *,
    store: BatchStateStore,
    run_id: str,
    worker_id: str,
    ref_no: str,
    settings: RunnerSettings,
    llm_semaphore: threading.Semaphore,
    stop_event: threading.Event,
    stage_specs: Sequence[StageSpec] | None = None,
    required_existing_outputs: Sequence[str] = (),
) -> bool:
    missing_outputs = [
        name
        for name in required_existing_outputs
        if not (settings.output_dir / ref_no / name).is_file()
    ]
    if missing_outputs:
        message = "现有产物不完整：" + ", ".join(missing_outputs)
        store.finish_document(
            ref_no,
            worker_id=worker_id,
            status="failed",
            error_message=message,
        )
        print(f"[failed] {ref_no}: {message}", file=sys.stderr)
        return False

    selected_stages = STAGES if stage_specs is None else stage_specs
    preview_run = any(
        item.stage_id == PREVIEW_STAGE.stage_id for item in selected_stages
    )
    for spec in selected_stages:
        if stop_event.is_set():
            message = "batch supervisor interrupted before next stage"
            store.finish_document(
                ref_no,
                worker_id=worker_id,
                status="interrupted_uncertain",
                error_message=message,
            )
            return False
        failure_name = PREVIEW_RECOVERABLE_FAILURES.get(spec.stage_id)
        failure_path = (
            settings.output_dir / ref_no / failure_name
            if failure_name is not None
            else None
        )
        if (
            preview_run
            and spec.uses_llm
            and not settings.force
            and failure_path is not None
            and failure_path.is_file()
            and not _outputs_exist(spec, settings.output_dir, ref_no)
        ):
            replay_args = [
                "--replay-failure",
                "--preview-relaxed",
                "--force",
            ]
            recovered, recovery_error = run_stage_process(
                store=store,
                run_id=run_id,
                worker_id=worker_id,
                ref_no=ref_no,
                spec=spec,
                settings=settings,
                stop_event=stop_event,
                extra_args=tuple(replay_args),
            )
            if recovered:
                print(f"[replayed] {ref_no}: {spec.stage_id} 已在模型调用前离线恢复")
                continue
            print(
                f"[replay-miss] {ref_no}: {spec.stage_id} 离线恢复失败："
                f"{recovery_error}；将执行正常 Stage",
                file=sys.stderr,
            )

        if spec.uses_llm:
            acquired = False
            store.heartbeat_document(
                ref_no=ref_no,
                worker_id=worker_id,
                lease_seconds=settings.lease_seconds,
                current_stage="waiting_for_llm_slot",
            )
            while not stop_event.is_set():
                acquired = llm_semaphore.acquire(
                    timeout=settings.heartbeat_seconds
                )
                if acquired:
                    break
                store.heartbeat_document(
                    ref_no=ref_no,
                    worker_id=worker_id,
                    lease_seconds=settings.lease_seconds,
                    current_stage="waiting_for_llm_slot",
                )
            if not acquired:
                message = "batch supervisor interrupted while waiting for LLM slot"
                store.finish_document(
                    ref_no,
                    worker_id=worker_id,
                    status="interrupted_uncertain",
                    error_message=message,
                )
                return False
            try:
                succeeded, error = run_stage_process(
                    store=store,
                    run_id=run_id,
                    worker_id=worker_id,
                    ref_no=ref_no,
                    spec=spec,
                    settings=settings,
                    stop_event=stop_event,
                )
            finally:
                llm_semaphore.release()
        else:
            succeeded, error = run_stage_process(
                store=store,
                run_id=run_id,
                worker_id=worker_id,
                ref_no=ref_no,
                spec=spec,
                settings=settings,
                stop_event=stop_event,
            )
        if not succeeded:
            status = "interrupted_uncertain" if stop_event.is_set() else "failed"
            if error and "remote request state may be unknown" in error:
                status = "interrupted_uncertain"
            failure_name = PREVIEW_RECOVERABLE_FAILURES.get(spec.stage_id)
            failure_path = (
                settings.output_dir / ref_no / failure_name
                if failure_name is not None
                else None
            )
            if (
                preview_run
                and status == "failed"
                and failure_path is not None
                and failure_path.is_file()
            ):
                recovery_args = [
                    "--replay-failure",
                    "--preview-relaxed",
                    "--force",
                ]
                recovered, recovery_error = run_stage_process(
                    store=store,
                    run_id=run_id,
                    worker_id=worker_id,
                    ref_no=ref_no,
                    spec=spec,
                    settings=settings,
                    stop_event=stop_event,
                    extra_args=tuple(recovery_args),
                )
                if recovered:
                    print(f"[recovered] {ref_no}: {spec.stage_id} 已离线恢复")
                    continue
                error = f"{error}; preview recovery failed: {recovery_error}"
            if (
                preview_run
                and status == "failed"
                and spec.stage_id != PREVIEW_STAGE.stage_id
            ):
                published, publish_error = run_stage_process(
                    store=store,
                    run_id=run_id,
                    worker_id=worker_id,
                    ref_no=ref_no,
                    spec=PREVIEW_STAGE,
                    settings=settings,
                    stop_event=stop_event,
                )
                if published:
                    store.finish_document(
                        ref_no,
                        worker_id=worker_id,
                        status="candidate_partial",
                        error_message=error,
                    )
                    print(
                        f"[partial] {ref_no}: {spec.stage_id} 失败，已发布候选结果"
                    )
                    return False
                error = f"{error}; candidate publish failed: {publish_error}"
            store.finish_document(
                ref_no,
                worker_id=worker_id,
                status=status,
                error_message=error,
            )
            print(f"[failed] {ref_no} {spec.stage_id}: {error}", file=sys.stderr)
            return False

    final_status = "candidate_complete" if preview_run else "succeeded"
    store.finish_document(ref_no, worker_id=worker_id, status=final_status)
    print(f"[done] {ref_no}")
    return True


def _worker_loop(
    *,
    index: int,
    store: BatchStateStore,
    run_id: str,
    allowed_statuses: Sequence[str],
    settings: RunnerSettings,
    llm_semaphore: threading.Semaphore,
    stop_event: threading.Event,
    stage_specs: Sequence[StageSpec] | None = None,
    required_existing_outputs: Sequence[str] = (),
) -> int:
    worker_id = f"{run_id}-worker-{index}"
    completed = 0
    while not stop_event.is_set():
        document = store.claim_next(
            run_id=run_id,
            worker_id=worker_id,
            allowed_statuses=allowed_statuses,
            lease_seconds=settings.lease_seconds,
        )
        if document is None:
            break
        if run_document(
            store=store,
            run_id=run_id,
            worker_id=worker_id,
            ref_no=str(document["ref_no"]),
            settings=settings,
            llm_semaphore=llm_semaphore,
            stop_event=stop_event,
            stage_specs=stage_specs,
            required_existing_outputs=required_existing_outputs,
        ):
            completed += 1
    return completed


def _configured_paths(config: dict) -> tuple[Path, Path]:
    paths = config.get("paths") or {}
    input_dir = Path(
        paths.get("input_dir")
        or EXTRACTION_ROOT / "processed_data" / "documents"
    )
    output_dir = Path(paths.get("output_dir") or EXTRACTION_ROOT / "output")
    return input_dir, output_dir


def _normalize_ref_no(value: str) -> str:
    ref_no = value.strip()
    if ref_no.endswith("_document.json"):
        ref_no = ref_no.removesuffix("_document.json")
    suffix = ref_no.removeprefix("reference_no_")
    if not ref_no.startswith("reference_no_") or not suffix.isdigit():
        raise BatchRunnerError(f"无效文献编号：{value!r}")
    return ref_no


def select_document_paths(
    input_dir: Path,
    *,
    ref_nos: Sequence[str] | None = None,
    ref_list_path: Path | None = None,
    max_documents: int | None = None,
) -> list[Path]:
    selected_ref_nos: list[str] = []
    if ref_nos:
        selected_ref_nos.extend(_normalize_ref_no(value) for value in ref_nos)
    elif ref_list_path is not None:
        path = ref_list_path.expanduser().resolve()
        if not path.is_file():
            raise BatchRunnerError(f"文献清单不存在：{path}")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                selected_ref_nos.append(_normalize_ref_no(value))

    if selected_ref_nos:
        unique_ref_nos = list(dict.fromkeys(selected_ref_nos))
        paths = [input_dir / f"{ref_no}_document.json" for ref_no in unique_ref_nos]
        missing = [path.name.removesuffix("_document.json") for path in paths if not path.is_file()]
        if missing:
            raise BatchRunnerError("未找到指定文献：" + ", ".join(missing))
        return paths

    paths = sorted(input_dir.glob("reference_no_*_document.json"))
    if max_documents is not None:
        if max_documents < 1:
            raise BatchRunnerError("max-documents 必须大于 0")
        paths = paths[:max_documents]
    return paths


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _decimal_value(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _increment(mapping: dict[str, int], key: str) -> None:
    mapping[key] = mapping.get(key, 0) + 1


def _stage4_coordinate_column_warnings(
    ref_nos: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    code = "table_property_column_represented_as_coordinate"
    columns: list[dict[str, Any]] = []
    for ref_no in ref_nos:
        payload = _read_json_object(
            output_dir / ref_no / LLM_STAGE_OUTPUTS["stage4_property"]
        )
        if payload is None:
            continue
        warnings = payload.get("warnings")
        if not isinstance(warnings, list):
            continue
        for warning in warnings:
            if not isinstance(warning, dict) or warning.get("code") != code:
                continue
            warning_columns = warning.get("columns")
            if not isinstance(warning_columns, list):
                continue
            for column in warning_columns:
                if isinstance(column, dict):
                    columns.append({"ref_no": ref_no, **column})
    return {
        code: {
            "column_count": len(columns),
            "columns": columns,
        }
    }


def _attempt_billing(
    attempt: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    stage = str(attempt["stage"])
    result_kind = attempt.get("result_kind")
    if stage not in LLM_STAGE_OUTPUTS or result_kind in {
        "cached",
        "not_started",
        "replayed",
        "replay_failed",
    }:
        return None, None
    if result_kind != "executed":
        return None, "remote request state may be unknown"

    ref_no = str(attempt["ref_no"])
    if attempt["status"] == "succeeded":
        artifact_path = output_dir / ref_no / LLM_STAGE_OUTPUTS[stage]
        container_key = "provenance"
    elif attempt["status"] == "failed" and stage in LLM_STAGE_FAILURES:
        artifact_path = output_dir / ref_no / LLM_STAGE_FAILURES[stage]
        container_key = None
    else:
        return None, "no auditable billing artifact for executed LLM attempt"

    try:
        if artifact_path.stat().st_mtime + 2 < float(attempt["started_at"]):
            return None, "billing artifact predates this attempt"
    except OSError:
        return None, "billing artifact is missing"
    payload = _read_json_object(artifact_path)
    if payload is None:
        return None, "billing artifact is unreadable"
    container = payload.get(container_key) if container_key else payload
    if not isinstance(container, dict):
        return None, "billing container is missing"
    cost = container.get("cost")
    usage = container.get("usage")
    if isinstance(cost, dict) and cost.get("status") == "not_applicable":
        zero_cost = all(
            _decimal_value(cost.get(field)) == Decimal("0")
            for field in ("input_cost", "output_cost", "total_cost")
        )
        usage_payload = usage if isinstance(usage, dict) else {}
        zero_usage = all(
            int(usage_payload.get(field) or 0) == 0
            for field in (
                "input_tokens",
                "output_tokens",
                "billable_input_tokens",
                "total_tokens",
            )
        )
        explicit_calls = container.get("call_count")
        zero_calls = explicit_calls in (None, 0)
        no_response = container.get("raw_response") is None
        if zero_cost and zero_usage and zero_calls and no_response:
            return None, None
        return None, "not_applicable billing artifact is inconsistent"
    if not isinstance(cost, dict) or cost.get("status") != "calculated":
        return None, "calculated cost is unavailable"
    total_cost = _decimal_value(cost.get("total_cost"))
    input_cost = _decimal_value(cost.get("input_cost"))
    output_cost = _decimal_value(cost.get("output_cost"))
    if total_cost is None or input_cost is None or output_cost is None:
        return None, "cost fields are invalid"
    usage = usage if isinstance(usage, dict) else {}
    call_count = container.get("call_count")
    if call_count is None and stage == "stage1_material_mention":
        call_count = container.get("chunk_count")
    if call_count is None:
        call_count = 1
    return {
        "currency": cost.get("currency"),
        "input_per_million": cost.get("input_per_million"),
        "output_per_million": cost.get("output_per_million"),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "call_count": int(call_count),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "billable_input_tokens": int(usage.get("billable_input_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }, None


def build_batch_acceptance(
    status_counts: dict[str, int],
    *,
    preview: bool,
) -> dict[str, Any]:
    expected_status = "candidate_complete" if preview else "succeeded"
    total = sum(int(value) for value in status_counts.values())
    accepted_count = int(status_counts.get(expected_status, 0))
    blocking_statuses = {
        status: int(count)
        for status, count in sorted(status_counts.items())
        if status != expected_status and int(count) > 0
    }
    accepted = total > 0 and accepted_count == total and not blocking_statuses
    return {
        "accepted": accepted,
        "expected_status": expected_status,
        "total": total,
        "accepted_count": accepted_count,
        "blocking_statuses": blocking_statuses,
    }


def build_run_summary(
    *,
    store: BatchStateStore,
    run_id: str,
    output_dir: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    snapshot = store.run_snapshot(run_id)
    document_statuses: dict[str, int] = {}
    attempt_statuses: dict[str, int] = {}
    stage_attempts: dict[str, int] = {}
    attempt_items: list[dict[str, Any]] = []
    for document in snapshot["documents"]:
        _increment(document_statuses, str(document["status"]))
    for attempt in snapshot["attempts"]:
        _increment(attempt_statuses, str(attempt["status"]))
        _increment(stage_attempts, str(attempt["stage"]))
        item = dict(attempt)
        started_at = float(item["started_at"])
        ended_at = item.get("ended_at")
        item["duration_seconds"] = (
            round(float(ended_at) - started_at, 3)
            if ended_at is not None
            else None
        )
        attempt_items.append(item)

    input_cost = Decimal("0")
    output_cost = Decimal("0")
    total_cost = Decimal("0")
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "billable_input_tokens": 0,
        "total_tokens": 0,
    }
    call_count = 0
    charged_attempts = 0
    cached_attempts = 0
    currency: str | None = None
    pricing: list[dict[str, Any]] = []
    unknown_attempts: list[dict[str, str]] = []
    for attempt in snapshot["attempts"]:
        if (
            attempt["stage"] in LLM_STAGE_OUTPUTS
            and attempt.get("result_kind") == "cached"
        ):
            cached_attempts += 1
        billing, error = _attempt_billing(attempt, output_dir)
        if error:
            unknown_attempts.append(
                {
                    "attempt_id": str(attempt["attempt_id"]),
                    "ref_no": str(attempt["ref_no"]),
                    "stage": str(attempt["stage"]),
                    "reason": error,
                }
            )
            continue
        if billing is None:
            continue
        attempt_currency = billing["currency"]
        if currency is None:
            currency = str(attempt_currency) if attempt_currency else None
        elif attempt_currency != currency:
            unknown_attempts.append(
                {
                    "attempt_id": str(attempt["attempt_id"]),
                    "ref_no": str(attempt["ref_no"]),
                    "stage": str(attempt["stage"]),
                    "reason": "currency does not match other attempts",
                }
            )
            continue
        charged_attempts += 1
        call_count += billing["call_count"]
        input_cost += billing["input_cost"]
        output_cost += billing["output_cost"]
        total_cost += billing["total_cost"]
        for key in usage_totals:
            usage_totals[key] += billing[key]
        price_item = {
            "currency": attempt_currency,
            "input_per_million": billing["input_per_million"],
            "output_per_million": billing["output_per_million"],
        }
        if price_item not in pricing:
            pricing.append(price_item)

    billing_status = "calculated"
    if unknown_attempts:
        billing_status = "partial"
    elif charged_attempts == 0:
        billing_status = "no_new_charge"
    if settings.get("validate_existing"):
        warning_ref_nos = [
            str(document["ref_no"])
            for document in snapshot["documents"]
        ]
    else:
        warning_ref_nos = list(dict.fromkeys(
            str(attempt["ref_no"])
            for attempt in snapshot["attempts"]
            if attempt["stage"] == "stage4_property"
            and attempt["status"] == "succeeded"
        ))
    artifact_warnings = _stage4_coordinate_column_warnings(
        warning_ref_nos,
        output_dir,
    )
    acceptance = build_batch_acceptance(
        document_statuses,
        preview=bool(settings.get("preview")),
    )
    return {
        "schema_version": "batch_summary.v1",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "documents": {
            "total": len(snapshot["documents"]),
            "status_counts": document_statuses,
            "items": snapshot["documents"],
        },
        "attempts": {
            "total": len(snapshot["attempts"]),
            "status_counts": attempt_statuses,
            "stage_counts": stage_attempts,
            "items": attempt_items,
        },
        "artifact_warnings": artifact_warnings,
        "acceptance": acceptance,
        "billing": {
            "scope": "new_executed_llm_attempts_only",
            "status": billing_status,
            "currency": currency,
            "charged_attempt_count": charged_attempts,
            "cached_attempt_count": cached_attempts,
            "call_count": call_count,
            "usage": usage_totals,
            "cost": {
                "input_cost": format(input_cost, "f"),
                "output_cost": format(output_cost, "f"),
                "total_cost": format(total_cost, "f"),
            },
            "pricing": pricing,
            "unknown_attempts": unknown_attempts,
        },
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="并行执行 Stage 0-6，支持中断识别和续跑")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state-db", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--llm-workers", type=int)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--lease-seconds", type=float, default=90.0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ref-no", action="append")
    selection.add_argument("--ref-list", type=Path)
    selection.add_argument("--max-documents", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-interrupted", action="store_true")
    parser.add_argument("--recheck-completed", action="store_true")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="仅使用 Stage 0-5 现有产物重新执行 Stage 6，不调用模型",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "预览模式：Stage 4 后插入 4R 表格恢复，各 Stage 带 --preview-relaxed，"
            "Stage 6 以降级校验产出 final.json/report.html，"
            "并始终发布 candidate.json 和 report_candidate.html"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _print_summary(summary: dict[str, int]) -> None:
    if not summary:
        print("批处理状态为空")
        return
    print("批处理状态：" + ", ".join(f"{key}={summary[key]}" for key in sorted(summary)))


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_pipeline_config(config_path)
    configured_input, configured_output = _configured_paths(config)
    input_dir = (args.input_dir or configured_input).expanduser().resolve()
    output_dir = (args.output_dir or configured_output).expanduser().resolve()
    state_db = (
        args.state_db or output_dir / "_batch" / "batch_state.sqlite3"
    ).expanduser().resolve()
    logs_dir = state_db.parent / "logs"
    concurrency = config.get("concurrency") or {}
    workers = int(args.workers or concurrency.get("max_doc_workers") or 1)
    llm_workers = int(args.llm_workers or concurrency.get("max_llm_workers") or 1)

    if workers < 1 or llm_workers < 1:
        raise BatchRunnerError("workers 和 llm-workers 必须大于 0")
    if args.validate_existing and args.force:
        raise BatchRunnerError("--validate-existing 不能与 --force 同时使用")
    if args.validate_existing and args.preview:
        raise BatchRunnerError("--validate-existing 不能与 --preview 同时使用")
    if args.validate_existing and not (args.ref_no or args.ref_list):
        raise BatchRunnerError(
            "--validate-existing 必须通过 --ref-no 或 --ref-list 明确选择文献"
        )
    if args.heartbeat_seconds <= 0:
        raise BatchRunnerError("heartbeat-seconds 必须大于 0")
    if args.lease_seconds <= args.heartbeat_seconds * 2:
        raise BatchRunnerError("lease-seconds 必须大于 heartbeat-seconds 的两倍")

    if args.status:
        store = BatchStateStore(state_db)
        store.recover_stale()
        _print_summary(store.summary())
        return 0
    if not input_dir.is_dir():
        raise BatchRunnerError(f"输入目录不存在：{input_dir}")

    document_paths = select_document_paths(
        input_dir,
        ref_nos=args.ref_no,
        ref_list_path=args.ref_list,
        max_documents=args.max_documents,
    )
    if not document_paths:
        raise BatchRunnerError(f"未找到 document JSON：{input_dir}")
    if args.dry_run:
        print(f"将登记 {len(document_paths)} 篇文献，doc workers={workers}，LLM workers={llm_workers}")
        print("运行模式：" + ("Preview Candidate" if args.preview else "Strict"))
        if args.ref_no or args.ref_list:
            print("文献：" + ", ".join(path.name.removesuffix("_document.json") for path in document_paths))
        return 0

    store = BatchStateStore(state_db)
    run_id = uuid.uuid4().hex
    registered = store.register_documents(document_paths)
    ref_nos = [path.name.removesuffix("_document.json") for path in document_paths]
    store.prepare_run(run_id, ref_nos)
    recovered = store.recover_stale()
    print(f"已登记 {registered} 篇文献；恢复过期 attempt {recovered} 个")

    if args.validate_existing:
        allowed_statuses = [
            "pending",
            "failed",
            "interrupted_uncertain",
            "succeeded",
        ]
    else:
        allowed_statuses = ["pending"]
        if args.retry_failed:
            allowed_statuses.extend(("failed", "candidate_partial"))
        if args.retry_interrupted:
            allowed_statuses.append("interrupted_uncertain")
        if args.recheck_completed:
            allowed_statuses.extend(
                ("succeeded", "candidate_complete", "candidate_partial")
            )
        allowed_statuses = list(dict.fromkeys(allowed_statuses))

    if args.validate_existing:
        stage_specs = (STAGES[-1],)
    elif args.preview:
        stage_specs = PREVIEW_STAGES
    else:
        stage_specs = STAGES
    required_existing_outputs = (
        VALIDATE_EXISTING_INPUTS if args.validate_existing else ()
    )

    settings = RunnerSettings(
        config_path=config_path,
        input_dir=input_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        force=args.force,
        heartbeat_seconds=args.heartbeat_seconds,
        lease_seconds=args.lease_seconds,
        preview=bool(args.preview),
    )
    llm_semaphore = threading.Semaphore(llm_workers)
    stop_event = threading.Event()
    completed = 0
    summary_path = (
        args.summary_out or state_db.parent / "summaries" / f"{run_id}.json"
    ).expanduser().resolve()
    summary_settings = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "state_db": str(state_db),
        "workers": workers,
        "llm_workers": llm_workers,
        "force": bool(args.force),
        "retry_failed": bool(args.retry_failed),
        "retry_interrupted": bool(args.retry_interrupted),
        "recheck_completed": bool(args.recheck_completed),
        "validate_existing": bool(args.validate_existing),
        "preview": bool(args.preview),
        "selected_ref_nos": ref_nos,
    }

    def persist_summary() -> dict[str, Any]:
        report = build_run_summary(
            store=store,
            run_id=run_id,
            output_dir=output_dir,
            settings=summary_settings,
        )
        write_json_atomic(summary_path, report)
        print(f"批次汇总：{summary_path}")
        return report

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = [
        executor.submit(
            _worker_loop,
            index=index + 1,
            store=store,
            run_id=run_id,
            allowed_statuses=allowed_statuses,
            settings=settings,
            llm_semaphore=llm_semaphore,
            stop_event=stop_event,
            stage_specs=stage_specs,
            required_existing_outputs=required_existing_outputs,
        )
        for index in range(workers)
    ]
    try:
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
    except KeyboardInterrupt:
        stop_event.set()
        print("收到中断，正在终止当前 Stage；相关任务将标记为 interrupted_uncertain", file=sys.stderr)
        executor.shutdown(wait=True, cancel_futures=True)
        persist_summary()
        return 130
    finally:
        if not stop_event.is_set():
            executor.shutdown(wait=True)

    report = persist_summary()
    summary = report["documents"]["status_counts"]
    billing = report["billing"]
    print(f"本轮完成 {completed} 篇")
    _print_summary(summary)
    print(
        "本轮新增费用："
        f"status={billing['status']}，"
        f"calls={billing['call_count']}，"
        f"total={billing['cost']['total_cost']} "
        f"{billing['currency'] or ''}".rstrip()
    )
    acceptance = report["acceptance"]
    print(
        "批次验收："
        f"accepted={str(acceptance['accepted']).lower()}，"
        f"expected_status={acceptance['expected_status']}，"
        f"accepted_count={acceptance['accepted_count']}/"
        f"{acceptance['total']}"
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
