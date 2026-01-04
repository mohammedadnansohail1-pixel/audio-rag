"""Tests for job dataclasses and enums."""

import json
from datetime import datetime, timezone

import pytest

from audio_rag.queue.job import (
    Priority,
    JobStatus,
    JobStage,
    STAGE_ORDER,
    IngestJob,
    JobResult,
    get_next_stage,
    get_stage_index,
)


class TestPriority:
    def test_priority_values(self):
        assert Priority.LOW.value == 1
        assert Priority.NORMAL.value == 2
        assert Priority.HIGH.value == 3
        assert Priority.CRITICAL.value == 4

    def test_priority_ordering(self):
        assert Priority.LOW < Priority.NORMAL < Priority.HIGH < Priority.CRITICAL


class TestJobStatus:
    def test_status_values(self):
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"


class TestJobStage:
    def test_stage_order_exists(self):
        assert len(STAGE_ORDER) > 0
        assert "queued" in STAGE_ORDER
        assert "complete" in STAGE_ORDER

    def test_get_stage_index(self):
        assert get_stage_index("queued") == 0
        assert get_stage_index("complete") == len(STAGE_ORDER) - 1

    def test_get_stage_index_invalid_raises(self):
        with pytest.raises(ValueError):
            get_stage_index("nonexistent")

    def test_get_next_stage(self):
        first_stage = STAGE_ORDER[0]
        second_stage = STAGE_ORDER[1]
        assert get_next_stage(first_stage) == second_stage
        
        last_stage = STAGE_ORDER[-1]
        assert get_next_stage(last_stage) is None


class TestIngestJob:
    def test_create_job_minimal(self):
        job = IngestJob(
            tenant_id="test_tenant",
            audio_path="/path/to/audio.mp3",
        )
        assert job.tenant_id == "test_tenant"
        assert job.audio_path == "/path/to/audio.mp3"
        assert job.priority == Priority.NORMAL
        assert job.job_id is not None
        assert job.created_at is not None

    def test_create_job_full(self):
        job = IngestJob(
            tenant_id="test_tenant",
            audio_path="/path/to/audio.mp3",
            priority=Priority.HIGH,
            metadata={"key": "value"},
            callback_url="https://example.com/callback",
        )
        assert job.priority == Priority.HIGH
        assert job.metadata == {"key": "value"}
        assert job.callback_url == "https://example.com/callback"

    def test_job_serialization(self):
        job = IngestJob(
            tenant_id="test_tenant",
            audio_path="/path/to/audio.mp3",
            metadata={"key": "value"},
        )
        
        job_dict = job.to_dict()
        assert job_dict["tenant_id"] == "test_tenant"
        assert job_dict["audio_path"] == "/path/to/audio.mp3"
        assert job_dict["metadata"] == {"key": "value"}
        
        job_json = job.to_json()
        parsed = json.loads(job_json)
        assert parsed["tenant_id"] == "test_tenant"

    def test_job_deserialization(self):
        original = IngestJob(
            tenant_id="test_tenant",
            audio_path="/path/to/audio.mp3",
            priority=Priority.HIGH,
        )
        
        restored = IngestJob.from_dict(original.to_dict())
        assert restored.tenant_id == original.tenant_id
        assert restored.audio_path == original.audio_path
        assert restored.priority == original.priority
        assert restored.job_id == original.job_id

    def test_job_from_json(self):
        original = IngestJob(
            tenant_id="test_tenant",
            audio_path="/path/to/audio.mp3",
        )
        
        restored = IngestJob.from_json(original.to_json())
        assert restored.tenant_id == original.tenant_id
        assert restored.job_id == original.job_id


class TestJobResult:
    def test_create_result_success(self):
        result = JobResult(
            job_id="test-job-id",
            status=JobStatus.COMPLETED,
            stage="complete",
            chunks_created=10,
        )
        assert result.is_success
        assert result.is_terminal

    def test_create_result_failed(self):
        result = JobResult(
            job_id="test-job-id",
            status=JobStatus.FAILED,
            stage="transcribed",
            error_message="Something went wrong",
            error_type="ProcessingError",
        )
        assert not result.is_success
        assert result.is_terminal
        assert result.error_message == "Something went wrong"

    def test_result_pending_not_terminal(self):
        result = JobResult(
            job_id="test-job-id",
            status=JobStatus.PENDING,
            stage="queued",
        )
        assert not result.is_terminal

    def test_result_running_not_terminal(self):
        result = JobResult(
            job_id="test-job-id",
            status=JobStatus.RUNNING,
            stage="transcribed",
        )
        assert not result.is_terminal
