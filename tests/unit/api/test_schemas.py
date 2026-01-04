"""Tests for API request/response schemas."""

import pytest
from pydantic import ValidationError

from audio_rag.api.schemas import (
    ErrorResponse,
    IngestRequest,
    IngestConfig,
    IngestResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
)


class TestErrorResponse:
    def test_create_error(self):
        error = ErrorResponse(
            error="Something went wrong",
            code="INTERNAL_ERROR",
            status=500,
        )
        assert error.error == "Something went wrong"
        assert error.code == "INTERNAL_ERROR"
        assert error.status == 500

    def test_error_with_details(self):
        error = ErrorResponse(
            error="Validation failed",
            code="VALIDATION_ERROR",
            status=422,
            request_id="req-123",
            details={"field": "name", "issue": "required"},
            retry_after=60,
        )
        assert error.details["field"] == "name"
        assert error.retry_after == 60


class TestIngestConfig:
    def test_default_values(self):
        config = IngestConfig()
        assert config.enable_diarization is True
        assert config.chunk_duration_seconds == 30.0

    def test_custom_values(self):
        config = IngestConfig(
            language="es",
            enable_diarization=False,
            max_speakers=5,
        )
        assert config.language == "es"
        assert config.enable_diarization is False
        assert config.max_speakers == 5

    def test_invalid_max_speakers(self):
        with pytest.raises(ValidationError):
            IngestConfig(max_speakers=0)
        with pytest.raises(ValidationError):
            IngestConfig(max_speakers=25)

    def test_invalid_chunk_duration(self):
        with pytest.raises(ValidationError):
            IngestConfig(chunk_duration_seconds=1)  # Too short


class TestIngestRequest:
    def test_minimal_request(self):
        request = IngestRequest()
        assert request.priority == "normal"
        assert request.collection is None

    def test_full_request(self):
        request = IngestRequest(
            collection="audio_rag_test_collection",
            priority="high",
            metadata={"lecture": "Week 1"},
            callback_url="https://example.com/webhook",
        )
        assert request.collection == "audio_rag_test_collection"
        assert request.priority == "high"

    def test_invalid_priority(self):
        with pytest.raises(ValidationError):
            IngestRequest(priority="urgent")

    def test_invalid_collection_pattern(self):
        with pytest.raises(ValidationError):
            IngestRequest(collection="Invalid-Collection!")


class TestIngestResponse:
    def test_create_response(self):
        response = IngestResponse(
            job_id="job-123",
            status="pending",
            collection="test_collection",
            filename="lecture.mp3",
            file_size_bytes=1024000,
            priority="normal",
        )
        assert response.job_id == "job-123"
        assert response.status == "pending"

    def test_with_estimated_wait(self):
        response = IngestResponse(
            job_id="job-123",
            status="pending",
            collection="test_collection",
            filename="lecture.mp3",
            file_size_bytes=1024000,
            priority="normal",
            estimated_wait_seconds=120,
        )
        assert response.estimated_wait_seconds == 120


class TestJobStatusResponse:
    def test_pending_job(self):
        from datetime import datetime, timezone
        
        response = JobStatusResponse(
            job_id="job-123",
            status="pending",
            stage="queued",
            created_at=datetime.now(timezone.utc),
        )
        assert response.status == "pending"
        assert response.progress is None

    def test_completed_job(self):
        from datetime import datetime, timezone
        
        now = datetime.now(timezone.utc)
        response = JobStatusResponse(
            job_id="job-123",
            status="completed",
            stage="complete",
            progress=1.0,
            created_at=now,
            started_at=now,
            completed_at=now,
            duration_seconds=120.5,
            result={"chunks_created": 15},
        )
        assert response.status == "completed"
        assert response.progress == 1.0
        assert response.result["chunks_created"] == 15

    def test_failed_job(self):
        from datetime import datetime, timezone
        
        response = JobStatusResponse(
            job_id="job-123",
            status="failed",
            stage="transcribed",
            created_at=datetime.now(timezone.utc),
            error="Out of memory",
        )
        assert response.status == "failed"
        assert response.error == "Out of memory"


class TestQueryRequest:
    def test_minimal_query(self):
        request = QueryRequest(query="What is machine learning?")
        assert request.query == "What is machine learning?"
        assert request.limit == 10
        assert request.offset == 0

    def test_full_query(self):
        request = QueryRequest(
            query="What is machine learning?",
            collection="test_collection",
            limit=5,
            offset=10,
            score_threshold=0.7,
            filters={"lecture": "Week 1"},
            include_context=False,
        )
        assert request.limit == 5
        assert request.score_threshold == 0.7
        assert request.include_context is False

    def test_empty_query_invalid(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", limit=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="test", limit=150)


class TestSearchResult:
    def test_create_result(self):
        result = SearchResult(
            chunk_id="chunk-001",
            text="This is the chunk text",
            score=0.89,
        )
        assert result.chunk_id == "chunk-001"
        assert result.score == 0.89

    def test_full_result(self):
        result = SearchResult(
            chunk_id="chunk-001",
            text="This is the chunk text",
            score=0.89,
            start_time=45.2,
            end_time=75.8,
            speaker="SPEAKER_00",
            metadata={"lecture": "Week 1"},
            context_before="Previous text",
            context_after="Next text",
        )
        assert result.start_time == 45.2
        assert result.speaker == "SPEAKER_00"


class TestQueryResponse:
    def test_create_response(self):
        response = QueryResponse(
            results=[
                SearchResult(chunk_id="1", text="text", score=0.9),
            ],
            total=1,
            query="test query",
            collection="test_collection",
            query_time_ms=45.2,
        )
        assert len(response.results) == 1
        assert response.total == 1
        assert response.query_time_ms == 45.2

    def test_empty_results(self):
        response = QueryResponse(
            results=[],
            total=0,
            query="no match",
            collection="test_collection",
            query_time_ms=10.0,
        )
        assert len(response.results) == 0
