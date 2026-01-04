"""Integration tests for jobs endpoint."""

import pytest
from fastapi.testclient import TestClient


class TestJobsEndpoint:
    def test_get_job_requires_auth(self):
        """Job status should require API key."""
        from audio_rag.api.app import create_app
        
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/api/v1/jobs/some-job-id")
        assert response.status_code == 401

    def test_get_job_not_found(self, client, api_headers):
        """Return 404 for non-existent job."""
        response = client.get(
            "/api/v1/jobs/nonexistent-job-id",
            headers=api_headers,
        )
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data.get("detail", data)

    def test_get_job_after_ingest(self, client, api_headers, tmp_path):
        """Check status of submitted job."""
        # First submit a job
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 100)
        
        with open(audio_file, "rb") as f:
            ingest_response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.mp3", f, "audio/mpeg")},
                headers=api_headers,
            )
        
        assert ingest_response.status_code == 202
        job_id = ingest_response.json()["job_id"]
        
        # Now check status
        response = client.get(
            f"/api/v1/jobs/{job_id}",
            headers=api_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in ["pending", "running", "completed", "failed"]
        assert "stage" in data
        assert "created_at" in data

    def test_cancel_job_not_found(self, client, api_headers):
        """Return 404 when cancelling non-existent job."""
        response = client.delete(
            "/api/v1/jobs/nonexistent-job-id",
            headers=api_headers,
        )
        assert response.status_code == 404

    def test_cancel_pending_job(self, client, api_headers, tmp_path):
        """Cancel a pending job."""
        # First submit a job
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 100)
        
        with open(audio_file, "rb") as f:
            ingest_response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.mp3", f, "audio/mpeg")},
                headers=api_headers,
            )
        
        job_id = ingest_response.json()["job_id"]
        
        # Cancel the job
        response = client.delete(
            f"/api/v1/jobs/{job_id}",
            headers=api_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "cancelled" in data
        assert "message" in data


class TestAPIInfo:
    def test_api_info(self, client, api_headers):
        """Get API version info."""
        response = client.get("/api/v1/", headers=api_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "v1"
        assert data["status"] == "active"
        assert "endpoints" in data
