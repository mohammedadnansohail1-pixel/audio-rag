"""Integration tests for ingest endpoint."""

import pytest
from fastapi.testclient import TestClient


class TestIngestEndpoint:
    def test_ingest_requires_auth(self):
        """Ingest should require API key."""
        from audio_rag.api.app import create_app
        
        app = create_app()
        client = TestClient(app)
        
        response = client.post("/api/v1/ingest")
        assert response.status_code == 401

    def test_ingest_valid_file(self, client, api_headers, tmp_path):
        """Upload valid audio file."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 100)
        
        with open(audio_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.mp3", f, "audio/mpeg")},
                data={"priority": "normal"},
                headers=api_headers,
            )
        
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["filename"] == "test.mp3"

    def test_ingest_with_metadata(self, client, api_headers, tmp_path):
        """Upload with metadata."""
        import json
        
        audio_file = tmp_path / "lecture.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 100)
        
        metadata = json.dumps({"lecture": "Week 1", "professor": "Dr. Smith"})
        
        with open(audio_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("lecture.mp3", f, "audio/mpeg")},
                data={"metadata": metadata, "priority": "high"},
                headers=api_headers,
            )
        
        assert response.status_code == 202
        data = response.json()
        assert data["priority"] == "high"

    def test_ingest_invalid_file_type(self, client, api_headers, tmp_path):
        """Reject non-audio files."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not audio content")
        
        with open(txt_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.txt", f, "text/plain")},
                headers=api_headers,
            )
        
        assert response.status_code == 422

    def test_ingest_empty_file(self, client, api_headers, tmp_path):
        """Reject empty files."""
        empty_file = tmp_path / "empty.mp3"
        empty_file.write_bytes(b"")
        
        with open(empty_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("empty.mp3", f, "audio/mpeg")},
                headers=api_headers,
            )
        
        assert response.status_code == 422

    def test_ingest_missing_file(self, client, api_headers):
        """Reject request without file."""
        response = client.post("/api/v1/ingest", headers=api_headers)
        assert response.status_code == 422

    def test_ingest_with_collection(self, client, api_headers, tmp_path):
        """Upload to specific collection."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 100)
        
        with open(audio_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.mp3", f, "audio/mpeg")},
                data={"collection": "audio_rag_test_cs_5500_fall2025"},
                headers=api_headers,
            )
        
        assert response.status_code == 202
        data = response.json()
        assert data["collection"] == "audio_rag_test_cs_5500_fall2025"
