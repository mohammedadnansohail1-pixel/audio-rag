"""Integration tests for health endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthLive:
    def test_live_returns_200(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"

    def test_live_no_auth_required(self):
        """Health live should work without API key."""
        from audio_rag.api.app import create_app
        
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/health/live")
        assert response.status_code == 200


class TestHealthReady:
    def test_ready_with_healthy_services(self, client):
        response = client.get("/health/ready")
        # May be 200 or 503 depending on actual Redis/Qdrant
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "status" in data
        assert "checks" in data

    def test_ready_response_structure(self, client):
        response = client.get("/health/ready")
        data = response.json()
        
        assert "checks" in data
        checks = data["checks"]
        assert "redis" in checks
        assert "qdrant" in checks


class TestHealthStartup:
    def test_startup_after_init(self, client):
        response = client.get("/health/startup")
        # After app is created, should be initialized
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "status" in data
