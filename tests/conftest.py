"""Shared test fixtures."""

import pytest
from fakeredis import FakeRedis
from fastapi.testclient import TestClient


@pytest.fixture
def fake_redis():
    """Isolated fake Redis for each test."""
    redis = FakeRedis(decode_responses=False)
    yield redis
    redis.flushall()


@pytest.fixture
def mock_connection_manager(fake_redis):
    """Mock RedisConnectionManager using fakeredis."""
    from unittest.mock import MagicMock
    
    manager = MagicMock()
    manager.get_connection.return_value = fake_redis
    manager.execute.side_effect = lambda cmd, *args, **kwargs: getattr(fake_redis, cmd)(*args, **kwargs)
    manager.is_healthy.return_value = True
    manager.close = MagicMock()
    manager._redis = fake_redis
    
    return manager


@pytest.fixture
def queue_config():
    """Default queue configuration for tests."""
    from audio_rag.queue.config import DEFAULT_QUEUE_CONFIG
    return DEFAULT_QUEUE_CONFIG


@pytest.fixture
def mock_queue(mock_connection_manager, queue_config):
    """AudioRAGQueue with fakeredis backend."""
    from audio_rag.queue import AudioRAGQueue
    from audio_rag.queue.validation import DEFAULT_JOB_VALIDATOR
    
    queue = AudioRAGQueue(
        connection_manager=mock_connection_manager,
        config=queue_config,
        validator=DEFAULT_JOB_VALIDATOR,
    )
    return queue


@pytest.fixture
def app(mock_queue):
    """FastAPI app with mocked dependencies."""
    from audio_rag.api.app import create_app
    from audio_rag.api.deps import get_queue, get_request_context, RequestContext
    
    app = create_app()
    
    def override_get_queue():
        return mock_queue
    
    def override_get_context():
        return RequestContext(
            request_id="test-request-id",
            tenant_id="test_tenant",
            tier="premium",
            api_key_name="test-key",
        )
    
    app.dependency_overrides[get_queue] = override_get_queue
    app.dependency_overrides[get_request_context] = override_get_context
    
    yield app
    app.dependency_overrides = {}


@pytest.fixture
def client(app):
    """Test client with overridden dependencies."""
    return TestClient(app)


@pytest.fixture
def api_headers():
    """Valid API headers for authenticated requests."""
    return {"X-API-Key": "dev-key-12345"}


@pytest.fixture
def sample_audio_path(tmp_path):
    """Create a sample audio file for testing."""
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"ID3" + b"\x00" * 100)
    return audio_file
