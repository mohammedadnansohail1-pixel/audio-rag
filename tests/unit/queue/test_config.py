"""Tests for queue configuration."""

import pytest
from pydantic import ValidationError

from audio_rag.queue.config import (
    QueueConfig,
    QueueDefinition,
    RedisConfig,
    WorkerConfig,
    DEFAULT_QUEUE_CONFIG,
)


class TestRedisConfig:
    def test_default_values(self):
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None

    def test_url_without_password(self):
        config = RedisConfig(host="myhost", port=6380, db=1)
        assert config.url == "redis://myhost:6380/1"

    def test_url_with_password(self):
        config = RedisConfig(host="myhost", password="secret")
        assert config.url == "redis://:secret@myhost:6379/0"

    def test_invalid_port(self):
        with pytest.raises(ValidationError):
            RedisConfig(port=0)
        with pytest.raises(ValidationError):
            RedisConfig(port=70000)

    def test_invalid_db(self):
        with pytest.raises(ValidationError):
            RedisConfig(db=-1)
        with pytest.raises(ValidationError):
            RedisConfig(db=16)


class TestQueueDefinition:
    def test_default_values(self):
        queue = QueueDefinition(name="test")
        assert queue.name == "test"
        assert queue.timeout > 0
        assert queue.max_depth > 0

    def test_invalid_timeout(self):
        with pytest.raises(ValidationError):
            QueueDefinition(name="test", timeout=10)  # Too low

    def test_invalid_max_depth(self):
        with pytest.raises(ValidationError):
            QueueDefinition(name="test", max_depth=0)


class TestWorkerConfig:
    def test_default_values(self):
        config = WorkerConfig()
        assert config.max_memory_mb > 0
        assert config.poll_interval > 0
        assert config.job_timeout > 0

    def test_invalid_memory(self):
        with pytest.raises(ValidationError):
            WorkerConfig(max_memory_mb=100)  # Too low


class TestQueueConfig:
    def test_default_config_valid(self):
        config = DEFAULT_QUEUE_CONFIG
        assert config.redis is not None
        assert len(config.queues) > 0

    def test_get_queue_names(self):
        config = DEFAULT_QUEUE_CONFIG
        names = config.get_queue_names()
        assert "high" in names
        assert "normal" in names
        assert "low" in names

    def test_get_queue(self):
        config = DEFAULT_QUEUE_CONFIG
        high_queue = config.get_queue("high")
        assert high_queue is not None
        assert high_queue.name == "high"

    def test_get_queue_nonexistent(self):
        config = DEFAULT_QUEUE_CONFIG
        assert config.get_queue("nonexistent") is None

    def test_get_timeout_for_queue(self):
        config = DEFAULT_QUEUE_CONFIG
        timeout = config.get_timeout_for_queue("normal")
        assert timeout > 0

    def test_get_timeout_for_nonexistent_queue(self):
        config = DEFAULT_QUEUE_CONFIG
        # Should return worker default timeout
        timeout = config.get_timeout_for_queue("nonexistent")
        assert timeout == config.worker.job_timeout
