"""Tests for job validation."""

import pytest

from audio_rag.queue.validation import (
    AudioValidator,
    TenantValidator,
    JobValidator,
    SUPPORTED_AUDIO_FORMATS,
)
from audio_rag.queue.job import IngestJob
from audio_rag.queue.exceptions import InvalidAudioError, TenantValidationError


class TestAudioValidator:
    def test_supported_formats(self):
        assert ".mp3" in SUPPORTED_AUDIO_FORMATS
        assert ".wav" in SUPPORTED_AUDIO_FORMATS
        assert ".flac" in SUPPORTED_AUDIO_FORMATS

    def test_validate_missing_file(self, tmp_path):
        validator = AudioValidator()
        fake_path = tmp_path / "nonexistent.mp3"
        
        with pytest.raises(InvalidAudioError):
            validator.validate(str(fake_path))

    def test_validate_unsupported_extension(self, tmp_path):
        validator = AudioValidator()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not audio")
        
        with pytest.raises(InvalidAudioError):
            validator.validate(str(txt_file))

    def test_validate_empty_file(self, tmp_path):
        validator = AudioValidator()
        empty_file = tmp_path / "empty.mp3"
        empty_file.write_bytes(b"")
        
        with pytest.raises(InvalidAudioError):
            validator.validate(str(empty_file))

    def test_validate_file_too_large(self, tmp_path):
        validator = AudioValidator(max_file_size_mb=0.001)  # 1KB limit
        large_file = tmp_path / "large.mp3"
        large_file.write_bytes(b"x" * 2000)  # 2KB
        
        with pytest.raises(InvalidAudioError):
            validator.validate(str(large_file))

    def test_validate_valid_file(self, tmp_path):
        validator = AudioValidator()
        valid_file = tmp_path / "test.mp3"
        valid_file.write_bytes(b"ID3" + b"\x00" * 100)
        
        # Should not raise
        result = validator.validate(str(valid_file))
        assert result is True or result is None


class TestTenantValidator:
    def test_valid_tenant_simple(self):
        validator = TenantValidator(strict_format=False)
        result = validator.validate("my_tenant_123")
        assert result is True or result is None

    def test_valid_tenant_strict(self):
        validator = TenantValidator(strict_format=True)
        result = validator.validate("audio_rag_unt_cs_5500_fall2025")
        assert result is True or result is None

    def test_invalid_tenant_empty(self):
        validator = TenantValidator()
        with pytest.raises(TenantValidationError):
            validator.validate("")

    def test_invalid_tenant_special_chars(self):
        validator = TenantValidator()
        with pytest.raises(TenantValidationError):
            validator.validate("tenant-with-dashes")

    def test_invalid_tenant_strict_format(self):
        validator = TenantValidator(strict_format=True)
        with pytest.raises(TenantValidationError):
            validator.validate("not_matching_pattern")

    def test_parse_tenant_id(self):
        validator = TenantValidator(strict_format=True)
        parts = validator.parse_tenant_id("audio_rag_unt_cs_5500_fall2025")
        
        assert parts is not None
        assert parts.get("university") == "unt"
        assert parts.get("department") == "cs"

    def test_build_tenant_id(self):
        validator = TenantValidator(strict_format=True)
        tenant_id = validator.build_tenant_id(
            university="unt",
            department="cs",
            course="5500",
            semester="fall2025",
        )
        assert tenant_id == "audio_rag_unt_cs_5500_fall2025"


class TestJobValidator:
    def test_validate_job_valid(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 100)
        
        validator = JobValidator(
            audio_validator=AudioValidator(),
            tenant_validator=TenantValidator(strict_format=False),
        )
        
        job = IngestJob(
            tenant_id="valid_tenant",
            audio_path=str(audio_file),
        )
        
        # Should not raise
        validator.validate(job)

    def test_validate_job_invalid_audio(self, tmp_path):
        validator = JobValidator(
            audio_validator=AudioValidator(),
            tenant_validator=TenantValidator(strict_format=False),
        )
        
        job = IngestJob(
            tenant_id="valid_tenant",
            audio_path="/nonexistent/path.mp3",
        )
        
        with pytest.raises(InvalidAudioError):
            validator.validate(job)

    def test_validate_job_invalid_tenant(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 100)
        
        validator = JobValidator(
            audio_validator=AudioValidator(),
            tenant_validator=TenantValidator(strict_format=False),
        )
        
        job = IngestJob(
            tenant_id="",
            audio_path=str(audio_file),
        )
        
        with pytest.raises(TenantValidationError):
            validator.validate(job)
