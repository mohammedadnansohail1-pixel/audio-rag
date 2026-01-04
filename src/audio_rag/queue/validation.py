"""Pre-queue validation for jobs.

Validates:
- Audio files (existence, format, size, duration)
- Tenant IDs (format, existence)
- Job parameters
"""

from __future__ import annotations

import logging
import mimetypes
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from audio_rag.queue.exceptions import InvalidAudioError, TenantValidationError

if TYPE_CHECKING:
    from audio_rag.queue.job import IngestJob

logger = logging.getLogger(__name__)


# Supported audio formats
SUPPORTED_AUDIO_FORMATS: set[str] = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
    ".webm",
    ".mp4",  # May contain audio
}

# MIME types for audio
SUPPORTED_MIME_TYPES: set[str] = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/x-m4a",
    "audio/mp4",
    "audio/aac",
    "audio/x-aac",
    "audio/x-ms-wma",
    "audio/opus",
    "audio/webm",
    "video/mp4",  # May contain audio track
    "video/webm",
}

# Tenant ID pattern: audio_rag_{university}_{department}_{course}_{semester}
# Each segment: lowercase alphanumeric, 2-32 chars
TENANT_ID_PATTERN = re.compile(
    r"^audio_rag_[a-z0-9]{2,32}_[a-z0-9]{2,32}_[a-z0-9]{2,32}_[a-z0-9]{2,32}$"
)

# Alternative simpler pattern for testing
TENANT_ID_SIMPLE_PATTERN = re.compile(r"^[a-z0-9_]{5,128}$")


class AudioValidator:
    """Validates audio files before queuing.
    
    Checks:
    - File exists
    - File extension is supported
    - MIME type is audio
    - File size within limits
    - Duration within limits (optional, requires ffprobe)
    """
    
    def __init__(
        self,
        max_file_size_mb: float = 500.0,
        max_duration_minutes: float = 180.0,
        min_duration_seconds: float = 1.0,
        check_duration: bool = True,
    ) -> None:
        """Initialize validator.
        
        Args:
            max_file_size_mb: Maximum file size in MB
            max_duration_minutes: Maximum audio duration in minutes
            min_duration_seconds: Minimum audio duration in seconds
            check_duration: Whether to check duration (requires ffprobe)
        """
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.max_duration_seconds = max_duration_minutes * 60
        self.min_duration_seconds = min_duration_seconds
        self.check_duration = check_duration
    
    def validate(self, audio_path: str | Path) -> None:
        """Validate an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Raises:
            InvalidAudioError: If validation fails
        """
        path = Path(audio_path)
        
        # Check existence
        if not path.exists():
            raise InvalidAudioError(
                "Audio file does not exist",
                audio_path=audio_path,
                reason="file_not_found",
            )
        
        if not path.is_file():
            raise InvalidAudioError(
                "Path is not a file",
                audio_path=audio_path,
                reason="not_a_file",
            )
        
        # Check extension
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_FORMATS:
            raise InvalidAudioError(
                f"Unsupported audio format: {suffix}",
                audio_path=audio_path,
                reason="unsupported_format",
            )
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and mime_type not in SUPPORTED_MIME_TYPES:
            logger.warning(
                f"Unexpected MIME type {mime_type} for {path}, proceeding anyway"
            )
        
        # Check file size
        file_size = path.stat().st_size
        if file_size == 0:
            raise InvalidAudioError(
                "Audio file is empty",
                audio_path=audio_path,
                reason="empty_file",
            )
        
        if file_size > self.max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            raise InvalidAudioError(
                f"Audio file too large: {size_mb:.1f}MB (max: {max_mb:.1f}MB)",
                audio_path=audio_path,
                reason="file_too_large",
            )
        
        # Check duration (optional)
        if self.check_duration:
            self._validate_duration(path)
    
    def _validate_duration(self, path: Path) -> None:
        """Validate audio duration using ffprobe.
        
        Args:
            path: Path to audio file
            
        Raises:
            InvalidAudioError: If duration is invalid or can't be determined
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                logger.warning(
                    f"ffprobe failed for {path}: {result.stderr}"
                )
                # Don't fail validation if ffprobe fails
                return
            
            duration_str = result.stdout.strip()
            if not duration_str:
                logger.warning(f"Could not determine duration for {path}")
                return
            
            duration = float(duration_str)
            
            if duration < self.min_duration_seconds:
                raise InvalidAudioError(
                    f"Audio too short: {duration:.1f}s (min: {self.min_duration_seconds:.1f}s)",
                    audio_path=str(path),
                    reason="duration_too_short",
                )
            
            if duration > self.max_duration_seconds:
                duration_min = duration / 60
                max_min = self.max_duration_seconds / 60
                raise InvalidAudioError(
                    f"Audio too long: {duration_min:.1f}min (max: {max_min:.1f}min)",
                    audio_path=str(path),
                    reason="duration_too_long",
                )
                
        except subprocess.TimeoutExpired:
            logger.warning(f"ffprobe timed out for {path}")
        except FileNotFoundError:
            logger.warning("ffprobe not found, skipping duration check")
        except ValueError as e:
            logger.warning(f"Could not parse duration for {path}: {e}")


class TenantValidator:
    """Validates tenant IDs.
    
    Ensures:
    - Tenant ID format is valid
    - Tenant exists (optional, requires tenant registry)
    - Tenant is active (optional)
    """
    
    def __init__(
        self,
        strict_format: bool = True,
        known_tenants: set[str] | None = None,
    ) -> None:
        """Initialize validator.
        
        Args:
            strict_format: If True, enforce full tenant ID pattern.
                          If False, allow simpler patterns (for testing).
            known_tenants: Optional set of known valid tenant IDs.
                          If provided, tenant must be in this set.
        """
        self.strict_format = strict_format
        self.known_tenants = known_tenants
    
    def validate(self, tenant_id: str) -> None:
        """Validate a tenant ID.
        
        Args:
            tenant_id: The tenant ID to validate
            
        Raises:
            TenantValidationError: If validation fails
        """
        if not tenant_id:
            raise TenantValidationError(
                "Tenant ID is required",
                tenant_id=tenant_id,
                reason="empty",
            )
        
        # Check format
        if self.strict_format:
            if not TENANT_ID_PATTERN.match(tenant_id):
                raise TenantValidationError(
                    "Invalid tenant ID format. Expected: audio_rag_{university}_{department}_{course}_{semester}",
                    tenant_id=tenant_id,
                    reason="invalid_format",
                )
        else:
            if not TENANT_ID_SIMPLE_PATTERN.match(tenant_id):
                raise TenantValidationError(
                    "Invalid tenant ID format. Must be lowercase alphanumeric with underscores, 5-128 chars",
                    tenant_id=tenant_id,
                    reason="invalid_format",
                )
        
        # Check if tenant is known (if registry provided)
        if self.known_tenants is not None:
            if tenant_id not in self.known_tenants:
                raise TenantValidationError(
                    "Unknown tenant",
                    tenant_id=tenant_id,
                    reason="unknown_tenant",
                )
    
    @staticmethod
    def parse_tenant_id(tenant_id: str) -> dict[str, str]:
        """Parse tenant ID into components.
        
        Args:
            tenant_id: Valid tenant ID
            
        Returns:
            Dict with keys: university, department, course, semester
            
        Raises:
            TenantValidationError: If tenant ID format is invalid
        """
        if not TENANT_ID_PATTERN.match(tenant_id):
            raise TenantValidationError(
                "Cannot parse invalid tenant ID",
                tenant_id=tenant_id,
                reason="invalid_format",
            )
        
        # Remove prefix and split
        parts = tenant_id[10:].split("_")  # Remove "audio_rag_"
        
        return {
            "university": parts[0],
            "department": parts[1],
            "course": parts[2],
            "semester": parts[3],
        }
    
    @staticmethod
    def build_tenant_id(
        university: str,
        department: str,
        course: str,
        semester: str,
    ) -> str:
        """Build tenant ID from components.
        
        Args:
            university: University code (e.g., "unt")
            department: Department code (e.g., "cs")
            course: Course code (e.g., "5500")
            semester: Semester code (e.g., "fall2025")
            
        Returns:
            Valid tenant ID
            
        Raises:
            TenantValidationError: If any component is invalid
        """
        # Normalize to lowercase
        parts = [
            university.lower(),
            department.lower(),
            course.lower(),
            semester.lower(),
        ]
        
        # Validate each part
        for name, value in zip(
            ["university", "department", "course", "semester"],
            parts,
        ):
            if not re.match(r"^[a-z0-9]{2,32}$", value):
                raise TenantValidationError(
                    f"Invalid {name}: must be 2-32 lowercase alphanumeric characters",
                    tenant_id=f"audio_rag_{'_'.join(parts)}",
                    reason=f"invalid_{name}",
                )
        
        return f"audio_rag_{'_'.join(parts)}"


class JobValidator:
    """Validates complete jobs before queuing."""
    
    def __init__(
        self,
        audio_validator: AudioValidator | None = None,
        tenant_validator: TenantValidator | None = None,
    ) -> None:
        """Initialize job validator.
        
        Args:
            audio_validator: Audio file validator
            tenant_validator: Tenant ID validator
        """
        self.audio_validator = audio_validator or AudioValidator()
        self.tenant_validator = tenant_validator or TenantValidator(strict_format=False)
    
    def validate(self, job: IngestJob) -> None:
        """Validate a job before queuing.
        
        Args:
            job: The job to validate
            
        Raises:
            InvalidAudioError: If audio validation fails
            TenantValidationError: If tenant validation fails
        """
        # Validate tenant
        self.tenant_validator.validate(job.tenant_id)
        
        # Validate audio
        self.audio_validator.validate(job.audio_path)
        
        logger.debug(f"Job validation passed: {job.job_id}")


# Default validator instances
DEFAULT_AUDIO_VALIDATOR = AudioValidator()
DEFAULT_TENANT_VALIDATOR = TenantValidator(strict_format=False)
DEFAULT_JOB_VALIDATOR = JobValidator()
