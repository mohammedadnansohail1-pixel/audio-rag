"""Resource manager for GPU/memory management."""

import gc
from dataclasses import dataclass, field
from typing import Any
from contextlib import contextmanager

from audio_rag.core.exceptions import ResourceError
from audio_rag.config import ResourceConfig
from audio_rag.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    vram_gb: float
    instance: Any
    component_type: str  # asr, diarization, embedding, tts


class ResourceManager:
    """Manages GPU VRAM and system memory for model loading.
    
    Ensures we don't exceed VRAM limits by tracking loaded models
    and unloading when necessary.
    """
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.max_vram = config.max_vram_gb
        self._loaded_models: dict[str, ModelInfo] = {}
        self._load_order: list[str] = []  # For LRU eviction
        logger.info(f"ResourceManager initialized: max_vram={self.max_vram}GB")
    
    @property
    def used_vram(self) -> float:
        """Total VRAM currently in use."""
        return sum(m.vram_gb for m in self._loaded_models.values())
    
    @property
    def available_vram(self) -> float:
        """Available VRAM."""
        return max(0, self.max_vram - self.used_vram)
    
    def get_gpu_memory_info(self) -> dict:
        """Get actual GPU memory info from CUDA."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                return {
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 2),
                    "free_gb": round(total - reserved, 2),
                }
        except Exception:
            pass
        
        return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "free_gb": 0}
    
    def can_load(self, vram_required: float) -> bool:
        """Check if we can load a model with given VRAM requirement."""
        return vram_required <= self.available_vram
    
    def register_model(
        self, name: str, instance: Any, vram_gb: float, component_type: str
    ) -> None:
        """Register a loaded model with the resource manager.
        
        Args:
            name: Unique identifier for the model
            instance: The model instance (must have unload() method)
            vram_gb: VRAM usage in GB
            component_type: Type of component (asr, diarization, etc.)
        """
        if name in self._loaded_models:
            logger.warning(f"Model {name} already registered, updating")
            self._load_order.remove(name)
        
        self._loaded_models[name] = ModelInfo(
            name=name,
            vram_gb=vram_gb,
            instance=instance,
            component_type=component_type,
        )
        self._load_order.append(name)
        
        logger.info(
            f"Registered model: {name} ({vram_gb}GB) - "
            f"Total VRAM: {self.used_vram:.1f}/{self.max_vram}GB"
        )
    
    def unregister_model(self, name: str) -> None:
        """Unregister a model (after it's been unloaded)."""
        if name in self._loaded_models:
            del self._loaded_models[name]
            self._load_order.remove(name)
            logger.debug(f"Unregistered model: {name}")
    
    def ensure_vram(self, required_gb: float) -> None:
        """Ensure enough VRAM is available, unloading models if needed.
        
        Uses LRU eviction - oldest loaded models are unloaded first.
        
        Args:
            required_gb: VRAM needed in GB
            
        Raises:
            ResourceError: If cannot free enough VRAM
        """
        if self.can_load(required_gb):
            return
        
        logger.info(
            f"Need {required_gb}GB VRAM, have {self.available_vram:.1f}GB - "
            f"unloading models..."
        )
        
        # Unload oldest models until we have enough space
        while not self.can_load(required_gb) and self._load_order:
            oldest_name = self._load_order[0]
            model_info = self._loaded_models[oldest_name]
            
            logger.info(f"Unloading {oldest_name} to free {model_info.vram_gb}GB")
            
            # Call unload on the instance
            if hasattr(model_info.instance, 'unload'):
                model_info.instance.unload()
            
            self.unregister_model(oldest_name)
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        if not self.can_load(required_gb):
            raise ResourceError(
                f"Cannot free enough VRAM. Need {required_gb}GB, "
                f"max available {self.max_vram}GB"
            )
        
        logger.info(f"VRAM available: {self.available_vram:.1f}GB")
    
    @contextmanager
    def acquire(self, name: str, vram_required: float):
        """Context manager to acquire VRAM for a model.
        
        Usage:
            with resource_manager.acquire("whisper", 6.0):
                # load and use model
                pass
            # VRAM tracking updated
        """
        self.ensure_vram(vram_required)
        
        try:
            yield
        finally:
            # Note: actual unloading is handled by the model itself
            # This just ensures VRAM is available before loading
            pass
    
    def status(self) -> dict:
        """Get current resource status."""
        return {
            "max_vram_gb": self.max_vram,
            "used_vram_gb": round(self.used_vram, 2),
            "available_vram_gb": round(self.available_vram, 2),
            "loaded_models": [
                {
                    "name": m.name,
                    "vram_gb": m.vram_gb,
                    "component_type": m.component_type,
                }
                for m in self._loaded_models.values()
            ],
            "gpu_info": self.get_gpu_memory_info(),
        }
    
    def unload_all(self) -> None:
        """Unload all models."""
        logger.info("Unloading all models...")
        
        for name in list(self._load_order):
            model_info = self._loaded_models[name]
            if hasattr(model_info.instance, 'unload'):
                model_info.instance.unload()
            self.unregister_model(name)
        
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("All models unloaded")
