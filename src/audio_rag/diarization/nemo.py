"""NVIDIA NeMo speaker diarization implementation."""
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

from audio_rag.config import DiarizationConfig
from audio_rag.core import BaseDiarizer, DiarizationError, TranscriptSegment
from audio_rag.diarization.base import DiarizationRegistry
from audio_rag.utils import get_logger, require_loaded, timed

logger = get_logger(__name__)

VRAM_ESTIMATE = 3.0


@DiarizationRegistry.register("nemo")
class NemoDiarizer(BaseDiarizer):
    """NVIDIA NeMo speaker diarization backend."""

    def __init__(self, config: DiarizationConfig):
        self.config = config
        self._msdd_model = None
        self._device = self._resolve_device(config.device)
        self._temp_dir = None
        logger.info(f"NemoDiarizer initialized: device={self._device}")

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self) -> None:
        if self._msdd_model is not None:
            return

        try:
            import torch
            from nemo.collections.asr.models import ClusteringDiarizer
            from omegaconf import OmegaConf

            logger.info("Loading NeMo ClusteringDiarizer...")
            self._temp_dir = tempfile.mkdtemp(prefix="nemo_diar_")

            config = OmegaConf.create({
                "device": self._device,
                "sample_rate": 16000,
                "num_workers": 1,
                "batch_size": 64,
                "verbose": True,
                "diarizer": {
                    "manifest_filepath": None,
                    "out_dir": self._temp_dir,
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": True,
                    "vad": {
                        "model_path": "vad_multilingual_marblenet",
                        "external_vad_manifest": None,
                        "parameters": {
                            "window_length_in_sec": 0.15,
                            "shift_length_in_sec": 0.01,
                            "smoothing": "median",
                            "overlap": 0.5,
                            "onset": 0.5,
                            "offset": 0.3,
                            "pad_onset": 0.1,
                            "pad_offset": 0.1,
                            "min_duration_on": 0.2,
                            "min_duration_off": 0.2,
                        }
                    },
                    "speaker_embeddings": {
                        "model_path": "titanet_large",
                        "parameters": {
                            "window_length_in_sec": 1.5,
                            "shift_length_in_sec": 0.75,
                            "multiscale_weights": [1, 1, 1, 1, 1],
                            "save_embeddings": False,
                        }
                    },
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": False,
                            "max_num_speakers": self.config.max_speakers or 8,
                            "enhanced_count_thres": 80,
                            "max_rp_threshold": 0.25,
                            "sparse_search_volume": 30,
                        }
                    },
                }
            })

            self._msdd_model = ClusteringDiarizer(cfg=config)
            logger.info("NeMo ClusteringDiarizer loaded successfully")

        except Exception as e:
            raise DiarizationError(f"Failed to load NeMo diarizer: {e}")

    def unload(self) -> None:
        if self._msdd_model is None:
            return
        
        import torch
        del self._msdd_model
        self._msdd_model = None
        
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._msdd_model is not None

    @property
    def vram_required(self) -> float:
        return VRAM_ESTIMATE

    @timed
    @require_loaded
    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[TranscriptSegment]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise DiarizationError(f"Audio file not found: {audio_path}")

        max_spk = max_speakers or self.config.max_speakers or 8

        try:
            logger.info(f"Diarizing {audio_path.name} with NeMo...")

            manifest_path = Path(self._temp_dir) / "manifest.json"
            with open(manifest_path, "w") as f:
                entry = {
                    "audio_filepath": str(audio_path.absolute()),
                    "offset": 0,
                    "duration": None,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": max_spk,
                    "rttm_filepath": None,
                    "uem_filepath": None,
                }
                f.write(json.dumps(entry) + "\n")

            self._msdd_model._cfg.diarizer.manifest_filepath = str(manifest_path)
            self._msdd_model._cfg.diarizer.clustering.parameters.max_num_speakers = max_spk
            self._msdd_model.diarize()

            rttm_dir = Path(self._temp_dir) / "pred_rttms"
            segments = []
            if rttm_dir.exists():
                for rttm_file in rttm_dir.glob("*.rttm"):
                    segments = self._parse_rttm(rttm_file)
                    break

            speakers = set(s.speaker for s in segments if s.speaker)
            logger.info(f"Diarization complete: {len(segments)} turns, {len(speakers)} speakers")
            return segments

        except Exception as e:
            raise DiarizationError(f"Diarization failed: {e}")

    def _parse_rttm(self, rttm_path: Path) -> list[TranscriptSegment]:
        segments = []
        with open(rttm_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    segments.append(TranscriptSegment(
                        text="",
                        start=start,
                        end=start + duration,
                        speaker=speaker,
                        confidence=None,
                        language=None,
                    ))
        segments.sort(key=lambda s: s.start)
        return segments
