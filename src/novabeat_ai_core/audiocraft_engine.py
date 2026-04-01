from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "facebook/musicgen-small"
    duration: int = 12
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    cfg_coef: float = 3.0
    sample_rate: int = 32000


class AudiocraftMusicGenerator:
    """Wrapper sobre Meta AudioCraft (MusicGen) para generación texto->audio."""

    def __init__(self, config: GenerationConfig | None = None):
        self.config = config or GenerationConfig()
        self._model = None

    def _load_dependencies(self) -> tuple[Any, Any]:
        try:
            models_module = importlib.import_module("audiocraft.models")
            audio_module = importlib.import_module("audiocraft.data.audio")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AudioCraft no está instalado. Instala extras con: pip install -e '.[audiocraft]'"
            ) from exc

        return models_module.MusicGen, audio_module.audio_write

    def _load_model(self):
        if self._model is not None:
            return self._model

        MusicGen, _ = self._load_dependencies()
        model = MusicGen.get_pretrained(self.config.model_name)
        model.set_generation_params(
            duration=self.config.duration,
            use_sampling=True,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            cfg_coef=self.config.cfg_coef,
        )
        self._model = model
        return model

    def generate_to_wav(self, prompt: str, output_path: str | Path) -> Path:
        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")

        model = self._load_model()
        _, audio_write = self._load_dependencies()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stem_without_suffix = output_path.with_suffix("")

        wav = model.generate([prompt])[0].cpu()
        audio_write(
            str(stem_without_suffix),
            wav,
            self.config.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )
        return stem_without_suffix.with_suffix(".wav")
