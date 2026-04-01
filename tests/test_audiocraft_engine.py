import pytest

from novabeat_ai_core.audiocraft_engine import AudiocraftMusicGenerator, GenerationConfig


def test_generation_config_defaults():
    config = GenerationConfig()
    assert config.model_name == "facebook/musicgen-small"
    assert config.duration == 12
    assert config.sample_rate == 32000


def test_generate_rejects_empty_prompt():
    generator = AudiocraftMusicGenerator()
    with pytest.raises(ValueError):
        generator.generate_to_wav("   ", "out.wav")


def test_missing_audiocraft_dependency(monkeypatch):
    generator = AudiocraftMusicGenerator()

    def fake_import_module(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="AudioCraft no está instalado"):
        generator._load_dependencies()
