# NovaBeat AI Core

Proyecto para generación de música con dos rutas:

1. **Meta AudioCraft (MusicGen)** como referencia SOTA.
2. **Pipeline propio 100% local** (sin conectarte a servicios externos), inspirado en ideas de MusicGen/Stable Audio: tokenización discreta + modelo autoregresivo.

## Objetivo de esta iteración

Implementar una base que puedas entrenar tú mismo localmente, reutilizando ideas de los modelos grandes pero sin depender de su inferencia remota.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Opcional para correr también AudioCraft en local (si tienes GPU/entorno preparado):

```bash
pip install -e .[audiocraft]
```

## CLI

La CLI usa subcomandos:

### 1) Referencia con Meta MusicGen

```bash
novabeat-generate meta-generate \
  --prompt "Cinematic ambient con cuerdas y percusión tribal suave" \
  --output ./artifacts/meta.wav \
  --model-name facebook/musicgen-small \
  --duration 16
```

### 2) Entrenamiento local propio

```bash
novabeat-generate local-train \
  --wav-dir ./data/wavs \
  --corpus-jsonl ./artifacts/local_corpus.jsonl \
  --model-out ./artifacts/local_model.json \
  --order 8 \
  --chunk-size 4096
```

### 3) Generación con modelo local entrenado

```bash
novabeat-generate local-generate \
  --model-in ./artifacts/local_model.json \
  --output ./artifacts/local_generated.wav \
  --seconds 10 \
  --temperature 1.0 \
  --top-k 64
```

## Cómo está hecho el pipeline local

- `MuLawTokenizer`: convierte waveform a tokens discretos (mu-law companding).
- `LocalMusicPipeline`: construye corpus `.jsonl` desde `.wav`, entrena modelo y genera audio.
- `LocalTokenModel`: modelo autoregresivo n-gram con backoff + top-k sampling.

Esto no copia pesos propietarios; es una implementación propia inspirada en la estrategia de tokenización/autoregresión que usan sistemas modernos.
