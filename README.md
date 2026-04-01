# NovaBeat AI Core

Proyecto para generación de música con dos rutas:

1. **Meta AudioCraft (MusicGen)** como referencia SOTA.
2. **Pipeline propio 100% local** (sin conectarte a servicios externos), inspirado en ideas de MusicGen/Stable Audio: tokenización discreta + modelo autoregresivo.

Además, incluye integración opcional con **Supabase** para guardar metadata de generaciones.

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

La CLI usa subcomandos.

### 1) Referencia con Meta MusicGen

```bash
novabeat-generate \
  --supabase-url https://jqobhkmtazyprlddfvwe.supabase.co \
  --supabase-key "$SUPABASE_KEY" \
  meta-generate \
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
novabeat-generate \
  --supabase-url https://jqobhkmtazyprlddfvwe.supabase.co \
  --supabase-key "$SUPABASE_KEY" \
  local-generate \
  --model-in ./artifacts/local_model.json \
  --output ./artifacts/local_generated.wav \
  --prompt "dark trap 140 bpm" \
  --seconds 10 \
  --temperature 1.0 \
  --top-k 64
```

## Tabla esperada en Supabase

Crea una tabla `generations` (o cambia `--supabase-table`) con columnas sugeridas:

- `id` bigint generated always as identity primary key
- `provider` text
- `prompt` text
- `output_file` text
- `created_at` timestamptz
- `extra` jsonb

## Cómo está hecho el pipeline local

- `MuLawTokenizer`: convierte waveform a tokens discretos (mu-law companding).
- `LocalMusicPipeline`: construye corpus `.jsonl` desde `.wav`, entrena modelo y genera audio.
- `LocalTokenModel`: modelo autoregresivo n-gram con backoff + top-k sampling.

Esto no copia pesos propietarios; es una implementación propia inspirada en la estrategia de tokenización/autoregresión que usan sistemas modernos.

## Entidades de base de datos (Supabase)

Ya hay una migración lista en `supabase/migrations/0001_generations.sql` y una guía rápida en `supabase/README.md`.

## Nota de resolución de conflictos

Si al hacer merge ves marcas como `<<<<<<<`, `=======` o `>>>>>>>`, elimina el bloque completo y conserva esta versión final del README.
