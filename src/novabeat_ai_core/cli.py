from __future__ import annotations

import argparse
from pathlib import Path

from .audiocraft_engine import AudiocraftMusicGenerator, GenerationConfig
from .local_pipeline import LocalMusicPipeline, LocalTrainingConfig


def generate_main() -> None:
    parser = argparse.ArgumentParser(description="NovaBeat CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cloud = sub.add_parser("meta-generate", help="Generar con MusicGen (AudioCraft)")
    cloud.add_argument("--prompt", required=True)
    cloud.add_argument("--output", type=Path, required=True)
    cloud.add_argument("--model-name", default="facebook/musicgen-small")
    cloud.add_argument("--duration", type=int, default=12)
    cloud.add_argument("--temperature", type=float, default=1.0)
    cloud.add_argument("--top-k", type=int, default=250)
    cloud.add_argument("--top-p", type=float, default=0.0)
    cloud.add_argument("--cfg-coef", type=float, default=3.0)
    cloud.add_argument("--sample-rate", type=int, default=32000)

    local_train = sub.add_parser("local-train", help="Entrenamiento 100% local")
    local_train.add_argument("--wav-dir", type=Path, required=True)
    local_train.add_argument("--corpus-jsonl", type=Path, required=True)
    local_train.add_argument("--model-out", type=Path, required=True)
    local_train.add_argument("--order", type=int, default=8)
    local_train.add_argument("--quantization-levels", type=int, default=256)
    local_train.add_argument("--sample-rate", type=int, default=32000)
    local_train.add_argument("--chunk-size", type=int, default=4096)

    local_gen = sub.add_parser("local-generate", help="Generar con modelo local entrenado")
    local_gen.add_argument("--model-in", type=Path, required=True)
    local_gen.add_argument("--output", type=Path, required=True)
    local_gen.add_argument("--seconds", type=int, default=8)
    local_gen.add_argument("--temperature", type=float, default=1.0)
    local_gen.add_argument("--top-k", type=int, default=64)
    local_gen.add_argument("--order", type=int, default=8)
    local_gen.add_argument("--quantization-levels", type=int, default=256)
    local_gen.add_argument("--sample-rate", type=int, default=32000)

    args = parser.parse_args()

    if args.cmd == "meta-generate":
        config = GenerationConfig(
            model_name=args.model_name,
            duration=args.duration,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            cfg_coef=args.cfg_coef,
            sample_rate=args.sample_rate,
        )
        generator = AudiocraftMusicGenerator(config=config)
        output = generator.generate_to_wav(prompt=args.prompt, output_path=args.output)
        print(f"Audio generado (Meta): {output}")
        return

    local_config = LocalTrainingConfig(
        order=args.order,
        quantization_levels=args.quantization_levels,
        sample_rate=args.sample_rate,
        chunk_size=getattr(args, "chunk_size", 4096),
    )
    pipeline = LocalMusicPipeline(local_config)

    if args.cmd == "local-train":
        corpus = pipeline.build_training_corpus(args.wav_dir, args.corpus_jsonl)
        model_path = pipeline.train_local_model(corpus, args.model_out)
        print(f"Corpus local: {corpus}")
        print(f"Modelo local entrenado: {model_path}")
        return

    output = pipeline.generate_from_model(
        model_path=args.model_in,
        output_wav=args.output,
        seconds=args.seconds,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"Audio generado (local): {output}")


if __name__ == "__main__":
    generate_main()
