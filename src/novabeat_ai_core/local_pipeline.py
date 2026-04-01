from __future__ import annotations

import base64
import json
import math
import random
import struct
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

State = Tuple[int, ...]


@dataclass(frozen=True)
class LocalTrainingConfig:
    order: int = 8
    quantization_levels: int = 256
    sample_rate: int = 32000
    chunk_size: int = 4096


class MuLawTokenizer:
    """Tokenizer discreto local inspirado en pipelines codec/token de MusicGen y Stable Audio."""

    def __init__(self, quantization_levels: int = 256):
        if quantization_levels < 32:
            raise ValueError("quantization_levels debe ser >= 32")
        self.quantization_levels = quantization_levels

    def encode_waveform(self, waveform: Sequence[float]) -> List[int]:
        mu = self.quantization_levels - 1
        out: List[int] = []
        for value in waveform:
            x = max(-1.0, min(1.0, float(value)))
            companded = math.copysign(math.log1p(mu * abs(x)) / math.log1p(mu), x)
            quantized = int(((companded + 1) / 2) * mu + 0.5)
            out.append(max(0, min(mu, quantized)))
        return out

    def decode_tokens(self, tokens: Sequence[int]) -> List[float]:
        mu = self.quantization_levels - 1
        out: List[float] = []
        for token in tokens:
            y = 2 * (int(token) / mu) - 1
            value = math.copysign((1 / mu) * ((1 + mu) ** abs(y) - 1), y)
            out.append(max(-1.0, min(1.0, value)))
        return out


class LocalTokenModel:
    def __init__(self, order: int = 8):
        if order < 1:
            raise ValueError("order debe ser >= 1")
        self.order = order
        self._counts: Dict[State, Counter[int]] = defaultdict(Counter)
        self._unigram: Counter[int] = Counter()

    def fit(self, sequences: Iterable[Sequence[int]]) -> None:
        for seq in sequences:
            self._unigram.update(seq)
            if len(seq) <= self.order:
                continue
            for idx in range(self.order, len(seq)):
                state = tuple(seq[idx - self.order : idx])
                self._counts[state][int(seq[idx])] += 1

    def generate(self, seed_tokens: Sequence[int], num_tokens: int, temperature: float = 1.0, top_k: int = 64) -> List[int]:
        if not self._unigram:
            raise RuntimeError("Modelo vacío: entrena primero")
        result = list(seed_tokens) or [self._sample(self._unigram, temperature, top_k)]
        while len(result) < num_tokens:
            state = tuple(result[-self.order :])
            counter = self._counts.get(state, self._unigram)
            result.append(self._sample(counter, temperature, top_k))
        return result[:num_tokens]

    @staticmethod
    def _sample(counter: Counter[int], temperature: float, top_k: int) -> int:
        items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        if top_k > 0:
            items = items[:top_k]
        total = sum(freq for _, freq in items)
        scaled = [(token, (freq / total) ** (1 / max(temperature, 1e-6))) for token, freq in items]
        z = sum(v for _, v in scaled)
        target = random.random() * z
        acc = 0.0
        for token, value in scaled:
            acc += value
            if acc >= target:
                return token
        return scaled[-1][0]

    def save(self, path: str | Path) -> None:
        payload = {
            "order": self.order,
            "counts": {"|".join(map(str, k)): dict(v) for k, v in self._counts.items()},
            "unigram": dict(self._unigram),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LocalTokenModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(order=int(payload["order"]))
        model._counts = defaultdict(Counter)
        for key, value in payload["counts"].items():
            state = tuple(int(x) for x in key.split("|"))
            model._counts[state] = Counter({int(k): int(v) for k, v in value.items()})
        model._unigram = Counter({int(k): int(v) for k, v in payload["unigram"].items()})
        return model


class LocalMusicPipeline:
    def __init__(self, config: LocalTrainingConfig | None = None):
        self.config = config or LocalTrainingConfig()
        self.tokenizer = MuLawTokenizer(self.config.quantization_levels)

    def wav_to_tokens(self, wav_path: str | Path) -> List[int]:
        samples, _ = self._read_wav_mono(wav_path)
        return self.tokenizer.encode_waveform(samples)

    def tokens_to_wav(self, tokens: Sequence[int], output_path: str | Path) -> Path:
        waveform = self.tokenizer.decode_tokens(tokens)
        self._write_wav_mono(output_path, waveform, self.config.sample_rate)
        return Path(output_path)

    def build_training_corpus(self, wav_dir: str | Path, output_jsonl: str | Path) -> Path:
        wav_dir = Path(wav_dir)
        output_jsonl = Path(output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        with output_jsonl.open("w", encoding="utf-8") as f:
            for wav_file in list(wav_dir.rglob("*.wav")):
                tokens = self.wav_to_tokens(wav_file)
                for start in range(0, len(tokens), self.config.chunk_size):
                    chunk = tokens[start : start + self.config.chunk_size]
                    if len(chunk) < self.config.order + 2:
                        continue
                    raw = struct.pack(f"<{len(chunk)}H", *chunk)
                    payload = {"source": str(wav_file), "tokens_b64": base64.b64encode(raw).decode("utf-8")}
                    f.write(json.dumps(payload) + "\n")
        return output_jsonl

    def train_local_model(self, jsonl_path: str | Path, model_out: str | Path) -> Path:
        model = LocalTokenModel(order=self.config.order)
        sequences: List[List[int]] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                raw = base64.b64decode(payload["tokens_b64"])
                count = len(raw) // 2
                seq = list(struct.unpack(f"<{count}H", raw))
                sequences.append(seq)
        model.fit(sequences)
        model.save(model_out)
        return Path(model_out)

    def generate_from_model(self, model_path: str | Path, output_wav: str | Path, seconds: int = 8, temperature: float = 1.0, top_k: int = 64, seed_tokens: Sequence[int] | None = None) -> Path:
        model = LocalTokenModel.load(model_path)
        target_tokens = seconds * self.config.sample_rate
        tokens = model.generate(seed_tokens or [], num_tokens=target_tokens, temperature=temperature, top_k=top_k)
        return self.tokens_to_wav(tokens, output_wav)

    @staticmethod
    def _read_wav_mono(path: str | Path) -> Tuple[List[float], int]:
        with wave.open(str(path), "rb") as w:
            frames = w.readframes(w.getnframes())
            sr = w.getframerate()
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
        if sampwidth != 2:
            raise ValueError("Solo PCM16 soportado")
        all_samples = struct.unpack(f"<{len(frames)//2}h", frames)
        if channels == 1:
            mono = [s / 32768.0 for s in all_samples]
        else:
            mono = []
            for i in range(0, len(all_samples), channels):
                frame = all_samples[i : i + channels]
                mono.append(sum(frame) / (channels * 32768.0))
        return mono, sr

    @staticmethod
    def _write_wav_mono(path: str | Path, samples: Sequence[float], sample_rate: int) -> None:
        pcm = [int(max(-1.0, min(1.0, s)) * 32767.0) for s in samples]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
