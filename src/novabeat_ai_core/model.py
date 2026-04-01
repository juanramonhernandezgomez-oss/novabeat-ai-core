from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .tokenizer import EventToken

State = Tuple[str, ...]


@dataclass
class NGramMusicModel:
    order: int = 4

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order debe ser >= 1")
        self._counts: Dict[State, Counter[str]] = defaultdict(Counter)
        self._unigram: Counter[str] = Counter()

    def fit(self, sequences: Iterable[Sequence[EventToken]]) -> None:
        for seq in sequences:
            serialized = [token.serialize() for token in seq]
            self._unigram.update(serialized)

            if len(serialized) <= self.order:
                continue

            for idx in range(self.order, len(serialized)):
                state = tuple(serialized[idx - self.order : idx])
                nxt = serialized[idx]
                self._counts[state][nxt] += 1

    def generate(
        self,
        seed_tokens: Sequence[EventToken] | None = None,
        num_events: int = 256,
        temperature: float = 1.0,
    ) -> List[EventToken]:
        if num_events <= 0:
            raise ValueError("num_events debe ser > 0")
        if temperature <= 0:
            raise ValueError("temperature debe ser > 0")
        if not self._unigram:
            raise RuntimeError("Modelo vacío: entrena primero con fit()")

        result = [t.serialize() for t in (seed_tokens or [])]
        if len(result) < self.order:
            pad = self._sample_from_counter(self._unigram, temperature)
            while len(result) < self.order:
                result.append(pad)

        while len(result) < num_events:
            state = tuple(result[-self.order :])
            options = self._counts.get(state)

            if not options:
                nxt = self._sample_from_counter(self._unigram, temperature)
            else:
                nxt = self._sample_from_counter(options, temperature)
            result.append(nxt)

        return [EventToken.deserialize(raw) for raw in result[:num_events]]

    @staticmethod
    def _sample_from_counter(counter: Counter[str], temperature: float) -> str:
        items = list(counter.items())
        total = sum(freq for _, freq in items)
        probs = [(token, (freq / total) ** (1 / temperature)) for token, freq in items]
        z = sum(p for _, p in probs)
        threshold = random.random() * z

        acc = 0.0
        for token, p in probs:
            acc += p
            if acc >= threshold:
                return token
        return probs[-1][0]

    def save(self, path: str | Path) -> None:
        payload = {
            "order": self.order,
            "counts": {
                "|||".join(state): dict(counter) for state, counter in self._counts.items()
            },
            "unigram": dict(self._unigram),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NGramMusicModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(order=int(payload["order"]))
        model._counts = defaultdict(Counter)
        for state, counter in payload["counts"].items():
            model._counts[tuple(state.split("|||"))] = Counter(counter)
        model._unigram = Counter(payload["unigram"])
        return model
