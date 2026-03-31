from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class EventToken:
    kind: str
    value: int

    def serialize(self) -> str:
        return f"{self.kind}:{self.value}"

    @staticmethod
    def deserialize(raw: str) -> "EventToken":
        kind, value = raw.split(":", 1)
        return EventToken(kind=kind, value=int(value))


class MidiTokenizer:
    """Convierte MIDI <-> tokens simbólicos simples."""

    def __init__(self, ticks_bin: int = 30):
        if ticks_bin <= 0:
            raise ValueError("ticks_bin debe ser > 0")
        self.ticks_bin = ticks_bin

    def midi_to_tokens(self, midi_path: str | Path) -> List[EventToken]:
        import mido

        midi = mido.MidiFile(midi_path)
        tokens: List[EventToken] = []

        for track in midi.tracks:
            accumulated_time = 0
            for msg in track:
                accumulated_time += msg.time
                if accumulated_time > 0:
                    shift = max(1, accumulated_time // self.ticks_bin)
                    tokens.append(EventToken("time_shift", int(shift)))
                    accumulated_time = 0

                if msg.type == "note_on" and msg.velocity > 0:
                    tokens.append(EventToken("note_on", msg.note))
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    tokens.append(EventToken("note_off", msg.note))

        return tokens

    def tokens_to_midi(
        self,
        tokens: Iterable[EventToken],
        output_path: str | Path,
        ticks_per_beat: int = 480,
    ) -> None:
        import mido

        midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        pending_time = 0
        for token in tokens:
            if token.kind == "time_shift":
                pending_time += max(0, token.value) * self.ticks_bin
                continue

            if token.kind == "note_on":
                track.append(
                    mido.Message(
                        "note_on",
                        note=max(0, min(127, token.value)),
                        velocity=64,
                        time=pending_time,
                    )
                )
                pending_time = 0
            elif token.kind == "note_off":
                track.append(
                    mido.Message(
                        "note_off",
                        note=max(0, min(127, token.value)),
                        velocity=64,
                        time=pending_time,
                    )
                )
                pending_time = 0

        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.save(output_path)
