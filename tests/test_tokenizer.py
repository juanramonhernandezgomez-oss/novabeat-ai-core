import pytest

from novabeat_ai_core.tokenizer import EventToken, MidiTokenizer

mido = pytest.importorskip("mido")


def test_tokens_to_midi(tmp_path):
    tokenizer = MidiTokenizer(ticks_bin=10)
    tokens = [
        EventToken("note_on", 60),
        EventToken("time_shift", 4),
        EventToken("note_off", 60),
    ]

    output = tmp_path / "sample.mid"
    tokenizer.tokens_to_midi(tokens, output)

    midi = mido.MidiFile(output)
    msgs = [msg for msg in midi.tracks[0] if not msg.is_meta]
    assert msgs[0].type == "note_on"
    assert msgs[1].type == "note_off"
    assert msgs[1].time == 40
