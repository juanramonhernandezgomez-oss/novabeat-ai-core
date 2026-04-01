from novabeat_ai_core.model import NGramMusicModel
from novabeat_ai_core.tokenizer import EventToken


def test_fit_generate_and_reload(tmp_path):
    seq = [
        EventToken("note_on", 60),
        EventToken("time_shift", 2),
        EventToken("note_off", 60),
        EventToken("note_on", 64),
        EventToken("time_shift", 2),
        EventToken("note_off", 64),
    ]

    model = NGramMusicModel(order=2)
    model.fit([seq, seq])

    generated = model.generate(num_events=12, temperature=1.0)
    assert len(generated) == 12
    assert all(token.kind in {"note_on", "note_off", "time_shift"} for token in generated)

    save_path = tmp_path / "model.json"
    model.save(save_path)
    loaded = NGramMusicModel.load(save_path)
    regenerated = loaded.generate(num_events=10)
    assert len(regenerated) == 10
