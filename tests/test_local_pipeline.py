import math

from novabeat_ai_core.local_pipeline import LocalMusicPipeline, LocalTokenModel, LocalTrainingConfig, MuLawTokenizer


def test_mulaw_roundtrip_shape():
    tok = MuLawTokenizer(quantization_levels=256)
    waveform = [math.sin(i / 20.0) for i in range(1024)]
    tokens = tok.encode_waveform(waveform)
    decoded = tok.decode_tokens(tokens)
    assert len(tokens) == 1024
    assert len(decoded) == len(waveform)


def test_local_model_fit_save_load_generate(tmp_path):
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    model = LocalTokenModel(order=3)
    model.fit([seq, seq])

    out = tmp_path / "local_model.json"
    model.save(out)
    loaded = LocalTokenModel.load(out)

    generated = loaded.generate(seed_tokens=[1, 2, 3], num_tokens=20, temperature=1.0, top_k=4)
    assert len(generated) == 20


def test_pipeline_wav_io_and_train(tmp_path):
    cfg = LocalTrainingConfig(order=4, sample_rate=8000, chunk_size=256)
    pipeline = LocalMusicPipeline(cfg)

    waveform = [0.2 * math.sin(2 * math.pi * 220 * i / cfg.sample_rate) for i in range(cfg.sample_rate)]

    wav_path = tmp_path / "sample.wav"
    pipeline._write_wav_mono(wav_path, waveform, cfg.sample_rate)

    corpus = pipeline.build_training_corpus(tmp_path, tmp_path / "corpus.jsonl")
    model_path = pipeline.train_local_model(corpus, tmp_path / "trained.json")
    generated = pipeline.generate_from_model(model_path, tmp_path / "generated.wav", seconds=1)

    assert corpus.exists()
    assert model_path.exists()
    assert generated.exists()
