import urllib.error

import pytest

from novabeat_ai_core.supabase_store import SupabaseConfig, SupabaseGenerationStore


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_supabase_save_success(monkeypatch, tmp_path):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["data"] = req.data
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    store = SupabaseGenerationStore(
        SupabaseConfig(url="https://demo.supabase.co", key="k", table="generations")
    )
    store.save_generation("local", "prompt", tmp_path / "a.wav", {"a": 1})

    assert captured["url"] == "https://demo.supabase.co/rest/v1/generations"
    assert captured["headers"]["Apikey"] == "k"
    assert captured["data"]


def test_supabase_save_http_error(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    store = SupabaseGenerationStore(SupabaseConfig(url="https://demo.supabase.co", key="k"))
    with pytest.raises(RuntimeError):
        store.save_generation("local", "prompt", tmp_path / "a.wav")
