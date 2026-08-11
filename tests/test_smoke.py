"""Live smoke tests against the running vLLM server (mark: server). Run with
the server up:  uv run --no-sync pytest tests/test_smoke.py -v -m server"""
import pytest
import requests

pytestmark = pytest.mark.server
BASE = "http://127.0.0.1:8000"


def test_lists_model():
    r = requests.get(f"{BASE}/v1/models", timeout=30)
    assert r.status_code == 200
    assert any(m["id"] == "muse-glimmer" for m in r.json()["data"])


def test_chat_roundtrip():
    # Muse-Glimmer is a reasoning model: it emits the chain-of-thought in the
    # `reasoning` channel first, then the answer in `content`. Give it enough
    # tokens to finish reasoning and actually answer (16 would hit finish=length
    # mid-reasoning and leave content empty).
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "muse-glimmer",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 512,
    }, timeout=180)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"].strip()
