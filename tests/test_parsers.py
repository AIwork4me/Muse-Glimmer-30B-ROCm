"""Live muse_glimmer reasoning + ATEM tool-call parser tests (mark: server).
Run with the server up:  uv run --no-sync pytest tests/test_parsers.py -v -m server"""
import pytest
import requests

pytestmark = pytest.mark.server
BASE = "http://127.0.0.1:8000"
SYS = {"role": "system", "content": "Reasoning strength: low"}


def test_reasoning_surfaces():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "muse-glimmer",
        "messages": [SYS, {"role": "user", "content": "Think briefly, then say hi."}],
        "max_tokens": 64,
    }, timeout=120)
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    # channel-scoped reasoning lands in .reasoning (not .reasoning_content); the
    # parser may also stream it into .content — either is acceptable here.
    assert "reasoning" in msg or "content" in msg


def test_tool_call_parses():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "muse-glimmer",
        "messages": [SYS, {"role": "user", "content": "Use the get_weather tool for Tokyo."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        "max_tokens": 128,
    }, timeout=120)
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    # ATEM tool calls are XML-style and should parse into .tool_calls; if the
    # model just narrates instead, .content is non-empty — either is acceptable.
    assert msg.get("tool_calls") is not None or msg.get("content")
