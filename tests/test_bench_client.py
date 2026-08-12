"""Tests for scripts/bench_client.py streaming.

Locks the `stream_options.include_usage = true` fix (Task 9 integration): without
it, OpenAI-compatible servers (llama.cpp, vLLM) omit the final usage chunk while
streaming, n_tokens falls back to 1, and agg_tok_s / tpot_median are corrupted.

The fake session gates the usage chunk on the client's include_usage flag —
exactly mirroring real server behavior — so a future refactor that drops
include_usage makes test_stream_one_requests_include_usage red.
"""
import asyncio
import json as jsonlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from bench_client import stream_one


class _FakeContent:
    """Yields bytes SSE lines, mimicking aiohttp response.content."""

    def __init__(self, lines):
        self._lines = [ln.encode() for ln in lines]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeResp:
    def __init__(self, lines):
        self.content = _FakeContent(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        # Real aiohttp raises ClientResponseError on 4xx/5xx; the fake session
        # only ever emits success streams, so this is a no-op.
        return None


class _FakeSession:
    """Records the POST body and emits the usage chunk ONLY when the client set
    stream_options.include_usage (the real server contract). Captures
    `last_body` so tests can assert the wire request shape directly.

    Pass send_usage=False to simulate a client/server pair where the usage chunk
    is NOT sent (the bug the fix guards against)."""

    def __init__(self, completion_tokens=10, n_delta=3, send_usage=True):
        self.last_body = None
        self._completion_tokens = completion_tokens
        self._n_delta = n_delta
        self._send_usage = send_usage

    def post(self, url, json=None, **kw):
        self.last_body = json or {}
        lines = []
        for _ in range(self._n_delta):
            lines.append('data: ' + jsonlib.dumps({"choices": [{"delta": {"content": "x"}}]}) + '\n\n')
        lines.append('data: ' + jsonlib.dumps({"choices": [{"index": 0, "finish_reason": "stop"}]}) + '\n\n')
        # Server-side gating: usage chunk appears iff the client asked for it.
        if self._send_usage and json and json.get("stream_options", {}).get("include_usage") is True:
            lines.append('data: ' + jsonlib.dumps({
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": self._completion_tokens},
            }) + '\n\n')
        lines.append("data: [DONE]\n\n")
        return _FakeResp(lines)


def _chat_payload():
    return {"_endpoint": "chat", "model": "muse-glimmer-30B", "max_tokens": 16,
            "temperature": 0, "top_p": 1.0, "top_k": 0, "seed": 0,
            "messages": [{"role": "user", "content": "hi"}]}


def test_stream_one_requests_include_usage_and_parses_completion_tokens():
    """End-to-end lock: with include_usage set, the usage chunk arrives and
    n_tokens == usage.completion_tokens. Dropping include_usage breaks BOTH the
    body assertion and (via server-side gating) the n_tokens assertion."""
    sess = _FakeSession(completion_tokens=10)
    res = asyncio.run(stream_one(sess, "http://x", _chat_payload()))
    # wire body MUST request usage while streaming
    assert sess.last_body["stream"] is True
    assert sess.last_body["stream_options"]["include_usage"] is True
    # parser pulls completion_tokens from the (present) usage chunk
    assert res["n_tokens"] == 10
    assert res["finish_reason"] == "stop"


def test_stream_one_falls_back_when_usage_chunk_absent():
    """Documents the corruption the fix prevents: if the usage chunk is absent
    (send_usage=False — exactly what real servers do when include_usage is not
    requested), stream_one can only fall back to n_tokens=1. If someone drops
    include_usage from stream_one, the fake server in the FIRST test above
    stops emitting the usage chunk and test_stream_one_requests_include_usage
    goes red on `n_tokens == 10`."""
    sess = _FakeSession(completion_tokens=10, send_usage=False)
    res = asyncio.run(stream_one(sess, "http://x", _chat_payload()))
    assert res["n_tokens"] == 1  # degenerate fallback — the bug signature
