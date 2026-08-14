"""Doc-content assertions for the UX-audit docs cluster (Fix-F).

Mirrors the existing doc-content test pattern (see
``test_dflash_nmax.test_troubleshooting_states_the_real_cap`` and
``test_llama_checkout_guard.test_troubleshooting_has_dirty_llama_checkout_entry``):
the docs promise behavior the scripts rely on, so pin the promise.

F-06/F-07/F-08 - the Quick start must carry the user past a bare server URL:
second-terminal guidance, a verified health + completion request pair, and the
reasoning-first `max_tokens` warning that keeps a first completion non-empty.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _src(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _quickstart(readme: str) -> str:
    return readme.split("## Quick start", 1)[1].split("\n## ", 1)[0]


def test_quickstart_tells_the_user_about_the_foreground_server() -> None:
    quickstart = _quickstart(_src("README.md"))
    assert "# OpenAI-compatible server: http://127.0.0.1:8080" in quickstart
    # F-06: the serving command never returns; say so instead of implying
    # "now open this URL" in the occupied terminal.
    assert "Ctrl-C stops the server" in quickstart
    assert "second terminal" in quickstart


def test_quickstart_has_a_verify_it_works_block() -> None:
    quickstart = _quickstart(_src("README.md"))
    assert "### Verify it works" in quickstart
    # F-07: both verified endpoints, with the observed health body shown.
    assert "curl http://127.0.0.1:8080/health" in quickstart
    assert '{"status":"ok"}' in quickstart
    assert "curl http://127.0.0.1:8080/v1/chat/completions" in quickstart
    # F-07+F-08: the example request must budget for hidden reasoning tokens.
    assert '"max_tokens":512' in quickstart


def test_quickstart_documents_reasoning_first_behavior() -> None:
    quickstart = _quickstart(_src("README.md"))
    # F-08: reasoning-first is disclosed near the verify block, including the
    # empty-content-at-small-max_tokens trap and the remedy.
    assert "reasoning-first" in quickstart
    assert "reasoning_content" in quickstart
    assert 'finish_reason:"length"' in quickstart
    assert "max_tokens` ≥ 512" in quickstart
    assert "troubleshooting.md#reasoning-length" in quickstart
