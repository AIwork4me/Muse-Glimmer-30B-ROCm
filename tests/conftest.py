import socket

import pytest

# Markers `gpu` and `server` are declared in pyproject.toml [tool.pytest.ini_options].
# Local:  uv run pytest -m gpu        CI:  uv run pytest -m "not gpu and not server"

BASE_URL = "http://127.0.0.1:8000"
_HOST, _PORT = "127.0.0.1", 8000


def _server_up() -> bool:
    """True if the vLLM OpenAI server is accepting connections on :8000."""
    try:
        with socket.create_connection((_HOST, _PORT), timeout=1):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip `server`-marked tests when no server is listening, so a bare
    `uv run pytest` is green out-of-the-box. The tests still RUN (not skip) once
    `scripts/03-serve-vllm.sh` is up — this only short-circuits the
    connection-refused failures seen when the server isn't running."""
    if _server_up():
        return
    skip = pytest.mark.skip(
        reason=f"no server at {BASE_URL} (start scripts/03-serve-vllm.sh)")
    for item in items:
        if any(m.name == "server" for m in item.iter_markers()):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL
