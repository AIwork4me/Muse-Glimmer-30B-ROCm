import pytest

# Markers `gpu` and `server` are declared in pyproject.toml [tool.pytest.ini_options].
# Local:  uv run pytest -m gpu        CI:  uv run pytest -m "not gpu and not server"

@pytest.fixture(scope="session")
def base_url():
    return "http://127.0.0.1:8000"
