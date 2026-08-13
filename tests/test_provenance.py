import json


def test_download_defaults_are_official_and_revision_pinned():
    bf16 = open("scripts/02-fetch-model.sh").read()
    gguf = open("scripts/gguf-quickstart.sh").read()
    helper = open("scripts/hf_parallel_get.py").read()
    assert 'HF_ENDPOINT:-https://huggingface.co' in bf16
    assert 'HF_ENDPOINT:-https://huggingface.co' in gguf
    assert 'os.environ.get("HF_ENDPOINT", "https://huggingface.co")' in helper
    assert '--revision "$MODEL_REVISION"' in bf16
    assert '--revision "$GGUF_REVISION"' in gguf


def test_llama_cpp_default_is_manifest_pin_not_head():
    stack = json.load(open("configs/validated-stack.json"))
    source = open("scripts/gguf-quickstart.sh").read()
    assert len(stack["llama_cpp"]["commit"]) == 40
    assert "LLAMA_CPP_REF" in source
    assert "checkout --detach FETCH_HEAD" in source
    assert "clone --depth 1 https://github.com/ggml-org/llama.cpp" not in source
    assert 'echo "llama.cpp commit: $ACTUAL_LLAMA_CPP_COMMIT"' in source


def test_fetch_scripts_verify_validated_artifacts():
    assert "verify_artifacts.py bf16" in open("scripts/02-fetch-model.sh").read()
    assert "verify_artifacts.py" in open("scripts/gguf-quickstart.sh").read()


def test_parallel_downloader_persists_resume_state_atomically():
    source = open("scripts/hf_parallel_get.py").read()
    assert "threading.RLock()" in source
    assert "os.replace(tmp_meta, meta_path)" in source


def test_parallel_downloader_rejects_parent_paths(tmp_path):
    import os
    import sys
    import pytest

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from hf_parallel_get import download_file

    with pytest.raises(ValueError, match="unsafe artifact path"):
        download_file("https://huggingface.co", "repo", "rev", "../escape",
                      str(tmp_path), 1)
