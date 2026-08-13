import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from verify_artifacts import load_set, verify


def test_artifact_manifest_shape_and_hashes():
    data = json.load(open("configs/artifact-manifest.json"))
    assert data["hash_algorithm"] == "sha256"
    assert set(data["sets"]) == {"bf16", "gguf"}
    for artifact_set in data["sets"].values():
        assert len(artifact_set["revision"]) == 40
        for record in artifact_set["files"]:
            assert record["size_bytes"] > 0
            assert len(record["sha256"]) == 64
            int(record["sha256"], 16)


def test_verify_artifacts_detects_size_and_hash(tmp_path):
    record = load_set("bf16")["files"][0]
    path = tmp_path / record["path"]
    path.write_bytes(b"wrong")
    failures = verify("bf16", tmp_path, [record["path"]])
    assert failures and "size" in failures[0]


def test_validated_stack_references_artifact_manifest():
    stack = json.load(open("configs/validated-stack.json"))
    assert stack["model"]["artifact_manifest"] == "configs/artifact-manifest.json"
    assert len(stack["llama_cpp"]["commit"]) == 40
