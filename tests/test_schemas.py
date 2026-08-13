"""Validate authoritative manifests and immutable historical benchmark cells."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(instance: Path, schema_name: str):
    schema = read_json(SCHEMAS / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(
        read_json(instance)
    )


def test_all_committed_schemas_are_valid():
    schemas = sorted(SCHEMAS.glob("*.schema.json"))
    assert schemas
    for path in schemas:
        Draft202012Validator.check_schema(read_json(path))


def test_reference_manifests_match_schemas():
    validate(ROOT / "configs/validated-stack.json", "validated-stack.schema.json")
    validate(ROOT / "configs/artifact-manifest.json", "artifact-manifest.schema.json")
    validate(ROOT / "configs/public-claims.json", "public-claims.schema.json")


def test_historical_rocm_7_2_1_cells_match_v1_schema():
    # Intentionally do not recurse: an active forward-validation track uses a
    # different directory and must not be coupled to this historical v1 schema.
    cells = sorted((ROOT / "docs/results/matrix").glob("cell-*.json"))
    assert len(cells) == 21
    for cell in cells:
        validate(cell, "benchmark-cell-v1.schema.json")


def test_hardware_submission_schema_accepts_documented_shape():
    sample = {
        "schema_version": 1,
        "hardware": {
            "gpu": "Example Radeon",
            "gfx_target": "gfx1100",
            "memory": "24 GiB dedicated VRAM",
        },
        "software": {
            "rocm": "example",
            "kernel": "example",
            "python": "3.12",
            "pytorch": "example",
            "llama_cpp_commit": "0" * 40,
        },
        "model": {
            "repository": "meta-models/Muse-Glimmer-30B-GGUF",
            "revision": "0" * 40,
        },
        "inference": {
            "path": "llama-cpp-gguf",
            "command": "llama-server <record exact command>",
        },
        "result": {"status": "pass"},
        "evidence": {
            "memory_methodology": "recorded process and system counters",
            "artifact_url": "https://example.invalid/evidence",
        },
    }
    schema = read_json(SCHEMAS / "hardware-validation.schema.json")
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(sample)
